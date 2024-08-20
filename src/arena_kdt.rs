/// Arena based Kdtree
/// Actually not faster.. Not really..
use std::num::{NonZero, NonZeroUsize};

use crate::{leaf::Leaf, KdLeaf, SpacialQueries, DIST, NB};
use indextree::{Arena, NodeId};

pub struct KdNode<'a, A> {
    split_axis: usize,
    split_axis_value: f64,
    leaves: &'a [Leaf<'a, f64, A>],
    min_bounds: Vec<f64>,
    max_bounds: Vec<f64>,
}

impl<'a, A> KdNode<'a, A> {
    fn is_not_leaf(&self) -> bool {
        self.leaves.is_empty()
    }

    fn from_data_is_leaf(data: &'a [Leaf<'a, f64, A>], min: Vec<f64>, max: Vec<f64>) -> Self {
        // because this is leaf, we can put whatever split axis and value
        Self {
            split_axis: 0,
            split_axis_value: 0.,
            leaves: data,
            min_bounds: min,
            max_bounds: max,
        }
    }
}

pub struct ArenaKdtree<'a, A> {
    dim: usize,
    tree: Arena<KdNode<'a, A>>,
    root: NodeId,
    d: DIST<f64>,
}

impl<'a, A: Copy> ArenaKdtree<'a, A> {
    fn find_bounds(data: &[Leaf<'a, f64, A>], dim: usize) -> (Vec<f64>, Vec<f64>) {
        let mut min_bounds = vec![f64::MAX; dim];
        let mut max_bounds = vec![f64::MIN; dim];

        for elem in data.iter() {
            for i in 0..dim {
                min_bounds[i] = min_bounds[i].min(elem.value_at(i));
                max_bounds[i] = max_bounds[i].max(elem.value_at(i));
            }
        }
        (min_bounds, max_bounds)
    }

    pub fn from_leaves(
        data: &'a mut [Leaf<'a, f64, A>],
        dim: usize,
        capacity: usize,
        d: DIST<f64>,
    ) -> Self {
        // Make sure data is not empty
        let mut arena: Arena<KdNode<'a, A>> = Arena::new();
        Self::from_leaves_unchecked(&mut arena, None, data, dim, 0, capacity);
        let root = arena.get_node_id_at(NonZeroUsize::MIN).unwrap();
        Self {
            dim: dim,
            tree: arena,
            d: d,
            root: root,
        }
    }

    fn from_leaves_unchecked(
        arena: &mut Arena<KdNode<'a, A>>,
        parent: Option<NodeId>,
        data: &'a mut [Leaf<'a, f64, A>],
        dim: usize,
        depth: usize,
        capacity: usize,
    ) {
        let axis = depth % dim;
        let n = data.len();
        let (min_bounds, max_bounds) = Self::find_bounds(data, dim);
        if n <= capacity {
            if let Some(p) = parent {
                p.append_value(
                    KdNode::from_data_is_leaf(data, min_bounds, max_bounds),
                    arena,
                );
            } else {
                arena.new_node(KdNode::from_data_is_leaf(data, min_bounds, max_bounds));
            }
        } else {
            let midpoint = min_bounds[axis] + (max_bounds[axis] - min_bounds[axis]) / 2.0;

            data.sort_unstable_by_key(|leaf| leaf.value_at(axis) >= midpoint);
            let split_idx = data.partition_point(|elem| elem.value_at(axis) < midpoint);

            let (left, right) = data.split_at_mut(split_idx);
            if left.is_empty() {
                if let Some(p) = parent {
                    p.append_value(
                        KdNode::from_data_is_leaf(right, min_bounds, max_bounds),
                        arena,
                    );
                } else {
                    arena.new_node(KdNode::from_data_is_leaf(right, min_bounds, max_bounds));
                }
            } else {
                let new_parent = arena.new_node(KdNode {
                    leaves: &[],
                    split_axis: axis,
                    split_axis_value: midpoint,
                    min_bounds: min_bounds,
                    max_bounds: max_bounds,
                });
                if let Some(p) = parent {
                    p.append(new_parent, arena);
                }
                Self::from_leaves_unchecked(
                    arena,
                    Some(new_parent),
                    left,
                    dim,
                    depth + 1,
                    capacity,
                );
                Self::from_leaves_unchecked(
                    arena,
                    Some(new_parent),
                    right,
                    dim,
                    depth + 1,
                    capacity,
                );
            }
        }
    }

    #[inline(always)]
    fn closest_dist_to_box(min_bounds: &[f64], max_bounds: &[f64], point: &[f64]) -> f64 {
        let mut dist = 0.;
        for i in 0..point.len() {
            if point[i] > max_bounds[i] {
                dist = dist + (point[i] - max_bounds[i]).powi(2);
            } else if point[i] < min_bounds[i] {
                dist = dist + (point[i] - min_bounds[i]).powi(2);
            }
        }
        dist
    }

    #[inline(always)]
    fn update_top_k(
        &self,
        node: &KdNode<A>,
        top_k: &mut Vec<NB<f64, A>>,
        k: usize,
        point: &[f64],
        max_dist_bound: f64,
    ) {
        let max_permissible_dist = max_dist_bound;
        // This is only called if is_leaf. Safe to unwrap.
        for element in node.leaves.iter() {
            let cur_max_dist = top_k.last().map(|nb| nb.dist).unwrap_or(max_dist_bound);
            let y = element.row_vec;
            let dist = self.d.dist(y, point);
            if dist <= max_permissible_dist && (dist < cur_max_dist || top_k.len() < k) {
                let nb = NB {
                    dist: dist,
                    item: element.item,
                };
                let idx: usize = top_k.partition_point(|s| s <= &nb);
                if idx < top_k.len() {
                    if top_k.len() + 1 > k {
                        top_k.pop();
                    }
                    top_k.insert(idx, nb);
                } else if top_k.len() < k {
                    top_k.push(nb);
                }
            }
        }
        // You can find code comments in arkadia.rs
    }

    fn knn_one_step(
        &self,
        pending: &mut Vec<(f64, NodeId)>,
        top_k: &mut Vec<NB<f64, A>>,
        k: usize,
        point: &[f64],
        max_dist_bound: f64,
        epsilon: f64,
    ) {
        // k > 0 is guaranteed.
        let current_max = if top_k.len() < k {
            max_dist_bound
        } else {
            top_k.last().unwrap().dist
        };
        let (dist_to_box, id) = pending.pop().unwrap(); // safe
        if dist_to_box > current_max {
            return;
        }
        let mut current = self.tree.get(id).unwrap(); // leaves will have data
        while current.get().is_not_leaf() {
            let split_axis = current.get().split_axis;
            let axis_value = current.get().split_axis_value;
            let (next, next_id) = if point[split_axis] < axis_value {
                let next_id = current.last_child().unwrap();
                let next = self.tree.get(next_id).unwrap();
                current = self.tree.get(current.first_child().unwrap()).unwrap();
                (next, next_id)
            } else {
                let next_id = current.first_child().unwrap();
                let next = self.tree.get(next_id).unwrap();
                current = self.tree.get(current.last_child().unwrap()).unwrap();
                (next, next_id)
            };

            let dist_to_box =
                Self::closest_dist_to_box(&next.get().min_bounds, &next.get().max_bounds, point); // (min dist from the box to point, the next Tree)
            if dist_to_box + epsilon < current_max {
                pending.push((dist_to_box, next_id));
            }
        }
        self.update_top_k(current.get(), top_k, k, point, max_dist_bound);
    }

    pub fn knn(&self, k: usize, point: &[f64], epsilon: f64) -> Option<Vec<NB<f64, A>>> {
        if k == 0 || (point.len() != self.dim) || (point.iter().any(|x| !x.is_finite())) {
            None
        } else {
            // Always allocate 1 more.
            let mut top_k = Vec::with_capacity(k + 1);
            let mut pending = Vec::with_capacity(k + 1);
            pending.push((f64::MIN, self.root));
            while !pending.is_empty() {
                self.knn_one_step(&mut pending, &mut top_k, k, point, f64::MAX, epsilon);
            }
            Some(top_k)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::arena_kdt::ArenaKdtree;

    use super::super::{matrix_to_empty_leaves, matrix_to_leaves};
    use ndarray::{arr1, Array2, ArrayView1, ArrayView2};

    pub fn squared_l2(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .fold(0., |acc, (&a, &b)| acc + (a - b) * (a - b))
    }

    fn random_10d_rows() -> [f64; 10] {
        rand::random()
    }

    fn generate_test_answer(
        mat: ArrayView2<f64>,
        point: ArrayView1<f64>,
        dist_func: fn(&[f64], &[f64]) -> f64,
    ) -> (Vec<usize>, Vec<f64>) {
        let mut ans_distances = mat
            .rows()
            .into_iter()
            .map(|v| dist_func(v.to_slice().unwrap(), &point.to_vec()))
            .collect::<Vec<_>>();

        let mut ans_argmins = (0..mat.nrows()).collect::<Vec<_>>();
        ans_argmins.sort_by(|&i, &j| ans_distances[i].partial_cmp(&ans_distances[j]).unwrap());
        ans_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        (ans_argmins, ans_distances)
    }

    #[test]
    fn test_10d_knn_l2_dist_arena_kdt() {
        // 10 nearest neighbors, matrix of size 1000 x 10
        let k = 10usize;
        let mut v = Vec::new();
        let rows = 5_000usize;
        for _ in 0..rows {
            v.extend_from_slice(&random_10d_rows());
        }

        let mat = Array2::from_shape_vec((rows, 10), v).unwrap();
        let mat = mat.as_standard_layout().to_owned();
        let point = arr1(&[0.5; 10]);
        // brute force test
        let (ans_argmins, ans_distances) =
            generate_test_answer(mat.view(), point.view(), squared_l2);

        let values = (0..rows).collect::<Vec<_>>();
        let binding = mat.view();
        let mut leaves = matrix_to_leaves(&binding, &values);

        let tree = ArenaKdtree::from_leaves(&mut leaves, 10, 40, crate::DIST::SQL2);
        let output = tree.knn(k, point.as_slice().unwrap(), 0f64);

        assert!(output.is_some());
        let output = output.unwrap();
        let indices = output.iter().map(|nb| nb.item).collect::<Vec<_>>();
        let distances = output.iter().map(|nb| nb.dist).collect::<Vec<_>>();

        assert_eq!(&ans_argmins[..k], &indices);
        for (d1, d2) in ans_distances[..k].iter().zip(distances.into_iter()) {
            assert!((d1 - d2).abs() < 1e-10);
        }
    }
}
