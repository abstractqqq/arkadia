use itertools::Itertools;
use ndarray::ArrayView1;
use num::Float;
use crate::{SplitMethod, NB, LeafElement};

#[inline(always)]
pub fn linf_dist<T:Float>(a1: ArrayView1<T>, a2: ArrayView1<T>) -> T {

    let a1 = a1.as_slice().unwrap();
    let a2 = a2.as_slice().unwrap();
    let mut norm = T::zero();
    let m = (a1.len() >> 2) << 2;

    for (x, y) in a1[m..].iter().copied().zip(a2[m..].iter().copied()) {
        norm = norm.max((x-y).abs())
    }
    for arr in a1.iter().copied().zip(a2.iter().copied()).array_chunks::<4>() {
        for (x, y) in arr {
            norm = norm.max((x-y).abs())
        }
    }
    norm

}

/// Kd Tree with L Infinity as metric on the k-dimensional space
pub struct LIKdtree<'a, T: Float + 'static, A> {
    dim: usize,
    // Nodes
    left: Option<Box<LIKdtree<'a, T, A>>>,
    right: Option<Box<LIKdtree<'a, T, A>>>,
    // Is a leaf node if this has values
    split_axis: Option<usize>,
    split_axis_value: Option<T>,
    min_bounds: Vec<T>,
    max_bounds: Vec<T>,
    // Data
    data: Option<&'a [LeafElement<'a, T, A>]>, // Not none when this is a leaf
}

impl<'a, T: Float + 'static, A: Copy> LIKdtree<'a, T, A> {

    pub fn build(data: &'a mut [LeafElement<'a, T, A>], how: SplitMethod) -> Result<Self, String> {
        if data.is_empty() {
            return Err("Empty data.".into());
        }
        let dim = data.last().unwrap().dim();
        let capacity = crate::utils::suggest_capacity(dim);
        Ok(Self::build_unchecked(data, dim, capacity, 0, how))
    }

    pub fn with_capacity(
        data: &'a mut [LeafElement<'a, T, A>],
        capacity: usize,
        how: SplitMethod,
    ) -> Result<Self, String> {
        if data.is_empty() {
            return Err("Empty data.".into());
        }
        let dim = data.last().unwrap().dim();
        if capacity == 0 {
            return Err("Zero capacity.".into());
        }
        Ok(Self::build_unchecked(data, dim, capacity, 0, how))
    }

    fn find_bounds(data: &[LeafElement<'a, T, A>], depth: usize, dim: usize) -> (Vec<T>, Vec<T>) {
        let mut min_bounds = vec![T::max_value(); dim];
        let mut max_bounds = vec![T::min_value(); dim];
        if depth == 0 {
            (min_bounds, max_bounds)
        } else {
            for elem in data.iter() {
                for i in 0..dim {
                    min_bounds[i] = min_bounds[i].min(elem.row_vec[i]);
                    max_bounds[i] = max_bounds[i].max(elem.row_vec[i]);
                }
            }
            (min_bounds, max_bounds)
        }
    }

    fn build_unchecked(
        data: &'a mut [LeafElement<'a, T, A>],
        dim: usize,
        capacity: usize,
        depth: usize,
        how: SplitMethod,
    ) -> Self {
        let n = data.len();
        let (min_bounds, max_bounds) = Self::find_bounds(data, depth, dim);
        if n <= capacity {
            LIKdtree {
                dim: dim,
                left: None,
                right: None,
                split_axis: None,
                split_axis_value: None,
                min_bounds: min_bounds,
                max_bounds: max_bounds,
                data: Some(data),
            }
        } else {
            let axis = depth % dim;
            let (split_axis_value, split_idx) = match how {
                SplitMethod::MIDPOINT => {
                    let midpoint = min_bounds[axis]
                        + (max_bounds[axis] - min_bounds[axis]) / T::from(2.0).unwrap();
                    data.sort_unstable_by(|l1, l2| {
                        (l1.row_vec[axis] >= midpoint).cmp(&(l2.row_vec[axis] >= midpoint))
                    }); // False <<< True. Now split by the first True location
                    let split_idx = data.partition_point(|elem| elem.row_vec[axis] < midpoint); // first index of True. If it doesn't exist, all points goes into left
                    (midpoint, split_idx)
                }
                SplitMethod::MEAN => {
                    let mut sum = T::zero();
                    for row in data.iter() {
                        sum = sum + row.row_vec[axis];
                    }
                    let mean = sum / T::from(n).unwrap();
                    data.sort_unstable_by(|l1, l2| {
                        (l1.row_vec[axis] >= mean).cmp(&(l2.row_vec[axis] >= mean))
                    }); // False <<< True. Now split by the first True location
                    let split_idx = data.partition_point(|elem| elem.row_vec[axis] < mean); // first index of True. If it doesn't exist, all points goes into left
                    (mean, split_idx)
                }
                SplitMethod::MEDIAN => {
                    data.sort_unstable_by(|l1, l2| {
                        l1.row_vec[axis].partial_cmp(&l2.row_vec[axis]).unwrap()
                    }); // False <<< True. Now split by the first True location
                    let half = n >> 1;
                    let split_value = data[half].row_vec[axis];
                    (split_value, half)
                }
            };

            let (left, right) = data.split_at_mut(split_idx);
            LIKdtree {
                dim: dim,
                left: Some(Box::new(Self::build_unchecked(
                    left,
                    dim,
                    capacity,
                    depth + 1,
                    how.clone(),
                ))),
                right: Some(Box::new(Self::build_unchecked(
                    right,
                    dim,
                    capacity,
                    depth + 1,
                    how,
                ))),
                split_axis: Some(axis),
                split_axis_value: Some(split_axis_value),
                min_bounds: min_bounds,
                max_bounds: max_bounds,
                data: None,
            }
        }
    }

    fn is_leaf(&self) -> bool {
        self.data.is_some()
    }

    #[inline(always)]
    // Computes the distance from the closest potential point in box to P
    fn closest_dist_to_box(min_bounds: &[T], max_bounds: &[T], point: ArrayView1<T>) -> T {
        // let mut dist = T::zero();
        let mut cur_max = T::min_value();
        let mut cur_min = T::max_value();
        for i in 0..point.len() {
            if point[i] > max_bounds[i] {
                cur_max = cur_max.max(point[i] - max_bounds[i]);
                cur_min = cur_min.min(point[i] - max_bounds[i]);
                // dist = dist.max((point[i] - max_bounds[i]).abs());
            } else if point[i] < min_bounds[i] {
                cur_max = cur_max.max(point[i] - min_bounds[i]);
                cur_min = cur_min.min(point[i] - min_bounds[i]);
            }
        }
        cur_max.abs().max(cur_min.abs())
    }

    #[inline(always)]
    fn update_top_k(
        &self,
        top_k: &mut Vec<NB<T, A>>,
        k: usize,
        point: ArrayView1<T>,
        max_dist_bound: T,
    ) {
        let max_permissible_dist = T::max_value().min(max_dist_bound);
        // This is only called if is_leaf. Safe to unwrap.
        for element in self.data.unwrap().iter() {
            let cur_max_dist = top_k.last().map(|nb| nb.dist).unwrap_or(max_dist_bound);
            let y = element.row_vec;
            let dist = linf_dist(y, point);
            if dist < cur_max_dist || (top_k.len() < k && dist <= max_permissible_dist) {
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

    #[inline(always)]
    fn update_nb_within(
        &self,
        neighbors: &mut Vec<NB<T, A>>,
        point: ArrayView1<T>,
        radius: T,
    ) {
        // This is only called if is_leaf. Safe to unwrap.
        for element in self.data.unwrap().iter() {
            let y = element.row_vec;
            let dist = linf_dist(y, point);
            if dist <= radius {
                neighbors.push(NB {
                    dist: dist,
                    item: element.item,
                });
            }
        }
    }


    pub fn knn(&self, k: usize, point: ArrayView1<T>, epsilon:T) -> Option<Vec<NB<T, A>>> {
        if k == 0 || (point.len() != self.dim) || (point.iter().any(|x| !x.is_finite())) {
            None
        } else {
            // Always allocate 1 more.
            let mut top_k = Vec::with_capacity(k + 1);
            let mut pending = Vec::with_capacity(k + 1);
            pending.push((T::min_value(), self));
            while !pending.is_empty() {
                Self::knn_one_step(
                    &mut pending,
                    &mut top_k,
                    k,
                    point,
                    T::max_value(),
                    epsilon
                );
            }
            Some(top_k)
        }
    }

    pub fn knn_leaf_elem(
        &self,
        k: usize,
        leaf_element: LeafElement<'a, T, A>,
        epsilon: T
    ) -> Option<Vec<NB<T, A>>> {
        if k == 0 || (leaf_element.dim() != self.dim) || (leaf_element.is_not_finite()) {
            None
        } else {
            // Always allocate 1 more.
            let mut top_k = Vec::with_capacity(k + 1);
            let mut pending = Vec::with_capacity(k + 1);
            pending.push((T::min_value(), self));
            while !pending.is_empty() {
                Self::knn_one_step(
                    &mut pending,
                    &mut top_k,
                    k,
                    leaf_element.row_vec,
                    T::max_value(),
                    epsilon
                );
            }
            Some(top_k)
        }
    }

    // For bounded, epsilon is 0
    pub fn knn_bounded(
        &self,
        k: usize,
        point: ArrayView1<T>,
        max_dist_bound: T,
    ) -> Option<Vec<NB<T, A>>> {
        if k == 0
            || (point.len() != self.dim)
            || (point.iter().any(|x| !x.is_finite()))
            || max_dist_bound <= T::zero() + T::epsilon()
        {
            None
        } else {
            // Always allocate 1 more.
            let mut top_k = Vec::with_capacity(k + 1);
            let mut pending = Vec::with_capacity(k + 1);
            pending.push((T::min_value(), self));
            while !pending.is_empty() {
                Self::knn_one_step(
                    &mut pending,
                    &mut top_k,
                    k,
                    point,
                    max_dist_bound,
                    T::zero()
                );
            }
            Some(top_k)
        }
    }

    pub fn knn_bounded_leaf_elem(
        &self,
        k: usize,
        leaf_element: LeafElement<'a, T, A>,
        max_dist_bound: T,
    ) -> Option<Vec<NB<T, A>>> {
        if k == 0
            || (leaf_element.dim() != self.dim)
            || (leaf_element.is_not_finite())
            || max_dist_bound <= T::zero() + T::epsilon()
        {
            None
        } else {
            // Always allocate 1 more.
            let mut top_k = Vec::with_capacity(k + 1);
            let mut pending = Vec::with_capacity(k + 1);
            pending.push((T::min_value(), self));
            while !pending.is_empty() {
                Self::knn_one_step(
                    &mut pending,
                    &mut top_k,
                    k,
                    leaf_element.row_vec,
                    max_dist_bound,
                    T::zero()
                );
            }
            Some(top_k)
        }
    }

    pub fn knn_one_step(
        pending: &mut Vec<(T, &LIKdtree<'a, T, A>)>,
        top_k: &mut Vec<NB<T, A>>,
        k: usize,
        point: ArrayView1<T>,
        max_dist_bound: T,
        epsilon: T,
    ) {
        let current_max = if top_k.len() < k {
            max_dist_bound
        } else {
            top_k.last().unwrap().dist
        };
        let (dist_to_box, tree) = pending.pop().unwrap(); // safe
        if dist_to_box > current_max {
            return;
        }
        let mut current = tree;
        while !current.is_leaf() {
            let split_axis = current.split_axis.unwrap();
            let axis_value = current.split_axis_value.unwrap();
            let next = if point[split_axis] < axis_value {
                let next = current.right.as_ref().unwrap().as_ref();
                current = current.left.as_ref().unwrap().as_ref();
                next
            } else {
                let next = current.left.as_ref().unwrap().as_ref();
                current = current.right.as_ref().unwrap().as_ref();
                next
            };
            let dist_to_box = Self::closest_dist_to_box(
                next.min_bounds.as_ref(),
                next.max_bounds.as_ref(),
                point,
            ); // (the next Tree, min dist from the box to point)
            if dist_to_box + epsilon < current_max {
                pending.push((dist_to_box, next));
            }
        }
        current.update_top_k(top_k, k, point, max_dist_bound);
    }

    pub fn within(&self, point: ArrayView1<T>, radius: T, sort: bool) -> Option<Vec<NB<T, A>>> {
        // radius is actually squared radius
        if radius <= T::zero() + T::epsilon() || (point.iter().any(|x| !x.is_finite())) {
            None
        } else {
            // Always allocate some.
            let mut neighbors = Vec::with_capacity(32);
            let mut pending = Vec::with_capacity(32);
            pending.push((T::min_value(), self));
            while !pending.is_empty() {
                Self::within_one_step(&mut pending, &mut neighbors, point, radius);
            }
            if sort {
                neighbors.sort_unstable();
            }
            neighbors.shrink_to_fit();
            Some(neighbors)
        }
    }

    pub fn within_leaf_elem(
        &self,
        leaf_element: LeafElement<'a, T, A>,
        radius: T,
        sort: bool,
    ) -> Option<Vec<NB<T, A>>> {
        // radius is actually squared radius
        if radius <= T::zero() + T::epsilon() || (leaf_element.is_not_finite()) {
            None
        } else {
            // Always allocate some.
            let mut neighbors = Vec::with_capacity(32);
            let mut pending = Vec::with_capacity(32);
            pending.push((T::min_value(), self));
            while !pending.is_empty() {
                Self::within_one_step(
                    &mut pending,
                    &mut neighbors,
                    leaf_element.row_vec,
                    radius,
                );
            }
            if sort {
                neighbors.sort_unstable();
            }
            neighbors.shrink_to_fit();
            Some(neighbors)
        }
    }

    pub fn within_count(&self, point: ArrayView1<T>, radius: T) -> Option<u32> {
        // radius is actually squared radius
        if radius <= T::zero() + T::epsilon() || (point.iter().any(|x| !x.is_finite())) {
            None
        } else {
            // Always allocate some.
            let mut cnt = 0u32;
            let mut pending = Vec::with_capacity(32);
            pending.push((T::min_value(), self));
            while !pending.is_empty() {
                cnt += Self::within_count_one_step(&mut pending, point, radius);
            }
            Some(cnt)
        }
    }

    pub fn within_count_leaf_elem(
        &self,
        leaf_element: LeafElement<'a, T, A>,
        radius: T,
    ) -> Option<u32> {
        if radius <= T::zero() + T::epsilon() || (leaf_element.is_not_finite()) {
            None
        } else {
            // Always allocate some.
            let mut cnt = 0u32;
            let mut pending = Vec::with_capacity(32);
            while !pending.is_empty() {
                cnt += Self::within_count_one_step(
                    &mut pending,
                    leaf_element.row_vec,
                    radius,
                );
            }
            Some(cnt)
        }
    }

    fn within_one_step(
        pending: &mut Vec<(T, &LIKdtree<'a, T, A>)>,
        neighbors: &mut Vec<NB<T, A>>,
        point: ArrayView1<T>,
        radius: T,
    ) {
        let (dist_to_box, tree) = pending.pop().unwrap(); // safe
        if dist_to_box > radius {
            return;
        }
        let mut current = tree;
        while !current.is_leaf() {
            let split_axis = current.split_axis.unwrap();
            let axis_value = current.split_axis_value.unwrap();
            let next = if point[split_axis] < axis_value {
                let next = current.right.as_ref().unwrap().as_ref();
                current = current.left.as_ref().unwrap().as_ref();
                next
            } else {
                let next = current.left.as_ref().unwrap().as_ref();
                current = current.right.as_ref().unwrap().as_ref();
                next
            };
            let dist_to_box = Self::closest_dist_to_box(
                next.min_bounds.as_ref(),
                next.max_bounds.as_ref(),
                point,
            ); // (the next Tree, min dist from the box to point)
            if dist_to_box <= radius {
                pending.push((dist_to_box, next));
            }
        }
        current.update_nb_within(neighbors, point, radius);
    }

    fn within_count_one_step(
        pending: &mut Vec<(T, &LIKdtree<'a, T, A>)>,
        point: ArrayView1<T>,
        radius: T,
    ) -> u32 {
        let (dist_to_box, tree) = pending.pop().unwrap(); // safe
        if dist_to_box > radius {
            0
        } else {
            let mut current = tree;
            while !current.is_leaf() {
                let split_axis = current.split_axis.unwrap();
                let axis_value = current.split_axis_value.unwrap();
                let next = if point[split_axis] < axis_value {
                    let next = current.right.as_ref().unwrap().as_ref();
                    current = current.left.as_ref().unwrap().as_ref();
                    next
                } else {
                    let next = current.left.as_ref().unwrap().as_ref();
                    current = current.right.as_ref().unwrap().as_ref();
                    next
                };
                let dist_to_box = Self::closest_dist_to_box(
                    next.min_bounds.as_ref(),
                    next.max_bounds.as_ref(),
                    point,
                ); // (the next Tree, min dist from the box to point)
                if dist_to_box <= radius {
                    pending.push((dist_to_box, next));
                }
            }
            // Return the count in current
            current.data.unwrap().iter().fold(0u32, |acc, element| {
                let y = element.row_vec;
                let dist = linf_dist(y, point);
                acc + (dist <= radius) as u32
            })
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, Array1, Array2};

    fn linf_dist_slice(a1: &[f64], a2: &[f64]) -> f64 {
        a1.iter().zip(a2.iter()).fold(0., |acc, (x, y)| acc.max((x - y).abs()))
    }

    fn random_10d_rows() -> [f64; 10] {
        rand::random()
    }

    #[test]
    fn test_linf() {

        for _ in 0..100 {
            let v1 = random_10d_rows();
            let v2 = random_10d_rows();
            let a1 = Array1::from_vec(v1.to_vec());
            let a2 = Array1::from_vec(v2.to_vec());
            let result = linf_dist(a1.view(), a2.view());
            let answer = linf_dist_slice(&v1, &v2);
            assert!(answer == result)
        }
    }

    #[test]
    fn test_10d_knn_linf_dist_midpoint() {
        // 10 nearest neighbors, matrix of size 1000 x 10
        let k = 10usize;
        let mut v = Vec::new();
        let rows = 1_000usize;
        for _ in 0..rows {
            v.extend_from_slice(&random_10d_rows());
        }

        let mat = Array2::from_shape_vec((rows, 10), v).unwrap();
        let mat = mat.as_standard_layout().to_owned();
        let point = arr1(&[0.5; 10]);
        // brute force test
        let mut ans_distances = mat
            .rows()
            .into_iter()
            .map(|v| linf_dist_slice(v.to_slice().unwrap(), &point.to_vec()))
            .collect::<Vec<_>>();
        let mut ans_argmins = (0..rows).collect::<Vec<_>>();
        ans_argmins.sort_by(|&i, &j| ans_distances[i].partial_cmp(&ans_distances[j]).unwrap());
        ans_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let values = (0..rows).collect::<Vec<_>>();
        let binding = mat.view();
        let mut leaf_elements = crate::utils::matrix_to_leaf_elements_linf(&binding, &values);

        let tree = LIKdtree::build(&mut leaf_elements, SplitMethod::MIDPOINT).unwrap();

        let output = tree.knn(k, point.view(), 0f64);

        assert!(output.is_some());
        let output = output.unwrap();
        let indices = output.iter().map(|nb| nb.item).collect::<Vec<_>>();
        let distances = output.iter().map(|nb| nb.dist).collect::<Vec<_>>();

        assert_eq!(&ans_argmins[..k], &indices);
        for (d1, d2) in ans_distances[..k].iter().zip(distances.into_iter()) {
            assert!((d1 - d2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_10d_knn_linf_dist_mean() {
        // 10 nearest neighbors, matrix of size 1000 x 10
        let k = 10usize;
        let mut v = Vec::new();
        let rows = 1_000usize;
        for _ in 0..rows {
            v.extend_from_slice(&random_10d_rows());
        }

        let mat = Array2::from_shape_vec((rows, 10), v).unwrap();
        let mat = mat.as_standard_layout().to_owned();
        let point = arr1(&[0.5; 10]);
        // brute force test
        let mut ans_distances = mat
            .rows()
            .into_iter()
            .map(|v| linf_dist_slice(v.to_slice().unwrap(), &point.to_vec()))
            .collect::<Vec<_>>();
        let mut ans_argmins = (0..rows).collect::<Vec<_>>();
        ans_argmins.sort_by(|&i, &j| ans_distances[i].partial_cmp(&ans_distances[j]).unwrap());
        ans_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let values = (0..rows).collect::<Vec<_>>();
        let binding = mat.view();
        let mut leaf_elements = crate::utils::matrix_to_leaf_elements_linf(&binding, &values);

        let tree = LIKdtree::build(&mut leaf_elements, SplitMethod::MEAN).unwrap();

        let output = tree.knn(k, point.view(), 0f64);

        assert!(output.is_some());
        let output = output.unwrap();
        let indices = output.iter().map(|nb| nb.item).collect::<Vec<_>>();
        let distances = output.iter().map(|nb| nb.dist).collect::<Vec<_>>();

        assert_eq!(&ans_argmins[..k], &indices);
        for (d1, d2) in ans_distances[..k].iter().zip(distances.into_iter()) {
            assert!((d1 - d2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_10d_knn_linf_dist_median() {
        // 10 nearest neighbors, matrix of size 1000 x 10
        let k = 10usize;
        let mut v = Vec::new();
        let rows = 1_000usize;
        for _ in 0..rows {
            v.extend_from_slice(&random_10d_rows());
        }

        let mat = Array2::from_shape_vec((rows, 10), v).unwrap();
        let mat = mat.as_standard_layout().to_owned();
        let point = arr1(&[0.5; 10]);
        // brute force test
        let mut ans_distances = mat
            .rows()
            .into_iter()
            .map(|v| linf_dist_slice(v.to_slice().unwrap(), &point.to_vec()))
            .collect::<Vec<_>>();
        let mut ans_argmins = (0..rows).collect::<Vec<_>>();
        ans_argmins.sort_by(|&i, &j| ans_distances[i].partial_cmp(&ans_distances[j]).unwrap());
        ans_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let values = (0..rows).collect::<Vec<_>>();
        let binding = mat.view();
        let mut leaf_elements = crate::utils::matrix_to_leaf_elements_linf(&binding, &values);

        let tree = LIKdtree::build(&mut leaf_elements, SplitMethod::MEDIAN).unwrap();

        let output = tree.knn(k, point.view(), 0f64);

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