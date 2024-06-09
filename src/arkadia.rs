use crate::distances::{squared_euclidean};
use ndarray::{Array1, ArrayView1, ArrayView2};
use num::Float;

pub enum KdtreeError {
    EmptyData,
    NonFiniteValue,
}

// Search Result
// (Data, and distance)
pub struct NB<T:Float, A>{
    pub dist: T,
    pub item: A,
}

impl <T: Float, A> PartialEq for NB<T, A> {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl <T: Float, A> PartialOrd for NB<T, A> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}

impl <T: Float, A> Eq for NB<T, A> {}

impl <T: Float, A> Ord for NB<T, A> {
    
    // Unwrap is safe because in all use cases, the data should not contain any non-finite values.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist.partial_cmp(&other.dist).unwrap()
    }
    
    fn max(self, other: Self) -> Self
    where
        Self: Sized,
    {
        std::cmp::max_by(self, other, |a, b| a.dist.partial_cmp(&b.dist).unwrap())
    }
    
    fn min(self, other: Self) -> Self
    where
        Self: Sized,
    {
        std::cmp::min_by(self, other, |a, b| a.dist.partial_cmp(&b.dist).unwrap())
    }
    
    fn clamp(self, min: Self, max: Self) -> Self
    where
        Self: Sized,
        Self: PartialOrd,
    {
        assert!(min <= max);
        if self.dist < min.dist {
            min
        } else if self.dist > max.dist {
            max
        } else {
            self
        }
    }
}

pub struct LeafElement<'a, T:Float, A> {
    pub data: A, 
    pub row_vec: ArrayView1<'a, T>,
    pub norm: T,
}

pub fn suggest_capacity(dim:usize) -> usize {
    if dim < 5 {
        8
    } else if dim < 12 {
        16
    } else {
        32
    }
}

pub struct Kdtree<'a, T: Float + 'static, A> {
    dim: usize,
    // capacity of leaf node
    capacity: usize,
    // Nodes
    left: Option<Box<Kdtree<'a, T, A>>>,
    right: Option<Box<Kdtree<'a, T, A>>>,
    // Is a leaf node if this has values
    split_axis: Option<usize>,
    split_axis_value: Option<T>,
    min_bounds: Vec<T>,
    max_bounds: Vec<T>,
    // Data
    data: Option<&'a [LeafElement<'a, T, A>]> // Not none when this is a leaf
}

impl<'a, T: Float + 'static, A: Copy> Kdtree<'a, T, A> {

    pub fn build(
        data: &'a mut [LeafElement<'a, T, A>],
        dim: usize,
    ) -> Self {
        let capacity = suggest_capacity(dim);
        Self::build_unchecked(data, dim, capacity, 0)
    }

    fn build_unchecked(
        data: &'a mut [LeafElement<'a, T, A>],
        dim: usize,
        capacity: usize,
        depth: usize,
    ) -> Self {
        let n = data.len();
        let mut min_bounds = vec![T::max_value(); dim];
        let mut max_bounds = vec![T::min_value(); dim];
        for elem in data.iter() {
            for i in 0..dim {
                min_bounds[i] = min_bounds[i].min(elem.row_vec[i]);
                max_bounds[i] = max_bounds[i].max(elem.row_vec[i]);
            }
        }
        if n <= capacity {
            Kdtree {
                dim: dim,
                capacity: capacity,
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
            // let half = n >> 1;
            let min_axis = min_bounds[axis];
            let max_axis = max_bounds[axis];
            let midpoint = min_axis + (max_axis - min_axis) / (T::one() + T::one());
            data.sort_unstable_by(
                |l1, l2| 
                (l1.row_vec[axis] >= midpoint).cmp(&(l2.row_vec[axis] >= midpoint))
            ); // False <<< True. Now split by the first True location

            let split_idx = data.partition_point(
                |elem| elem.row_vec[axis] < midpoint
            ); // first index of True. If it doesn't exist, all points goes into left
            let (left, right) = data.split_at_mut(split_idx);
            Kdtree {
                dim: dim,
                capacity: capacity,
                left: Some(Box::new(Self::build_unchecked(
                    left,
                    dim,
                    capacity,
                    depth + 1
                ))),
                right: Some(Box::new(Self::build_unchecked(
                    right,
                    dim,
                    capacity,
                    depth + 1
                ))),
                split_axis: Some(axis),
                split_axis_value: Some(midpoint),
                min_bounds: min_bounds,
                max_bounds: max_bounds,
                data: None,
            }
        }
    }

    fn is_leaf(&self) -> bool {
        self.data.is_some()
    }

    // Create the closest point in the box bounded by oppo to point and its norm
    fn closest_dist_to_box(min_bounds:&[T], max_bounds:&[T], point: ArrayView1<T>) -> T {
        let mut dist = T::zero();
        for i in 0..point.len() {
            if point[i] > max_bounds[i] {
                dist = dist + (point[i] - max_bounds[i]).powi(2);
            } else if point[i] < min_bounds[i] {
                dist = dist + (point[i] - min_bounds[i]).powi(2);
            }
        }
        dist
    }

    /// Updates the current top K with the incoming nb
    #[inline(always)]
    fn push_to_top_k(top_k: &mut Vec<NB<T, A>>, k:usize, nb: NB<T, A>) {
        let idx: usize = top_k.partition_point(|s| s <= &nb);
        if idx < top_k.len() { 
            if top_k.len() + 1 > k {
                top_k.pop();
            } // This ensures top_k has k elements and no need to allocate
            top_k.insert(idx, nb);
        } else if top_k.len() < k {
            // Note if idx < top_k.len() is false, then index == top_k.len() for free!
            top_k.push(nb); // by the end of the push, len is at most k.
        } // Do nothing if idx >= k, because that means no partition point was found within existing values
    }

    #[inline(always)]
    fn update_top_k(
        &self, 
        top_k: &mut Vec<NB<T, A>>, 
        k:usize, 
        point:ArrayView1<T>,
        point_norm_cache: T,
    ) {
        // This is only called if is_leaf. Safe to unwrap.
        for element in self.data.unwrap().iter() {
            let cur_max_dist = top_k.last().map(|nb| nb.dist).unwrap_or(T::max_value());
            let y = element.row_vec;
            // This hack actually produces faster Euclidean dist calculation (because .dot is unrolled in ndarray)
            let dot_product = y.dot(&point);
            let dist = point_norm_cache + element.norm - dot_product - dot_product;
            if dist < cur_max_dist || top_k.len() < k {
                Self::push_to_top_k(
                    top_k, 
                    k, 
                    NB {
                        dist: dist,
                        item: element.data,
                    }
                );
            }
            // No need to check if we already have k elements and the dist is >= current max dist
        }
    }

    pub fn knn(
        &self, 
        k:usize, 
        point:ArrayView1<T>,
    ) -> Option<Vec<NB<T, A>>> {

        if k == 0 
            || (point.len() != self.dim) 
            || (point.iter().any(|v| !v.is_finite())) 
        {
            None
        } else {
            let point_norm = point.dot(&point);
            let mut top_k = Vec::with_capacity(k+1);
            self.knn_unchecked(
                &mut top_k, 
                k, 
                point, 
                0,
                point_norm,
            );
            Some(top_k)
        }
    }

    pub fn knn_non_recurse(
        &self,
        k:usize, 
        point: ArrayView1<T>,
    ) -> Option<Vec<NB<T,A>>> {

        if k == 0 
            || (point.len() != self.dim) 
            || (point.iter().any(|v| !v.is_finite())) 
        {
            None
        } else {
            let mut top_k = Vec::with_capacity(k+1);
            // min heap. Need to negate the dist in later operations
            let mut pending = Vec::new();
            pending.push((T::min_value(), self));
            let point_norm = point.dot(&point);
            // let mut cur_pending_max_dist = T::zero();
            // let mut cur_max_dist = T::max_value();
            while !pending.is_empty() {
                Self::knn_one_step(&mut pending, &mut top_k, k, point, point_norm);
            }
            Some(top_k)
        }

    }

    pub fn knn_one_step(
        pending:&mut Vec<(T, &Kdtree<'a, T, A>)>,
        top_k:&mut Vec<NB<T, A>>, 
        k: usize,
        point: ArrayView1<T>,
        point_norm_cache: T
    ) {
        let current_max = if top_k.len() < k {
            T::max_value() // replace this with distance bound if doing bounded search
        } else {
            top_k.last().unwrap().dist
        };
        let (dist_to_box, tree) = pending.pop().unwrap();
        if dist_to_box > current_max {
            return
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
                point
            ); // The next Tree, plus the min dist from the box to point
            if dist_to_box < current_max {
                pending.push((dist_to_box, next));
            }
        }     
        current.update_top_k(top_k, k, point, point_norm_cache);
    }

    #[inline(always)]
    pub fn knn_unchecked(
        &self, 
        top_k:&mut Vec<NB<T, A>>, 
        k:usize, 
        point: ArrayView1<T>, 
        depth: usize,
        point_norm_cache: T,
    ) {
        
        // let axis = depth % self.dim;
        if self.is_leaf() {
            self.update_top_k(
                top_k, 
                k, 
                point,
                point_norm_cache
            );
            return
        }
        
        // Must exist
        let axis_value = self.split_axis_value.unwrap();
        let axis = self.split_axis.unwrap();
        let (next, oppo) = if point[axis] < axis_value {
            ( // Next = Self.left, opposite = self.right
                self.left.as_ref().unwrap(),
                self.right.as_ref().unwrap(),
            )
        } else {
            ( // right, left
                self.right.as_ref().unwrap(),
                self.left.as_ref().unwrap(),
            )
        };
        
        next.knn_unchecked(
            top_k, 
            k, 
            point, 
            depth + 1, 
            point_norm_cache
        );
        
        let cur_max_dist = top_k.last().map(|nb| nb.dist).unwrap_or(T::max_value());
        // We can rule out some leaf/branches by some simple math
        // However, these rules are not effective if dim is large. This has something to do with
        // the volume of the ball and the volume of the hypercube in higher dimensions.

        if oppo.is_leaf() {
            let perp_dist = (point[axis] - axis_value).powi(2);
            if cur_max_dist > perp_dist {
                oppo.knn_unchecked(
                    top_k, 
                    k, 
                    point, 
                    depth + 1, 
                    point_norm_cache
                );
            }
        } else {
            let dist_to_box = Self::closest_dist_to_box(
                oppo.min_bounds.as_ref(), 
                oppo.max_bounds.as_ref(), 
                point
            );
            if cur_max_dist > dist_to_box {
                oppo.knn_unchecked(
                    top_k, 
                    k, 
                    point, 
                    depth + 1, 
                    point_norm_cache
                );
            }
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, Array2};

    fn random_10d_rows() -> [f64; 10] {
        rand::random()
    }

    fn random_3d_rows() -> [f64; 3] {
        rand::random()
    }

    #[test]
    fn test_10d_knn() {
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
        let mut ans_distances = mat.rows().into_iter().map(|v| {
            squared_euclidean(v, point.view())
        }).collect::<Vec<_>>();
        let mut ans_argmins = (0..rows).collect::<Vec<_>>();
        ans_argmins.sort_by(|&i, &j| ans_distances[i].partial_cmp(&ans_distances[j]).unwrap());
        ans_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    

        let mut data = 
            mat
            .rows()
            .into_iter()
            .enumerate()
            .filter(|(_, arr)| arr.iter().all(|x| x.is_finite()))
            .map(|(i, arr)| LeafElement{ data: i, row_vec: arr, norm: arr.dot(&arr)})
            .collect::<Vec<_>>();
        let tree = Kdtree::build(
            &mut data,
            mat.ncols(),
        );
    
        let output = tree.knn(k, point.view());
    
        assert!(output.is_some());
        let output = output.unwrap();
        let indices = output.iter().map(|nb| nb.item).collect::<Vec<_>>();
        let distances = output.iter().map(|nb| nb.dist).collect::<Vec<_>>();
    
        assert_eq!(&ans_argmins[..k], &indices);
        for (d1, d2) in ans_distances[..k].iter().zip(distances.into_iter()) {
            assert!((d1-d2).abs() < 1e-10);
        }

    }

}
