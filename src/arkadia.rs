use crate::distances::{squared_euclidean};
use ndarray::{Array1, ArrayView1, ArrayView2};
use num::Float;


pub enum KdtreeError {
    EmptyData,
    NonFiniteValue,
}

// Search ID
// The identification of a point within the Kd-tree during a search. 
// (index, distance from point)
#[derive(Debug)]
pub struct SID<T: Float>{
    pub id: usize,
    pub dist: T
}

impl <T:Float> PartialEq for SID<T> {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl <T:Float> PartialOrd for SID<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}

impl <T:Float> Eq for SID<T> {}

impl <T:Float> Ord for SID<T> {
    
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


pub struct Kdtree<'a, T: Float + 'static> {
    dim: usize,
    // capacity of leaf node
    capacity: usize,
    // Nodes
    left: Option<Box<Self>>,
    right: Option<Box<Self>>,
    // Is a leaf node if this has values
    split_idx: Option<usize>,
    indices: Option<&'a [usize]>, // if leaf, this is always Some. But this can be empty.
    // Data
    data: ArrayView2<'a, T>,
}

impl<'a, T: Float + 'static> Kdtree<'a, T> {


    pub fn build(
        data: ArrayView2<'a, T>, // each row in data will be called either a row or y
        dim: usize,
        capacity: usize,
        depth: usize,
        indices: &'a mut [usize],
    ) -> Kdtree<'a, T> {
        let n = indices.len();
        if n <= capacity {
            Kdtree {
                dim: dim,
                capacity: capacity,
                left: None,
                right: None,
                split_idx: None,
                indices: Some(indices),
                data: data,
            }
        } else {
            let axis = depth % dim;
            let half = n >> 1;
            let column = data.column(axis);
            indices.sort_unstable_by(|&i, &j| column[i].partial_cmp(&column[j]).unwrap());
            let split_idx = indices[half];
            let (left, right) = indices.split_at_mut(half);
            Kdtree {
                dim: dim,
                capacity: capacity,
                left: Some(Box::new(Self::build(
                    data,
                    dim,
                    capacity,
                    depth + 1,
                    left,
                ))),
                right: Some(Box::new(Self::build(
                    data,
                    dim,
                    capacity,
                    depth + 1,
                    right,
                ))),
                split_idx: Some(split_idx),

                indices: None,
                data: data,
            }
        }
    }

    fn compute_row_norms(&self) -> Vec<T> {
        self.data.rows().into_iter().map(|y| y.dot(&y)).collect()
    }

    fn is_leaf(&self) -> bool {
        self.indices.is_some()
    }

    #[inline]
    fn get_sid(&self, i:usize, point: ArrayView1<T>) -> SID<T> {
        SID {
            id: i,
            dist: squared_euclidean(point, self.data.row(i))
        }
    }

    fn best_in_leaf(&self, point: ArrayView1<T>) -> Option<SID<T>> {
        // This is only called if is_leaf. Safe to unwrap.
        let indices = self.indices.unwrap();
        if indices.len() > 0 {
            let mut min_dist: T = T::max_value();
            let mut min_idx: usize = 0;
            for i in indices.iter().cloned() {
                let test = self.data.row(i);
                let dist = squared_euclidean(test, point);
                if dist < min_dist {
                    min_dist = dist;
                    min_idx = i;
                }
            }
            Some(SID{id: min_idx, dist: min_dist})
        } else {
            None
        }
    }

    /// Updates the current top K with the incoming SID
    #[inline]
    fn update_top_k(mut top_k: Vec<SID<T>>, k:usize, sid: SID<T>) -> Vec<SID<T>> {
        let idx = top_k.partition_point(|s| *s < sid);
        if idx < top_k.len() {
            if top_k.len() + 1 > k {
                // This ensures top_k has k elements and no need to allocate
                top_k.pop();
            }
            top_k.insert(idx, sid);
        } else if idx < k {
            // This case is true if and only if (top_k.len() <= idx < k)
            top_k.push(sid); // by the end of the push, len is at most k.
        } // Do nothing if idx >= k, because that means no partition point was found within existing values
        top_k
    }

    #[inline]
    fn best_k_in_leaf(
        &self, 
        mut top_k: Vec<SID<T>>, 
        k:usize, 
        point:ArrayView1<T>,
        point_norm_cache: T,
    ) -> Vec<SID<T>> {
        // This is only called if is_leaf. Safe to unwrap.
        for i in self.indices.unwrap().iter().cloned() {
            let y = self.data.row(i);
            let dot_product = y.dot(&point);
            // Can further reduce this computation but need to be more careful.
            // This reduction only works when running multiple queries at once and may not work well with
            // multithreading. Not implemented yet.
            let y_norm = y.dot(&y);
            // This hack actually produces faster Euclidean dist calculation.
            let dist = point_norm_cache + y_norm - dot_product - dot_product;
            
            let p = SID{id: i, dist: dist};
            if let Some(cur_last) = top_k.last() {
                if cur_last.dist <= p.dist && top_k.len() >= k {
                    continue; // No need to check if we already have k elements and the dist is >= current max dist
                }
            }
            top_k = Self::update_top_k(top_k, k, p);
        }
        top_k
    }

    fn compare_points(p1: Option<SID<T>>, p2: Option<SID<T>>) -> Option<SID<T>> {
        if p1.is_none() {
            return p2;
        }

        if p2.is_none() {
            return p1;
        }

        let p1 = p1.unwrap();
        let p2 = p2.unwrap();

        if p1 < p2 {
            Some(p1)
        } else {
            Some(p2)
        }
    }

    pub fn nearest_neighbor(&self, point: ArrayView1<T>, depth: usize) -> Option<SID<T>> {
        if point.len() != self.dim {
            return None;
        }

        if self.is_leaf() {
            return self.best_in_leaf(point);
        }

        let axis = depth % self.dim;

        // Must exist
        let split_idx = self.split_idx.unwrap();
        let split_pt = self.data.row(split_idx);
        let dist = squared_euclidean(point, split_pt);

        let (next, oppo) = if point[axis] < split_pt[axis] {
            // Next = Self.left, opposite = self.right
            (
                self.left.as_ref().unwrap(),
                self.right.as_ref().unwrap(),
            )
        } else {
            (
                self.right.as_ref().unwrap(),
                self.left.as_ref().unwrap(),
            )
        };

        let candidate = next.nearest_neighbor(point, depth + 1);
        // This is safe because both candidate and Some((split_idx, dist)) cannot be None
        let mut best = Self::compare_points(Some(SID{id: split_idx, dist: dist}), candidate).unwrap();
        // Since split_idx is not None, we always have a best here
        // Square the RHS because the distance in best is squared 
        if best.dist > (point[axis] - split_pt[axis]).powi(2) {
            let best2 = oppo.nearest_neighbor(point, depth + 1);
            best = Self::compare_points(Some(best), best2).unwrap();
        }

        Some(best)
    }

    pub fn k_nearest_neighbors(
        &self, 
        k:usize, 
        point:ArrayView1<T>,
    ) -> Option<Vec<SID<T>>> {

        if k == 0 
            || (point.len() != self.dim) 
            || (point.iter().any(|v| !v.is_finite())) 
        {
            None
        } else {
            let point_norm = point.dot(&point);
            Some(self.k_nearest_neighbors_unchecked(
                Vec::with_capacity(k), 
                k, 
                point, 
                0,
                point_norm,
            ))
        }
    }

    #[inline]
    pub fn k_nearest_neighbors_unchecked(
        &self, 
        mut candidate:Vec<SID<T>>, 
        k:usize, 
        point: ArrayView1<T>, 
        depth: usize,
        point_norm_cache: T,
    ) -> Vec<SID<T>> {

        if self.is_leaf() {
            return self.best_k_in_leaf(candidate, k, point, point_norm_cache)
        }

        let axis = depth % self.dim;

        // Must exist
        let split_idx = self.split_idx.unwrap();
        let split_pt = self.data.row(split_idx);

        let (next, oppo) = if point[axis] < split_pt[axis] {
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

        candidate = next.k_nearest_neighbors_unchecked(candidate, k, point, depth + 1, point_norm_cache);
        let curret_max_dist = candidate.last().map(|x| x.dist).unwrap_or(T::max_value());
        let perpendicular_dist = (point[axis] - split_pt[axis]).powi(2);
        // If current_max_dist > perpendicular_dist, then
        // there is a chance we need to update candidate from opposite branch
        if curret_max_dist > perpendicular_dist {
            candidate = oppo.k_nearest_neighbors_unchecked(candidate, k, point, depth + 1, point_norm_cache);
        }
        candidate
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
    fn dim_2_nearest_neighbor() {
        let mat = arr2(&[
            [4.0, 7.0],
            [7.0, 13.0],
            [9.0, 4.0],
            [11.0, 10.0],
            [16.0, 10.0],
            [15.0, 3.0],
            [14.0, 11.0],
        ]);
        
        let mut indices = (0..mat.nrows()).collect::<Vec<usize>>();
        let indices = indices.as_mut_slice();
        let tree = Kdtree::build(
            mat.view(),
            mat.ncols(),
            1,
            0,
            indices,
        );

        let pt = arr1(&[14.0, 9.0]);
        let output = tree.nearest_neighbor(pt.view(), 0);

        assert!(output.is_some());

        let index = output.unwrap().id;
        assert_eq!(index, 6);
    }

    #[test]
    fn dim_3_random_nearest_neighbor() {

        let mut v = Vec::new();
        let rows = 1000usize;
        for _ in 0..rows {
            v.extend_from_slice(&random_3d_rows());
        }

        let mat = Array2::from_shape_vec((rows, 3), v).unwrap();
        let point = arr1(&[0.5, 0.5, 0.5]);
        // brute force test
        let distances = mat.rows().into_iter().map(|v| {
            squared_euclidean(v, point.view())
        }).collect::<Vec<_>>();

        let mut argmin = 0usize;
        let mut min_dist = f64::MAX;
        for (i, d) in distances.into_iter().enumerate() {
            if d < min_dist {
                min_dist = d;
                argmin = i;
            }
        }

        println!("Distance Brute Force: {}", min_dist);

        let mut indices = (0..mat.nrows()).collect::<Vec<usize>>();
        let indices = indices.as_mut_slice();
        let tree = Kdtree::build(
            mat.view(),
            mat.ncols(),
            8,
            0,
            indices,
        );

        let output = tree.nearest_neighbor(point.view(), 0);
        assert!(output.is_some());
        let output = output.unwrap();
        let index = output.id;
        let dist = output.dist;
        println!("Distance Kdtree: {}", dist);
        assert_eq!(argmin, index);
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
    

        let mut indices = (0..mat.nrows()).collect::<Vec<usize>>();
        let indices = indices.as_mut_slice();
        let tree = Kdtree::build(
            mat.view(),
            mat.ncols(),
            32,
            0,
            indices,
        );
    
        let output = tree.k_nearest_neighbors(k, point.view());
    
        assert!(output.is_some());
        let output = output.unwrap();
        let indices = output.iter().map(|sid| sid.id).collect::<Vec<_>>();
        let distances = output.iter().map(|sid| sid.dist).collect::<Vec<_>>();
    
        assert_eq!(&ans_argmins[..k], &indices);
        assert_eq!(&ans_distances[..k], &distances);

    }

}
