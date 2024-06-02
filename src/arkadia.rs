use crate::distances::squared_euclidean;
use ndarray::{Array1, ArrayView1, ArrayView2};
use num::Float;
use dary_heap::QuaternaryHeap;

pub enum KdtreeError {
    EmptyData,
    NonFiniteValue,
}


// Search ID
// The identification of a point within the Kd-tree during a search. 
// (index, distance from point)
pub struct SID<T: Float>{
    id: usize,
    dist: T
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
    left: Option<Box<Kdtree<'a, T>>>,
    right: Option<Box<Kdtree<'a, T>>>,
    // Is a leaf node if this has values
    split_idx: Option<usize>,
    indices: Option<Vec<usize>>, // if leaf, this is always Some. But this can be empty.
    // Data
    data: ArrayView2<'a, T>,
}

impl<'a, T: Float + 'static> Kdtree<'a, T> {
    pub fn build(
        data: ArrayView2<'a, T>,
        dim: usize,
        capacity: usize,
        depth: usize,
        mut indices: Vec<usize>,
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
            let half = n >> 1;
            let axis = depth % dim;
            let column = data.column(axis);
            let s = indices.as_mut_slice();
            s.sort_unstable_by(|i, j| column[*i].partial_cmp(&column[*j]).unwrap());

            Kdtree {
                dim: dim,
                capacity: capacity,
                left: Some(Box::new(Self::build(
                    data,
                    dim,
                    capacity,
                    depth + 1,
                    s[0..half].to_vec(),
                ))),
                right: Some(Box::new(Self::build(
                    data,
                    dim,
                    capacity,
                    depth + 1,
                    s[half..].to_vec(),
                ))),
                split_idx: Some(s[half]),
                indices: None,
                data: data,
            }
        }
    }

    fn is_leaf(&self) -> bool {
        self.indices.is_some()
    }

    fn best_in_leaf(&self, point: ArrayView1<T>) -> Option<SID<T>> {
        // This is only called if we pass is_leaf
        let indices = self.indices.as_ref().unwrap();
        if indices.len() > 0 {
            let mut min_dist: T = T::max_value();
            let mut min_idx: usize = 0;
            for i in self.indices.as_ref().unwrap().iter() {
                let test = self.data.row(*i);
                let dist = squared_euclidean(test, point);
                if dist < min_dist {
                    min_dist = dist;
                    min_idx = *i;
                }
            }
            Some(SID{id: min_idx, dist: min_dist})
        } else {
            None
        }
    }

    fn best_k_in_leaf(&self, k:usize, point:ArrayView1<T>) -> Option<QuaternaryHeap<SID<T>>> {

        let indices = self.indices.as_ref().unwrap();
        if indices.len() > 0 {
            let qheap: dary_heap::DaryHeap<SID<T>, 4> = QuaternaryHeap::with_capacity(k);

            todo!()

        } else {
            None
        }
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

        if p1.dist < p2.dist {
            Some(p1)
        } else {
            Some(p2)
        }
    }

    pub fn closest_neighbor(&self, point: ArrayView1<T>, depth: usize) -> Option<SID<T>> {
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

        if point[axis] < split_pt[axis] {
            // Next = Self.left, opposite = self.right
            let candidate = self
                .left
                .as_ref()
                .unwrap()
                .closest_neighbor(point, depth + 1);
            // This is safe because both candidate and Some((split_idx, dist)) cannot be None
            let mut best = Self::compare_points(Some(SID{id: split_idx, dist: dist}), candidate).unwrap();
            // Since split_idx is not None, we always have a best here

            // Square the RHS because the distance in best is squared 
            if best.dist > (point[axis] - split_pt[axis]).powi(2) {
                let best2 = self
                    .right
                    .as_ref()
                    .unwrap()
                    .closest_neighbor(point, depth + 1);
                best = Self::compare_points(Some(best), best2).unwrap();
            }

            Some(best)
        } else {
            // Next = Self.right, Opposite = self.left
            let candidate = self
                .right
                .as_ref()
                .unwrap()
                .closest_neighbor(point, depth + 1);
            let mut best = Self::compare_points(Some(SID{id: split_idx, dist: dist}), candidate).unwrap();

            // Square the RHS because the distance in best is squared 
            if best.dist > (point[axis] - split_pt[axis]).powi(2) {
                let best2 = self
                    .left
                    .as_ref()
                    .unwrap()
                    .closest_neighbor(point, depth + 1);
                best = Self::compare_points(Some(best), best2).unwrap();
            }

            Some(best)
        }
    }

    pub fn index_to_point(&self, index: usize) -> Option<ArrayView1<T>> {
        if index < self.data.nrows() {
            Some(self.data.row(index))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, Array2};

    fn random_2d_rows() -> [f64; 2] {
        rand::random()
    }

    fn random_3d_rows() -> [f64; 3] {
        rand::random()
    }

    #[test]
    fn dim_2_nearest_neighbor() {
        let matrix = arr2(&[
            [4.0, 7.0],
            [7.0, 13.0],
            [9.0, 4.0],
            [11.0, 10.0],
            [16.0, 10.0],
            [15.0, 3.0],
            [14.0, 11.0],
        ]);

        let tree = Kdtree::build(
            matrix.view(),
            matrix.ncols(),
            1,
            0,
            (0..matrix.nrows()).collect(),
        );

        let pt = arr1(&[14.0, 9.0]);
        let output = tree.closest_neighbor(pt.view(), 0);

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

        let tree = Kdtree::build(
            mat.view(),
            mat.ncols(),
            8,
            0,
            (0..mat.nrows()).collect(),
        );

        let output = tree.closest_neighbor(point.view(), 0);
        assert!(output.is_some());
        let output = output.unwrap();
        let index = output.id;
        let dist = output.dist;
        println!("Distance Kdtree: {}", dist);
        assert_eq!(argmin, index);
    }
}
