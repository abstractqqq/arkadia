use ndarray::{Array1, ArrayView1, ArrayView2};
use num::Float;
use crate::distances::squared_euclidean;

pub enum KdtreeError {
    EmptyData,
    NonFiniteValue,
}

pub struct Kdtree<'a, T:Float + 'static> {
    
    dim: usize,
    // capacity of leaf node
    capacity: usize, 
    
    // Nodes
    left: Option<Box<Kdtree<'a, T>>>,
    right: Option<Box<Kdtree<'a, T>>>,
    // Is a leaf node if this has values
    split_index: Option<usize>,
    indices: Option<Vec<usize>>, // if leaf, this is always Some. But this can be empty.
    // Data
    data: ArrayView2<'a, T>

}

impl<'a, T:Float + 'static> Kdtree<'a, T> {

    fn build(
        data:ArrayView2<'a, T>, 
        dim:usize, 
        capacity:usize, 
        depth:usize,
        mut indices: Vec<usize>
    ) -> Kdtree<'a, T> {

        let n = indices.len();
        if n <= capacity {
            Kdtree {
                dim: dim,
                capacity: capacity,
                left: None,
                right: None,
                split_index:None,
                indices: Some(indices),
                data: data
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
                left: Some(
                    Box::new(
                        Self::build(data, dim, capacity, depth + 1, s[0..half].to_vec())
                    )
                ),
                right: Some(
                    Box::new(
                        Self::build(data, dim, capacity, depth + 1, s[half..].to_vec())
                    )
                ),
                split_index: Some(s[half]),
                indices: None,
                data: data
            }

        }

    }

    fn is_leaf(&self) -> bool {
        self.indices.is_some()
    }

    fn best_in_leaf(&self, point:ArrayView1<T>) -> Option<(usize, T)> {
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
            Some((min_idx, min_dist))
        } else {
            None
        }
    }

    fn closest_neighbor(&self, point: ArrayView1<T>, depth:usize) -> Option<(usize, T)> {

        if self.is_leaf() {
            return self.best_in_leaf(point)
        }
        
        let axis = depth % self.dim;



        // Must exist
        let pivot_idx = self.split_index.unwrap();
        let pivot = self.data.row(pivot_idx);

        // if point[axis] < pivot[axis] {
        //     let next_branch = self.left.unwrap();
        //     let op_branch = self.right.unwrap();
        // } else {
        //     let next_branch = self.right.unwrap();
        //     let op_branch = self.left.unwrap();
        // }



        todo!()

    }

}