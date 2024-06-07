use crate::distances::{squared_euclidean};
use ndarray::{Array1, ArrayView1, ArrayView2};
use num::Float;

pub enum KdtreeError {
    EmptyData,
    NonFiniteValue,
}

pub struct LeafElement<'a, T:Float, A> {
    pub data: A, 
    pub row_vec: ArrayView1<'a, T>
}

// Search Result
// (Data, and distance)
pub struct NB<T:Float, A>{
    pub data: A ,
    pub dist: T
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

pub struct Kdtree<'a, T: Float + 'static, A> {
    dim: usize,
    // capacity of leaf node
    capacity: usize,
    // Nodes
    left: Option<Box<Kdtree<'a, T, A>>>,
    right: Option<Box<Kdtree<'a, T, A>>>,
    // Is a leaf node if this has values
    split_axis_value: Option<T>,
    // Data
    data: Option<&'a [LeafElement<'a, T, A>]> // Not none when this is a leaf
}

impl<'a, T: Float + 'static, A: Copy> Kdtree<'a, T, A> {

    pub fn build(
        data: &'a mut [LeafElement<'a, T, A>],
        dim: usize,
        capacity: usize,
        depth: usize,
    ) -> Self {
        let n = data.len();
        if n <= capacity {
            Kdtree {
                dim: dim,
                capacity: capacity,
                left: None,
                right: None,
                split_axis_value: None,
                data: Some(data),
            }
        } else {

            let axis = depth % dim;
            // let half = n >> 1;

            let mut min = T::max_value();
            let mut max = T::min_value();
            for elem in data.iter() {
                min = min.min(elem.row_vec[axis]);
                max = max.max(elem.row_vec[axis]);
            }
            let midpoint = min + (max - min) / (T::one() + T::one());
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
                left: Some(Box::new(Self::build(
                    left,
                    dim,
                    capacity,
                    depth + 1
                ))),
                right: Some(Box::new(Self::build(
                    right,
                    dim,
                    capacity,
                    depth + 1
                ))),
                split_axis_value: Some(midpoint),
                data: None,
            }
        }
    }

    fn is_leaf(&self) -> bool {
        self.data.is_some()
    }

    /// Updates the current top K with the incoming nb
    #[inline(always)]
    fn update_top_k(mut top_k: Vec<NB<T, A>>, k:usize, nb: NB<T, A>) -> Vec<NB<T, A>> {
        let idx: usize = top_k.partition_point(|s| s <= &nb);
        if idx < top_k.len() { 
            if top_k.len() + 1 > k {
                top_k.pop();
            } // This ensures top_k has k elements and no need to allocate
            top_k.insert(idx, nb);
        } else if top_k.len() < k {
            // Note if idx < top_k.len() is false, then index == top_k.len()! No need to check this.
            top_k.push(nb); // by the end of the push, len is at most k.
        } // Do nothing if idx >= k, because that means no partition point was found within existing values
        top_k
    }

    #[inline(always)]
    fn best_k_in_leaf(
        &self, 
        mut top_k: Vec<NB<T, A>>, 
        k:usize, 
        point:ArrayView1<T>,
        point_norm_cache: T,
    ) -> Vec<NB<T, A>> {
        // This is only called if is_leaf. Safe to unwrap.
        for element in self.data.unwrap().iter() {
            let cur_max_dist = top_k.last().map(|nb| nb.dist).unwrap_or(T::max_value());
            let y = element.row_vec;
            // This hack actually produces faster Euclidean dist calculation (because .dot is unrolled in ndarray)
            let dot_product = y.dot(&point);
            let y_norm = y.dot(&y);
            let dist = point_norm_cache + y_norm - dot_product - dot_product;
            if cur_max_dist <= dist && top_k.len() >= k {
                continue; // No need to check if we already have k elements and the dist is >= current max dist
            }
            top_k = Self::update_top_k(
                top_k, 
                k, 
                NB {
                    data: element.data,
                    dist:dist
                }
            );
        }
        top_k
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
            Some(self.knn_unchecked(
                Vec::with_capacity(k), 
                k, 
                point, 
                0,
                point_norm,
            ))
        }
    }

    #[inline(always)]
    pub fn knn_unchecked(
        &self, 
        mut candidate:Vec<NB<T, A>>, 
        k:usize, 
        point: ArrayView1<T>, 
        depth: usize,
        point_norm_cache: T,
    ) -> Vec<NB<T, A>> {
        
        let axis = depth % self.dim;
        if self.is_leaf() {
            return self.best_k_in_leaf(
                candidate, 
                k, 
                point,
                point_norm_cache
            )
        }
        
        // Must exist
        let axis_value = self.split_axis_value.unwrap();
        
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
        
        candidate = next.knn_unchecked(
            candidate, 
            k, 
            point, 
            depth + 1, 
            point_norm_cache
        );
        
        let cur_max_dist = candidate.last().map(|s| s.dist).unwrap_or(T::max_value());
        let perp_dist = (point[axis] - axis_value).powi(2);
        // If current_max_dist > perpendicular_dist, then
        // there is a chance we need to update candidate from opposite branch
        if cur_max_dist > perp_dist {
            candidate = oppo.knn_unchecked(
                candidate, 
                k, 
                point, 
                depth + 1, 
                point_norm_cache
            );
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
            .map(|(i, arr)| LeafElement{ data: i, row_vec: arr})
            .collect::<Vec<_>>();
        let tree = Kdtree::build(
            &mut data,
            mat.ncols(),
            32,
            0,
        );
    
        let output = tree.knn(k, point.view());
    
        assert!(output.is_some());
        let output = output.unwrap();
        let indices = output.iter().map(|nb| nb.data).collect::<Vec<_>>();
        let distances = output.iter().map(|nb| nb.dist).collect::<Vec<_>>();
    
        assert_eq!(&ans_argmins[..k], &indices);
        for (d1, d2) in ans_distances[..k].iter().zip(distances.into_iter()) {
            assert!((d1-d2).abs() < 1e-10);
        }

    }

}
