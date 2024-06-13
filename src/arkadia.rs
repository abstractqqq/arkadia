use ndarray::{ArrayView1, ArrayView2};
use num::Float;

// IMPORTANT!
// This crate is intentionally built to be imperfect.
// E.g.
// I am not checking whether all row_vecs have the same dimension in LeafElements.
// The reason for this is that I only intend to use this crate in my Python package polars_ds.
// Since I do not plan to make this a general purpose Kdtree package for Rust yet, I do not
// want to make those requirements.
// E.g.
// I am not properly defining error types because
// it will be casted to PolarsErrors when integrated with polars_ds.
// E.g.
// within_count returns a u32 as opposed to usize because that can help me skip a type conversion when used with Polars.

pub fn suggest_capacity(dim: usize) -> usize {
    if dim < 5 {
        8
    } else if dim < 10 {
        16
    } else if dim < 15 {
        32
    } else if dim < 20 {
        64
    } else if dim < 30 {
        128
    } else {
        256
    }
}

pub fn matrix_to_leaf_elements<'a, T: Float + 'static, A: Copy>(
    matrix: &'a ArrayView2<'a, T>,
    values: &'a [A],
) -> Vec<LeafElement<'a, T, A>> {
    values
        .iter()
        .zip(matrix.rows().into_iter())
        .filter(|(_, arr)| arr.iter().all(|x| x.is_finite()))
        .map(|(a, arr)| LeafElement {
            item: a.clone(),
            row_vec: arr,
            norm: arr.dot(&arr),
        })
        .collect::<Vec<_>>()
}

#[derive(Clone, Default)]
pub enum SplitMethod {
    #[default]
    MIDPOINT, // min + (max - min) / 2
    MEAN,
    MEDIAN,
}

// NB: Neighbor, search result
// (Data, and distance)
pub struct NB<T: Float, A> {
    pub dist: T,
    pub item: A,
}

impl<T: Float, A> PartialEq for NB<T, A> {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl<T: Float, A> PartialOrd for NB<T, A> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}

impl<T: Float, A> Eq for NB<T, A> {}

impl<T: Float, A> Ord for NB<T, A> {
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

#[derive(Debug)]
pub struct LeafElement<'a, T: Float, A> {
    pub item: A,
    pub row_vec: ArrayView1<'a, T>,
    pub norm: T,
}

impl<'a, T: Float, A> LeafElement<'a, T, A> {
    pub fn dim(&self) -> usize {
        self.row_vec.len()
    }

    pub fn is_finite(&self) -> bool {
        self.row_vec.iter().all(|x| x.is_finite())
    }
}

pub fn squared_euclidean<T: Float>(a: &[T], b: &[T]) -> T {
    a.iter()
        .zip(b.iter())
        .fold(T::zero(), |acc, (&x, &y)| acc + (x - y) * (x - y))
}

pub struct Kdtree<'a, T: Float + 'static, A> {
    dim: usize,
    // Nodes
    left: Option<Box<Kdtree<'a, T, A>>>,
    right: Option<Box<Kdtree<'a, T, A>>>,
    // Is a leaf node if this has values
    split_axis: Option<usize>,
    split_axis_value: Option<T>,
    min_bounds: Vec<T>,
    max_bounds: Vec<T>,
    // Data
    data: Option<&'a [LeafElement<'a, T, A>]>, // Not none when this is a leaf
}

impl<'a, T: Float + 'static, A: Copy> Kdtree<'a, T, A> {
    pub fn build(data: &'a mut [LeafElement<'a, T, A>], how: SplitMethod) -> Result<Self, String> {
        if data.is_empty() {
            return Err("Empty data.".into());
        }
        let dim = data.last().unwrap().dim();
        let capacity = suggest_capacity(dim);
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
            Kdtree {
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
            Kdtree {
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
    // Create the closest point in the box bounded by oppo to point and its norm
    fn closest_dist_to_box(min_bounds: &[T], max_bounds: &[T], point: ArrayView1<T>) -> T {
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

    #[inline(always)]
    fn update_top_k(
        &self,
        top_k: &mut Vec<NB<T, A>>,
        k: usize,
        point: ArrayView1<T>,
        point_norm_cache: T,
        max_dist_bound: T,
    ) {
        let max_permissible_dist = T::max_value().min(max_dist_bound);
        // This is only called if is_leaf. Safe to unwrap.
        for element in self.data.unwrap().iter() {
            let cur_max_dist = top_k.last().map(|nb| nb.dist).unwrap_or(max_dist_bound);
            let y = element.row_vec;
            // This hack actually produces faster Euclidean dist calculation (because .dot is unrolled in ndarray)
            let dot_product = y.dot(&point);
            let dist = point_norm_cache + element.norm - dot_product - dot_product;
            if dist < cur_max_dist || (top_k.len() < k && dist <= max_permissible_dist) {
                let nb = NB {
                    dist: dist,
                    item: element.item,
                };
                let idx: usize = top_k.partition_point(|s| s <= &nb);
                if idx < top_k.len() {
                    if top_k.len() + 1 > k {
                        top_k.pop();
                    } // This ensures top_k has k elements and no need to allocate
                    top_k.insert(idx, nb);
                } else if top_k.len() < k {
                    // Note if idx < top_k.len() is false, then index == top_k.len() for free!
                    // by the end of the push, len is at most k.
                    top_k.push(nb);
                } // Do nothing if idx >= k, because that means no partition point was found within existing values
            }
            // No need to check if we already have k elements and the dist is >= current max dist
        }
    }

    #[inline(always)]
    fn update_nb_within(
        &self,
        neighbors: &mut Vec<NB<T, A>>,
        point: ArrayView1<T>,
        point_norm_cache: T,
        radius: T,
    ) {
        // This is only called if is_leaf. Safe to unwrap.
        for element in self.data.unwrap().iter() {
            let y = element.row_vec;
            // This hack actually produces faster Euclidean dist calculation (because .dot is unrolled in ndarray)
            let dot_product = y.dot(&point);
            let dist = point_norm_cache + element.norm - dot_product - dot_product;
            if dist <= radius {
                neighbors.push(NB {
                    dist: dist,
                    item: element.item,
                });
            }
        }
    }

    pub fn knn(&self, k: usize, point: ArrayView1<T>) -> Option<Vec<NB<T, A>>> {
        if k == 0 || (point.len() != self.dim) || (point.iter().any(|x| !x.is_finite())) {
            None
        } else {
            // Always allocate 1 more.
            let mut top_k = Vec::with_capacity(k + 1);
            let mut pending = Vec::with_capacity(k + 1);
            pending.push((T::min_value(), self));
            let point_norm = point.dot(&point);
            while !pending.is_empty() {
                Self::knn_one_step(
                    &mut pending,
                    &mut top_k,
                    k,
                    point,
                    point_norm,
                    T::max_value(),
                );
            }
            Some(top_k)
        }
    }

    pub fn knn_leaf_elem(
        &self,
        k: usize,
        leaf_element: LeafElement<'a, T, A>,
    ) -> Option<Vec<NB<T, A>>> {
        if k == 0 || (leaf_element.dim() != self.dim) || (!leaf_element.is_finite()) {
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
                    leaf_element.norm,
                    T::max_value(),
                );
            }
            Some(top_k)
        }
    }

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
            let point_norm = point.dot(&point);
            while !pending.is_empty() {
                Self::knn_one_step(
                    &mut pending,
                    &mut top_k,
                    k,
                    point,
                    point_norm,
                    max_dist_bound,
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
            || (!leaf_element.is_finite())
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
                    leaf_element.norm,
                    max_dist_bound,
                );
            }
            Some(top_k)
        }
    }

    pub fn knn_one_step(
        pending: &mut Vec<(T, &Kdtree<'a, T, A>)>,
        top_k: &mut Vec<NB<T, A>>,
        k: usize,
        point: ArrayView1<T>,
        point_norm_cache: T,
        max_dist_bound: T,
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
            if dist_to_box < current_max {
                pending.push((dist_to_box, next));
            }
        }
        current.update_top_k(top_k, k, point, point_norm_cache, max_dist_bound);
    }

    pub fn within(&self, point: ArrayView1<T>, radius: T, sort: bool) -> Option<Vec<NB<T, A>>> {
        // radius is actually squared radius
        if radius <= T::zero() + T::epsilon() || (point.iter().any(|x| !x.is_finite())) {
            None
        } else {
            // Always allocate some.
            let mut neighbors = Vec::with_capacity(20);
            let mut pending = Vec::with_capacity(20);
            let point_norm = point.dot(&point);
            pending.push((T::min_value(), self));
            while !pending.is_empty() {
                Self::within_one_step(&mut pending, &mut neighbors, point, point_norm, radius);
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
        if radius <= T::zero() + T::epsilon() || (!leaf_element.is_finite()) {
            None
        } else {
            // Always allocate some.
            let mut neighbors = Vec::with_capacity(20);
            let mut pending = Vec::with_capacity(20);
            pending.push((T::min_value(), self));
            while !pending.is_empty() {
                Self::within_one_step(
                    &mut pending,
                    &mut neighbors,
                    leaf_element.row_vec,
                    leaf_element.norm,
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
            let mut pending = Vec::with_capacity(20);
            let point_norm = point.dot(&point);
            pending.push((T::min_value(), self));
            while !pending.is_empty() {
                cnt += Self::within_count_one_step(&mut pending, point, point_norm, radius);
            }
            Some(cnt)
        }
    }

    pub fn within_count_leaf_elem(
        &self,
        leaf_element: LeafElement<'a, T, A>,
        radius: T,
    ) -> Option<u32> {
        if radius <= T::zero() + T::epsilon() || (!leaf_element.is_finite()) {
            None
        } else {
            // Always allocate some.
            let mut cnt = 0u32;
            let mut pending = Vec::with_capacity(20);
            while !pending.is_empty() {
                cnt += Self::within_count_one_step(
                    &mut pending,
                    leaf_element.row_vec,
                    leaf_element.norm,
                    radius,
                );
            }
            Some(cnt)
        }
    }

    pub fn within_one_step(
        pending: &mut Vec<(T, &Kdtree<'a, T, A>)>,
        neighbors: &mut Vec<NB<T, A>>,
        point: ArrayView1<T>,
        point_norm_cache: T,
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
        current.update_nb_within(neighbors, point, point_norm_cache, radius);
    }

    pub fn within_count_one_step(
        pending: &mut Vec<(T, &Kdtree<'a, T, A>)>,
        point: ArrayView1<T>,
        point_norm_cache: T,
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
                let dot_product = y.dot(&point);
                let dist = point_norm_cache + element.norm - dot_product - dot_product;
                acc + (dist <= radius) as u32
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kdtree::distance::squared_euclidean;
    use ndarray::{arr1, Array2};

    fn random_10d_rows() -> [f64; 10] {
        rand::random()
    }

    #[test]
    fn test_10d_knn_l2_dist_midpoint() {
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
            .map(|v| squared_euclidean(v.to_slice().unwrap(), &point.to_vec()))
            .collect::<Vec<_>>();
        let mut ans_argmins = (0..rows).collect::<Vec<_>>();
        ans_argmins.sort_by(|&i, &j| ans_distances[i].partial_cmp(&ans_distances[j]).unwrap());
        ans_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let values = (0..rows).collect::<Vec<_>>();
        let binding = mat.view();
        let mut leaf_elements = matrix_to_leaf_elements(&binding, &values);

        let tree = Kdtree::build(&mut leaf_elements, SplitMethod::MIDPOINT).unwrap();

        let output = tree.knn(k, point.view());

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
    fn test_10d_knn_l2_dist_bounded() {
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
            .map(|v| squared_euclidean(v.to_slice().unwrap(), &point.to_vec()))
            .collect::<Vec<_>>();
        let mut ans_argmins = (0..rows).collect::<Vec<_>>();
        ans_argmins.sort_by(|&i, &j| ans_distances[i].partial_cmp(&ans_distances[j]).unwrap());
        ans_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let bound: f64 = 0.28; // usually random data will give min dist ~ 0.24, 0.25
        let idx = ans_distances.partition_point(|d| d < &bound);
        let (mut ans_argmins, _) = ans_argmins.split_at(idx);
        let (mut ans_distances, _) = ans_distances.split_at(idx);
        ans_argmins = &ans_argmins[..(k).min(ans_argmins.len())];
        ans_distances = &ans_distances[..(k).min(ans_distances.len())];

        let values = (0..rows).collect::<Vec<_>>();
        let binding = mat.view();
        let mut leaf_elements = matrix_to_leaf_elements(&binding, &values);

        let tree = Kdtree::build(&mut leaf_elements, SplitMethod::MIDPOINT).unwrap();

        let output = tree.knn_bounded(k, point.view(), bound);

        assert!(output.is_some());
        let output = output.unwrap();
        let indices = output.iter().map(|nb| nb.item).collect::<Vec<_>>();
        let distances = output.iter().map(|nb| nb.dist).collect::<Vec<_>>();

        // May have <= k elements
        assert_eq!(ans_argmins.len(), indices.len());
        assert_eq!(ans_distances.len(), distances.len());
        assert_eq!(&ans_argmins, &indices);
        for (d1, d2) in ans_distances.iter().zip(distances.into_iter()) {
            assert!((d1 - d2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_10d_knn_l2_dist_within() {
        // 10 nearest neighbors, matrix of size 1000 x 10
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
            .map(|v| squared_euclidean(v.to_slice().unwrap(), &point.to_vec()))
            .collect::<Vec<_>>();
        let mut ans_argmins = (0..rows).collect::<Vec<_>>();
        ans_argmins.sort_by(|&i, &j| ans_distances[i].partial_cmp(&ans_distances[j]).unwrap());
        ans_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let bound: f64 = 0.29; // usually random data will give min dist ~ 0.24, 0.25
        let idx = ans_distances.partition_point(|d| d < &bound);
        let (ans_argmins, _) = ans_argmins.split_at(idx);
        let (ans_distances, _) = ans_distances.split_at(idx);

        let values = (0..rows).collect::<Vec<_>>();
        let binding = mat.view();
        let mut leaf_elements = matrix_to_leaf_elements(&binding, &values);

        let tree = Kdtree::build(&mut leaf_elements, SplitMethod::MIDPOINT).unwrap();

        let output = tree.within(point.view(), bound, true);

        assert!(output.is_some());
        let output = output.unwrap();
        let indices = output.iter().map(|nb| nb.item).collect::<Vec<_>>();
        let distances = output.iter().map(|nb| nb.dist).collect::<Vec<_>>();

        // May have <= k elements
        assert_eq!(ans_argmins.len(), indices.len());
        assert_eq!(ans_distances.len(), distances.len());
        assert_eq!(&ans_argmins, &indices);
        for (d1, d2) in ans_distances.iter().zip(distances.into_iter()) {
            assert!((d1 - d2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_10d_knn_l2_dist_within_count() {
        // 10 nearest neighbors, matrix of size 1000 x 10
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
            .map(|v| squared_euclidean(v.to_slice().unwrap(), &point.to_vec()))
            .collect::<Vec<_>>();
        let mut ans_argmins = (0..rows).collect::<Vec<_>>();
        ans_argmins.sort_by(|&i, &j| ans_distances[i].partial_cmp(&ans_distances[j]).unwrap());
        ans_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let bound: f64 = 0.29; // usually random data will give min dist ~ 0.24, 0.25
        let idx = ans_distances.partition_point(|d| d < &bound);

        let values = (0..rows).collect::<Vec<_>>();
        let binding = mat.view();
        let mut leaf_elements = matrix_to_leaf_elements(&binding, &values);

        let tree = Kdtree::build(&mut leaf_elements, SplitMethod::MIDPOINT).unwrap();

        let output = tree.within_count(point.view(), bound);

        // May have <= k elements
        assert!(output.is_some());
        assert_eq!(idx as u32, output.unwrap());
    }

    #[test]
    fn test_10d_knn_l2_dist_mean() {
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
            .map(|v| squared_euclidean(v.to_slice().unwrap(), &point.to_vec()))
            .collect::<Vec<_>>();
        let mut ans_argmins = (0..rows).collect::<Vec<_>>();
        ans_argmins.sort_by(|&i, &j| ans_distances[i].partial_cmp(&ans_distances[j]).unwrap());
        ans_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let values = (0..rows).collect::<Vec<_>>();
        let binding = mat.view();
        let mut leaf_elements = matrix_to_leaf_elements(&binding, &values);

        let tree = Kdtree::build(&mut leaf_elements, SplitMethod::MEAN).unwrap();

        let output = tree.knn(k, point.view());

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
    fn test_10d_knn_l2_dist_median() {
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
            .map(|v| squared_euclidean(v.to_slice().unwrap(), &point.to_vec()))
            .collect::<Vec<_>>();
        let mut ans_argmins = (0..rows).collect::<Vec<_>>();
        ans_argmins.sort_by(|&i, &j| ans_distances[i].partial_cmp(&ans_distances[j]).unwrap());
        ans_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let values = (0..rows).collect::<Vec<_>>();
        let binding = mat.view();
        let mut leaf_elements = matrix_to_leaf_elements(&binding, &values);

        let tree = Kdtree::build(&mut leaf_elements, SplitMethod::MEDIAN).unwrap();

        let output = tree.knn(k, point.view());

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
