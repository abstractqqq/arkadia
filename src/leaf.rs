use ndarray::ArrayView1;
use num::Float;

pub struct LeafWithNorm<'a, T: Float + 'static, A: Copy> {
    pub item: A,
    pub row_vec: &'a [T],
    pub norm: T,
}

impl<'a, T: Float, A: Copy> From<(&A, ArrayView1<'a, T>)> for LeafWithNorm<'a, T, A> {
    fn from(value: (&A, ArrayView1<'a, T>)) -> Self {
        let arr = value.1;
        LeafWithNorm {
            item: value.0.clone(),
            row_vec: arr.to_slice().unwrap(),
            norm: arr.dot(&arr),
        }
    }
}

pub struct Leaf<'a, T: Float, A: Copy> {
    pub item: A,
    pub row_vec: &'a [T],
}

impl<'a, T: Float, A: Copy> From<(&A, ArrayView1<'a, T>)> for Leaf<'a, T, A> {
    fn from(value: (&A, ArrayView1<'a, T>)) -> Self {
        Leaf {
            item: value.0.clone(),
            row_vec: value.1.to_slice().unwrap(),
        }
    }
}

pub trait KdLeaf<'a, T: Float> {
    fn dim(&self) -> usize;

    fn is_finite(&self) -> bool;

    fn is_not_finite(&self) -> bool;

    fn vec(&self) -> &'a [T];

    fn norm(&self) -> T;
}

impl<'a, T: Float, A: Copy> KdLeaf<'a, T> for LeafWithNorm<'a, T, A> {
    fn dim(&self) -> usize {
        self.row_vec.len()
    }

    fn is_finite(&self) -> bool {
        self.row_vec.iter().all(|x| x.is_finite())
    }

    fn is_not_finite(&self) -> bool {
        self.row_vec.iter().any(|x| !x.is_finite())
    }

    fn vec(&self) -> &'a [T] {
        self.row_vec
    }

    fn norm(&self) -> T {
        self.norm
    }
}

impl<'a, T: Float, A: Copy> KdLeaf<'a, T> for Leaf<'a, T, A> {
    fn dim(&self) -> usize {
        self.row_vec.len()
    }

    fn is_finite(&self) -> bool {
        self.row_vec.iter().all(|x| x.is_finite())
    }

    fn is_not_finite(&self) -> bool {
        self.row_vec.iter().any(|x| !x.is_finite())
    }

    fn vec(&self) -> &'a [T] {
        self.row_vec
    }

    /// Do not use .norm on Leaf. It only exists for LeafWithNorm
    fn norm(&self) -> T {
        T::nan()
    }
}
