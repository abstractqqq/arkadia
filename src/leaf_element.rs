use ndarray::ArrayView1;
use num::Float;

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

    pub fn is_not_finite(&self) -> bool {
        self.row_vec.iter().any(|x| !x.is_finite())
    }
}