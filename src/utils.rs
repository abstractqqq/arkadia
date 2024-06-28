use ndarray::ArrayView2;
use crate::leaf::{Leaf, LeafWithNorm};
use num::Float;

// ---

#[derive(Clone, Default)]
pub enum SplitMethod {
    #[default]
    MIDPOINT, // min + (max - min) / 2
    MEAN,
    MEDIAN,
}


// ---

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

pub fn matrix_to_leaves_w_norm<'a, T: Float + 'static, A: Copy>(
    matrix: &'a ArrayView2<'a, T>,
    values: &'a [A],
) -> Vec<LeafWithNorm<'a, T, A>> {
    values
        .iter()
        .zip(matrix.rows().into_iter())
        .map(|pair| pair.into())
        .collect::<Vec<_>>()
}

pub fn matrix_to_leaves<'a, T: Float + 'static, A: Copy>(
    matrix: &'a ArrayView2<'a, T>,
    values: &'a [A],
) -> Vec<Leaf<'a, T, A>> {
    values
        .iter()
        .zip(matrix.rows().into_iter())
        .map(|pair| pair.into())
        .collect::<Vec<_>>()
}