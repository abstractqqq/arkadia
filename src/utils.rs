use ndarray::ArrayView2;
use crate::leaf_element::LeafElement;
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

pub fn matrix_to_leaf_elements_no_norm<'a, T: Float + 'static, A: Copy>(
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
            norm:  T::nan() // Don't compute. This won't be used. 
            // arr.iter().fold(T::zero(), |acc, x| acc.max(x.abs())),
        })
        .collect::<Vec<_>>()
}