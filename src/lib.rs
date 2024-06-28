#![feature(iter_array_chunks)]
/// IMPORTANT!
/// This crate is intentionally built to be imperfect.
/// E.g.
/// I am not checking whether all row_vecs have the same dimension in &[Leaf].
/// The reason for this is that I only intend to use this crate in my Python package polars_ds,
/// where it is always the case that that's true.
/// Since I do not plan to make this a general purpose Kdtree package for Rust yet, I do not
/// want to do those checks.
/// E.g.
/// I am not properly defining error types because
/// it will be casted to PolarsErrors when integrated with polars_ds, OR it will be used as Python errors
/// and it is more convenient to just use strings.
/// E.g.
/// within_count returns a u32 as opposed to usize because that can help me skip a type conversion when used with Polars.


pub mod arkadia;
pub mod arkadia_lp;
pub mod utils;
pub mod neighbor;
pub mod leaf;

pub use utils::{matrix_to_leaves_w_norm, matrix_to_leaves, suggest_capacity, SplitMethod};
pub use arkadia::Kdtree;
pub use arkadia_lp::{LpKdtree, LP};
pub use neighbor::NB;
pub use leaf::{Leaf, KdLeaf, LeafWithNorm};


// ---------------------------------------------------------------------------------------------------------
use num::Float;

#[derive(Clone, Default)]
pub enum KNNMethod {
    DInvW, // Distance Inversion Weighted. E.g. Use (1/(1+d)) to weight the regression / classification
    #[default]
    NoW // No Weight
}


/// KD Tree Queries
pub trait KDTQ<'a, T:Float, A> {

    fn knn_leaf(&self, k: usize, leaf: impl KdLeaf<'a, T>, epsilon: T) -> Option<Vec<NB<T, A>>>;

    fn knn_bounded_leaf(&self, k: usize, leaf: impl KdLeaf<'a, T>, max_dist_bound: T) -> Option<Vec<NB<T, A>>>;

    fn within_leaf(&self, leaf: impl KdLeaf<'a, T>, radius: T, sort: bool) -> Option<Vec<NB<T, A>>>;

    fn within_leaf_count(&self, leaf: impl KdLeaf<'a, T>, radius: T) -> Option<u32>;

}


pub trait KNNRegressor<'a, T:Float + Into<f64>, A: Float + Into<f64>> : KDTQ<'a, T, A> {

    fn knn_regress(&self, k: usize, leaf: impl KdLeaf<'a, T>, max_dist_bound: T, how:KNNMethod) -> Option<f64> {
        let knn = self.knn_bounded_leaf(k, leaf, max_dist_bound);
        match knn {
            Some(nn) => {
                match how {
                    KNNMethod::DInvW => {
                        let weights = nn.iter()
                            .map(|nb| (nb.dist + T::one()).recip().into())
                            .collect::<Vec<f64>>();
                        let sum = weights.iter().copied().sum::<f64>();
                        Some(
                            nn
                            .into_iter()
                            .zip(weights.into_iter())
                            .fold(0f64, |acc, (nb, w)| acc + w * nb.to_item().into())
                            / sum
                        )
                    },
                    KNNMethod::NoW => {
                        let n = nn.len() as f64;
                        Some(
                            nn.into_iter()
                            .fold(A::zero(), |acc, nb| acc + nb.to_item())
                            .into()
                            / n
                        )
                    },
                }
            },
            None => None,
        }
    }

}

pub trait KNNClassifier<'a, T:Float > : KDTQ<'a, T, u32> {

    fn knn_regress(&self, k: usize, leaf: impl KdLeaf<'a, T>, max_dist_bound: T, how:KNNMethod) -> Option<u32> {
        let knn = self.knn_bounded_leaf(k, leaf, max_dist_bound);
        match knn {
            Some(nn) => {
                match how {
                    KNNMethod::DInvW => {
                        todo!()
                    },
                    KNNMethod::NoW => {
                        todo!()
                    },
                }
            },
            None => None,
        }
    }

}