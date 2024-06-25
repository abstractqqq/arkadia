#![feature(iter_array_chunks)]
/// IMPORTANT!
/// This crate is intentionally built to be imperfect.
/// E.g.
/// I am not checking whether all row_vecs have the same dimension in LeafElements.
/// The reason for this is that I only intend to use this crate in my Python package polars_ds,
/// where it is always the case that that's true.
/// Since I do not plan to make this a general purpose Kdtree package for Rust yet, I do not
/// want to meet those requirements.
/// E.g.
/// I am not properly defining error types because
/// it will be casted to PolarsErrors when integrated with polars_ds.
/// E.g.
/// within_count returns a u32 as opposed to usize because that can help me skip a type conversion when used with Polars.


pub mod arkadia;
pub mod arkadia_linf;
pub mod arkadia_l1;
pub mod utils;
pub mod neighbor;
pub mod leaf_element;

pub use utils::{matrix_to_leaf_elements, matrix_to_leaf_elements_no_norm, suggest_capacity, SplitMethod};
pub use arkadia::Kdtree;
pub use arkadia_linf::{LIKdtree, linf_dist};
pub use arkadia_l1::{L1Kdtree, l1_dist};
pub use neighbor::NB;
pub use leaf_element::LeafElement;
