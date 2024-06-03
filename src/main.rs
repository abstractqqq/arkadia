mod arkadia;
mod distances;

use arkadia::Kdtree;
use ndarray::prelude::*;

use rand;
use crate::distances::squared_euclidean;


fn main() {

    use std::time::Instant;

    let k = 10usize;
    let mut v = Vec::new();
    let rows = 5_000usize;
    let dim = 20;
    for _ in 0..rows {
        let data = (0..dim).map(|_| rand::random()).collect::<Vec<_>>();
        v.extend_from_slice(&data);
    }

    let mat = Array2::from_shape_vec((rows, dim), v).unwrap();
    let mat = mat.as_standard_layout().to_owned();
    let point = arr1(&vec![0.5; dim]);
    // brute force test
    let now = Instant::now();
    let mut distances = mat.rows().into_iter().map(|v| {
        squared_euclidean(v, point.view())
    }).collect::<Vec<_>>();
    let mut argmins = (0..rows).collect::<Vec<_>>();
    argmins.sort_by(|&i, &j| distances[i].partial_cmp(&distances[j]).unwrap());
    let elapsed = now.elapsed();
    println!("Brute force time spent for 1 query: {}s", elapsed.as_secs_f32());

    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!("{:?}", &argmins[..k]);
    println!("{:?}", &distances[..k]);

    let now = Instant::now();

    let mut indices = (0..mat.nrows()).collect::<Vec<usize>>();
    let indices = indices.as_mut_slice();
    let tree = Kdtree::build(
        mat.view(),
        mat.ncols(),
        32,
        0,
        indices,
    );

    let _ = tree.k_nearest_neighbors(k, point.view());
    let elapsed = now.elapsed();
    println!("Kdtree total time spent: {}s", elapsed.as_secs_f32());

    let now = Instant::now();
    let output = tree.k_nearest_neighbors(k, point.view());
    let elapsed = now.elapsed();
    println!("Kdtree 1 query time spent: {}s", elapsed.as_secs_f32());

    assert!(output.is_some());
    let output = output.unwrap();
    let indices = output.iter().map(|sid| sid.id).collect::<Vec<_>>();
    let distances = output.iter().map(|sid| sid.dist).collect::<Vec<_>>();

    println!("{:?}", indices);
    println!("{:?}", distances);


}
