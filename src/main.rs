mod arkadia;
mod distances;

use arkadia::{Kdtree, LeafElement};
use ndarray::prelude::*;

use kdtree::KdTree as kd;

use rand::{self, random};
use crate::distances::squared_euclidean;


fn main() {

    use std::time::Instant;

    let k: usize = 20usize;
    let mut v = Vec::new();
    let rows = 100_000usize;
    let dim = 3;
    println!("Dim: {}, k: {}, size: {}", dim, k, rows);
    for _ in 0..rows {
        let data = (0..dim).map(|_| rand::random()).collect::<Vec<_>>();
        v.extend_from_slice(&data);
    }

    let mat = Array2::from_shape_vec((rows, dim), v).unwrap();
    let mat = mat.as_standard_layout().to_owned();
    let point = arr1(&vec![random::<f64>(); dim]);
    let point_slice = point.as_slice().unwrap();
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
    let random_points:Vec<Array1<f64>> = (0..1000).map(|_| Array1::from_vec(vec![random(); dim])).collect();

    let now = Instant::now();
    let mut data = 
        mat
        .rows()
        .into_iter()
        .enumerate()
        .map(|(i, arr)| LeafElement{ 
            data: i, 
            row_vec: arr, 
            norm: arr.dot(&arr)
        })
        .collect::<Vec<_>>();
    let tree = Kdtree::build(
        &mut data,
        mat.ncols(),
    );
    let elapsed = now.elapsed();
    println!("My Kdtree build time: {}s", elapsed.as_secs_f32());

    let now = Instant::now();
    let output = tree.knn_non_recurse(k, point.view());
    // .knn(k, point.view());
    let elapsed = now.elapsed();
    println!("My Kdtree 1 query time spent: {}s", elapsed.as_secs_f32());
    assert!(output.is_some());
    let output = output.unwrap();
    let indices = output.iter().map(|nb| nb.item).collect::<Vec<_>>();
    let distances = output.iter().map(|nb| nb.dist).collect::<Vec<_>>();

    println!("{:?}", indices);
    println!("{:?}", distances);

    let now = Instant::now();
    for pt in random_points.iter() {
        let output = tree.knn(k, pt.view());
    }
    let elapsed = now.elapsed();
    println!("My Kdtree 1k query time spent: {}s", elapsed.as_secs_f32());


    let now = Instant::now();

    let mut kd = kd::with_capacity(dim, 64);
    for (i, row) in mat.rows().into_iter().enumerate() {
        let sl = row.to_slice().unwrap();
        let _ = kd.add(sl, i);
    }
    let elapsed = now.elapsed();
    println!("Kdtree package build time: {}s", elapsed.as_secs_f32());

    let now = Instant::now();
    let output = kd.nearest(point_slice, k, &kdtree::distance::squared_euclidean);
    let elapsed = now.elapsed();
    println!("Kdtree package 1 query time spent: {}s", elapsed.as_secs_f32());
    let output = output.unwrap();
    let indices = output.iter().map(|t| t.1).collect::<Vec<_>>();
    let distances = output.iter().map(|t| t.0).collect::<Vec<_>>();

    println!("{:?}", indices);
    println!("{:?}", distances);

    let now = Instant::now();
    for pt in random_points.iter()  {
        let point_slice = pt.as_slice().unwrap();
        let output = kd.nearest(point_slice, k, &kdtree::distance::squared_euclidean);
    }
    let elapsed = now.elapsed();
    println!("Kdtree package 1k query time spent: {}s", elapsed.as_secs_f32());

}
