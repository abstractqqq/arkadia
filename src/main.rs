mod arkadia;
mod distances;

use arkadia::Kdtree;
use ndarray::prelude::*;

fn main() {
    let matrix = arr2(&[
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0],
        [5.0, 5.0, 5.0],
        [6.0, 6.0, 6.0],
    ]);

    let tree = Kdtree::build(
        matrix.view(),
        matrix.ncols(),
        16,
        0,
        (0..matrix.nrows()).collect(),
    );

    let pt = arr1(&[1.1, 1.1, 1.1]);
    let output = tree.closest_neighbor(pt.view(), 0);

    println!("{:?}\n", output);
    println!("{:?}\n", tree.index_to_point(output.unwrap().0));
}
