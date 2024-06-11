use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::random;
use ndarray::{Array2, Array1, arr1};
use arkadia::{LeafElement, Kdtree, suggest_capacity, matrix_to_leaf_elements, SplitMethod};
use kdtree as kd;

fn set_up_data() -> (Array2<f64>, Vec<Array1<f64>>) {

    let mut v = Vec::new();
    let rows = 50_000usize;
    let dim = 5;
    for _ in 0..rows {
        let data = (0..dim).map(|_| rand::random::<f64>()).collect::<Vec<_>>();
        v.extend_from_slice(&data);
    }

    let mat = Array2::from_shape_vec((rows, dim), v).unwrap();
    let mat = mat.as_standard_layout().to_owned();


    let mut points = Vec::new();
    for _ in 0..200 {
        let random_vec = [
            random::<f64>(), 
            random::<f64>(), 
            random::<f64>(), 
            random::<f64>(), 
            random::<f64>()
        ];
        points.push(arr1(&random_vec));

    }
    (mat, points)
}


fn criterion_benchmark(c: &mut Criterion) {

    let k: usize = 10usize;
    let (matrix, points) = set_up_data();
    let values = (0..matrix.nrows()).collect::<Vec<_>>();

    let binding = matrix.view();
    let mut leaf_elements = matrix_to_leaf_elements(&binding, &values);
    let tree = Kdtree::build(
        &mut leaf_elements
        , matrix.ncols()
        , SplitMethod::default() // defaults to midpoint
    ); // For random uniform data, doesn't matter which method to choose

    let mut kd_tree = kd::KdTree::with_capacity(5, 16);
    for (i, row) in matrix.rows().into_iter().enumerate() {
        let sl = row.to_slice().unwrap();
        let _ = kd_tree.add(sl, i);
    }


    c.bench_function("KdTree Package 200 queries", 
    |b| b.iter(|| {
        for rv in points.iter() {
            let point_slice = rv.as_slice().unwrap();
            let _ = kd_tree.nearest(point_slice, k, &kd::distance::squared_euclidean);
        }
    }));

    c.bench_function("ARKaDia 200 queries", 
    |b| b.iter(|| {
        for rv in points.iter() {
            let _ = tree.knn(k, rv.view());
        }
    }));
}


criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
