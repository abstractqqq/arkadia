use arkadia::{
    matrix_to_leaves, matrix_to_leaves_w_norm, suggest_capacity, Kdtree, LpKdtree, SplitMethod, LP,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kdtree as kd;
use ndarray::{arr1, Array1, Array2};

fn linf_dist_slice(a1: &[f64], a2: &[f64]) -> f64 {
    a1.iter()
        .zip(a2.iter())
        .fold(0., |acc, (x, y)| acc.max((x - y).abs()))
}

fn set_up_data(dim: usize) -> (Array2<f64>, Vec<Array1<f64>>) {
    let mut v = Vec::new();
    let rows = 50_000usize;
    for _ in 0..rows {
        let data = (0..dim).map(|_| rand::random::<f64>()).collect::<Vec<_>>();
        v.extend_from_slice(&data);
    }

    let mat = Array2::from_shape_vec((rows, dim), v).unwrap();
    let mat = mat.as_standard_layout().to_owned();

    let mut points = Vec::new();
    for _ in 0..200 {
        let random_vec = (0..dim).map(|_| rand::random::<f64>()).collect::<Vec<_>>();
        points.push(arr1(&random_vec));
    }
    (mat, points)
}

fn knn_queries_3d(c: &mut Criterion) {
    let k: usize = 10usize;
    let dim: usize = 3usize;
    let (matrix, points) = set_up_data(dim);
    let values = (0..matrix.nrows()).collect::<Vec<_>>();

    let binding = matrix.view();
    let mut leaf_elements = matrix_to_leaves_w_norm(&binding, &values);
    // For random uniform data, doesn't matter which method to choose. The kdtree package also uses midpoint
    let tree = Kdtree::from_leaves(
        &mut leaf_elements,
        SplitMethod::default(), // defaults to midpoint
    )
    .unwrap();

    let mut kd_tree = kd::KdTree::with_capacity(dim, suggest_capacity(dim));
    for (i, row) in matrix.rows().into_iter().enumerate() {
        let sl = row.to_slice().unwrap();
        let _ = kd_tree.add(sl, i);
    }

    c.bench_function("KdTree Package 200 10NN queries (3D)", |b| {
        b.iter(|| {
            for rv in points.iter() {
                let point_slice = rv.as_slice().unwrap();
                let _ = kd_tree.nearest(point_slice, k, &kd::distance::squared_euclidean);
            }
        })
    });

    c.bench_function("ARKaDia 200 10NN queries (3D)", |b| {
        b.iter(|| {
            for rv in points.iter() {
                let _ = tree.knn(k, rv.view(), 0f64);
            }
        })
    });
}

fn knn_queries_5d(c: &mut Criterion) {
    let k: usize = 10usize;
    let dim: usize = 5usize;
    let (matrix, points) = set_up_data(dim);
    let values = (0..matrix.nrows()).collect::<Vec<_>>();

    let binding = matrix.view();
    let mut leaf_elements = matrix_to_leaves_w_norm(&binding, &values);
    // For random uniform data, doesn't matter which method to choose
    let tree = Kdtree::from_leaves(
        &mut leaf_elements,
        SplitMethod::default(), // defaults to midpoint
    )
    .unwrap();

    let mut kd_tree = kd::KdTree::with_capacity(dim, suggest_capacity(dim));
    for (i, row) in matrix.rows().into_iter().enumerate() {
        let sl = row.to_slice().unwrap();
        let _ = kd_tree.add(sl, i);
    }

    c.bench_function("KdTree Package 200 10NN queries (5D)", |b| {
        b.iter(|| {
            for rv in points.iter() {
                let point_slice = rv.as_slice().unwrap();
                let _ = kd_tree.nearest(point_slice, k, &kd::distance::squared_euclidean);
            }
        })
    });

    c.bench_function("ARKaDia 200 10NN queries (5D)", |b| {
        b.iter(|| {
            for rv in points.iter() {
                let _ = tree.knn(k, rv.view(), 0f64);
            }
        })
    });
}

fn knn_queries_5d_linf(c: &mut Criterion) {
    let k: usize = 10usize;
    let dim: usize = 5usize;
    let (matrix, points) = set_up_data(dim);
    let values = (0..matrix.nrows()).collect::<Vec<_>>();

    let binding = matrix.view();
    let mut leaf_elements = matrix_to_leaves(&binding, &values);
    // For random uniform data, doesn't matter which method to choose
    let tree = LpKdtree::from_leaves(
        &mut leaf_elements,
        SplitMethod::default(), // defaults to midpoint
        LP::LINF,
    )
    .unwrap();

    let mut kd_tree = kd::KdTree::with_capacity(dim, suggest_capacity(dim));
    for (i, row) in matrix.rows().into_iter().enumerate() {
        let sl = row.to_slice().unwrap();
        let _ = kd_tree.add(sl, i);
    }

    c.bench_function(
        "KdTree Package 200 10NN queries with L Inf dist (5D)",
        |b| {
            b.iter(|| {
                for rv in points.iter() {
                    let point_slice = rv.as_slice().unwrap();
                    let _ = kd_tree.nearest(point_slice, k, &linf_dist_slice);
                }
            })
        },
    );

    c.bench_function("ARKaDia 200 10NN queries with L Inf dist (5D)", |b| {
        b.iter(|| {
            for rv in points.iter() {
                let _ = tree.knn(k, rv.view(), 0f64);
            }
        })
    });
}

fn knn_queries_10d(c: &mut Criterion) {
    let k: usize = 10usize;
    let dim: usize = 10usize;
    let (matrix, points) = set_up_data(dim);
    let values = (0..matrix.nrows()).collect::<Vec<_>>();

    let binding = matrix.view();
    let mut leaf_elements = matrix_to_leaves_w_norm(&binding, &values);
    // For random uniform data, doesn't matter which method to choose
    let tree = Kdtree::from_leaves(
        &mut leaf_elements,
        SplitMethod::default(), // defaults to midpoint
    )
    .unwrap();

    let mut kd_tree = kd::KdTree::with_capacity(dim, suggest_capacity(dim));
    for (i, row) in matrix.rows().into_iter().enumerate() {
        let sl = row.to_slice().unwrap();
        let _ = kd_tree.add(sl, i);
    }

    c.bench_function("KdTree Package 200 10NN queries (10D)", |b| {
        b.iter(|| {
            for rv in points.iter() {
                let point_slice = rv.as_slice().unwrap();
                let _ = kd_tree.nearest(point_slice, k, &kd::distance::squared_euclidean);
            }
        })
    });

    c.bench_function("ARKaDia 200 10NN queries (10D)", |b| {
        b.iter(|| {
            for rv in points.iter() {
                let _ = tree.knn(k, rv.view(), 0f64);
            }
        })
    });
}

fn knn_queries_10d_linf(c: &mut Criterion) {
    let k: usize = 10usize;
    let dim: usize = 10usize;
    let (matrix, points) = set_up_data(dim);
    let values = (0..matrix.nrows()).collect::<Vec<_>>();

    let binding = matrix.view();
    let mut leaf_elements = matrix_to_leaves(&binding, &values);
    // For random uniform data, doesn't matter which method to choose
    let tree = LpKdtree::from_leaves(
        &mut leaf_elements,
        SplitMethod::default(), // defaults to midpoint
        LP::LINF,
    )
    .unwrap();

    let mut kd_tree = kd::KdTree::with_capacity(dim, suggest_capacity(dim));
    for (i, row) in matrix.rows().into_iter().enumerate() {
        let sl = row.to_slice().unwrap();
        let _ = kd_tree.add(sl, i);
    }

    c.bench_function(
        "KdTree Package 200 10NN queries with L Inf dist (10D)",
        |b| {
            b.iter(|| {
                for rv in points.iter() {
                    let point_slice = rv.as_slice().unwrap();
                    let _ = kd_tree.nearest(point_slice, k, &linf_dist_slice);
                }
            })
        },
    );

    c.bench_function("ARKaDia 200 10NN queries with L Inf dist (10D)", |b| {
        b.iter(|| {
            for rv in points.iter() {
                let _ = tree.knn(k, rv.view(), 0f64);
            }
        })
    });
}

fn within_queries(c: &mut Criterion) {
    let (matrix, points) = set_up_data(5);
    let values = (0..matrix.nrows()).collect::<Vec<_>>();

    let binding = matrix.view();
    let mut leaf_elements = matrix_to_leaves_w_norm(&binding, &values);
    // For random uniform data, doesn't matter which method to choose
    let tree = Kdtree::from_leaves(
        &mut leaf_elements,
        SplitMethod::default(), // defaults to midpoint
    )
    .unwrap();

    let mut kd_tree = kd::KdTree::with_capacity(5, 16);
    for (i, row) in matrix.rows().into_iter().enumerate() {
        let sl = row.to_slice().unwrap();
        let _ = kd_tree.add(sl, i);
    }

    c.bench_function("KdTree Package 200 within radius queries (sorted)", |b| {
        b.iter(|| {
            for rv in points.iter() {
                let point_slice = rv.as_slice().unwrap();
                let _ = kd_tree.within(point_slice, 0.29, &kd::distance::squared_euclidean);
            }
        })
    });

    c.bench_function("ARKaDia 200 within radius queries (sorted)", |b| {
        b.iter(|| {
            for rv in points.iter() {
                let _ = tree.within(rv.view(), 0.29, true);
            }
        })
    });

    c.bench_function("ARKaDia 200 within radius queries (unsorted)", |b| {
        b.iter(|| {
            for rv in points.iter() {
                let _ = tree.within(rv.view(), 0.29, false);
            }
        })
    });
}

fn within_count_queries(c: &mut Criterion) {
    let (matrix, points) = set_up_data(5);
    let values = (0..matrix.nrows()).collect::<Vec<_>>();

    let binding = matrix.view();
    let mut leaf_elements = matrix_to_leaves_w_norm(&binding, &values);
    // For random uniform data, doesn't matter which method to choose
    let tree = Kdtree::from_leaves(
        &mut leaf_elements,
        SplitMethod::default(), // defaults to midpoint
    )
    .unwrap();

    let mut kd_tree = kd::KdTree::with_capacity(5, 16);
    for (i, row) in matrix.rows().into_iter().enumerate() {
        let sl = row.to_slice().unwrap();
        let _ = kd_tree.add(sl, i);
    }

    c.bench_function("ARKaDia 200 within radius count queries", |b| {
        b.iter(|| {
            for rv in points.iter() {
                let _ = tree.within_count(rv.view(), 0.29);
            }
        })
    });
}

criterion_group!(
    benches,
    knn_queries_3d,
    knn_queries_5d,
    knn_queries_5d_linf,
    knn_queries_10d_linf,
    knn_queries_10d,
    within_queries,
    within_count_queries
);
criterion_main!(benches);
