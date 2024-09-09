use arkadia::{
    arena_kdt::ArenaKdtree, kdt::{OwnedKDT, DIST, KDT}, matrix_to_leaves, matrix_to_leaves_owned, matrix_to_leaves_w_row_num, suggest_capacity, utils::SplitMethod, SpacialQueries
};
use criterion::{criterion_group, criterion_main, Criterion};
use kdtree as kd;
use ndarray::{arr1, Array1, Array2};

// All tests with 50_000 rows of data

fn linf_dist_slice(a1: &[f64], a2: &[f64]) -> f64 {
    a1.iter()
        .copied()
        .zip(a2.iter().copied())
        .fold(0., |acc, (x, y)| acc.max((x - y).abs()))
}

fn set_up_data_for_construction(dim: usize, nrows:usize) -> Array2<f64> {
    let mut v = Vec::new();

    for _ in 0..nrows {
        let data = (0..dim).map(|_| rand::random::<f64>()).collect::<Vec<_>>();
        v.extend_from_slice(&data);
    }
    let mat = Array2::from_shape_vec((nrows, dim), v).unwrap();
    let mat = mat.as_standard_layout().to_owned();
    mat
}

fn set_up_data(dim: usize, n: usize) -> (Array2<f64>, Vec<Array1<f64>>) {
    let mut v = Vec::new();
    let rows = 50_000usize;
    for _ in 0..rows {
        let data = (0..dim).map(|_| rand::random::<f64>()).collect::<Vec<_>>();
        v.extend_from_slice(&data);
    }

    let mat = Array2::from_shape_vec((rows, dim), v).unwrap();
    let mat = mat.as_standard_layout().to_owned();

    let mut points = Vec::new();
    for _ in 0..n {
        let random_vec = (0..dim).map(|_| rand::random::<f64>()).collect::<Vec<_>>();
        points.push(arr1(&random_vec));
    }
    (mat, points)
}

fn knn_queries_3d(c: &mut Criterion) {
    let k: usize = 10usize;
    let dim: usize = 3usize;
    let (matrix, points) = set_up_data(dim, 200);
    let values = (0..matrix.nrows()).collect::<Vec<_>>();

    let binding = matrix.view();

    let mut leaf_elements = matrix_to_leaves(&binding, &values);
    // For random uniform data, doesn't matter which method to choose. The kdtree package also uses midpoint

    let tree = KDT::from_leaves(&mut leaf_elements, DIST::SQL2).unwrap();

    // suggest_capacity(dim)
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
                let _ = tree.knn(k, rv.as_slice().unwrap(), 0f64);
            }
        })
    });
}

fn knn_queries_5d(c: &mut Criterion) {
    let k: usize = 10usize;
    let dim: usize = 5usize;
    let (matrix, points) = set_up_data(dim, 200);
    let values = (0..matrix.nrows()).collect::<Vec<_>>();

    let binding = matrix.view();

    let mut leaf_elements = matrix_to_leaves(&binding, &values);
    let mut leaf_elements = matrix_to_leaves(&binding, &values);
    // For random uniform data, doesn't matter which method to choose

    let tree = KDT::from_leaves(&mut leaf_elements, DIST::SQL2).unwrap();

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
                let _ = tree.knn(k, rv.as_slice().unwrap(), 0f64);
            }
        })
    });
}

fn knn_queries_5d_linf(c: &mut Criterion) {
    let k: usize = 10usize;
    let dim: usize = 5usize;
    let (matrix, points) = set_up_data(dim, 200);
    let values = (0..matrix.nrows()).collect::<Vec<_>>();

    let binding = matrix.view();
    let mut leaf_elements = matrix_to_leaves(&binding, &values);
    // For random uniform data, doesn't matter which method to choose
    let tree = KDT::from_leaves(&mut leaf_elements, DIST::LINF).unwrap();

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
                let _ = tree.knn(k, rv.as_slice().unwrap(), 0f64);
            }
        })
    });
}

fn knn_10d_tree_construction(c: &mut Criterion) {

    let dim: usize = 10usize;
    let matrix = set_up_data_for_construction(dim, 50_000);
    let binding = matrix.view();
    // For random uniform data, doesn't matter which method to choose

    c.bench_function("Kdtree package tree construction", |b| {
        b.iter(|| {
            let mut kd_tree = kd::KdTree::with_capacity(dim, suggest_capacity(dim));
            for (i, row) in matrix.rows().into_iter().enumerate() {
                let sl = row.to_slice().unwrap();
                let _ = kd_tree.add(sl, i);
            }
        })
    });

    c.bench_function("Arkadia package tree construction", |b| {
        b.iter(|| {
            let mut leaf_elements = matrix_to_leaves_w_row_num(&binding);
            let tree = KDT::from_leaves(&mut leaf_elements, DIST::SQL2).unwrap();
        })
    });

    c.bench_function("Arkadia package unchecked tree construction", |b| {
        b.iter(|| {
            let mut leaf_elements =  matrix_to_leaves_w_row_num(&binding);
            let tree = KDT::from_leaves_unchecked(&mut leaf_elements, DIST::SQL2);
        })
    });

    c.bench_function("Arkadia package bulk load tree construction", |b| {
        b.iter(|| {
            let mut leaf_elements =  matrix_to_leaves_w_row_num(&binding);
            let tree = KDT::from_leaves_bulk_load(&mut leaf_elements, dim, suggest_capacity(dim), 0, DIST::SQL2);
        })
    });

    c.bench_function("Arkadia ArenaKDT package tree construction", |b| {
        b.iter(|| {
            let mut leaf_elements =  matrix_to_leaves_w_row_num(&binding);
            let arena_kdt = ArenaKdtree::from_leaves(
                &mut leaf_elements,
                dim,
                suggest_capacity(dim),
                DIST::SQL2,
            );
        })
    });
}

fn knn_queries_3d_2(c: &mut Criterion) {
    let k: usize = 10usize;
    let dim: usize = 3usize;
    let (matrix, points) = set_up_data(dim, 400);
    let values = (0..matrix.nrows()).collect::<Vec<_>>();

    let binding = matrix.view();
    let mut leaf_elements = matrix_to_leaves(&binding, &values);
    let mut leaf_elements2 = leaf_elements.clone();
    // For random uniform data, doesn't matter which method to choose

    let tree = KDT::from_leaves(&mut leaf_elements, DIST::SQL2).unwrap();

    let arena_kdt =
        ArenaKdtree::from_leaves(&mut leaf_elements2, dim, suggest_capacity(dim), DIST::SQL2);

    let mut kd_tree = kd::KdTree::with_capacity(dim, suggest_capacity(dim));
    for (i, row) in matrix.rows().into_iter().enumerate() {
        let sl = row.to_slice().unwrap();
        let _ = kd_tree.add(sl, i);
    }

    c.bench_function(
        &format!("KdTree Package {} 10NN queries ({}D)", points.len(), dim),
        |b| {
            b.iter(|| {
                for rv in points.iter() {
                    let point_slice = rv.as_slice().unwrap();
                    let _ = kd_tree.nearest(point_slice, k, &kd::distance::squared_euclidean);
                }
            })
        },
    );

    c.bench_function(
        &format!("Arkadia {} 10NN queries ({}D)", points.len(), dim),
        |b| {
            b.iter(|| {
                for rv in points.iter() {
                    let _ = tree.knn(k, rv.as_slice().unwrap(), 0f64);
                }
            })
        },
    );

    c.bench_function(
        &format!("Arkadia ArenaKdt {} 10NN queries ({}D)", points.len(), dim),
        |b| {
            b.iter(|| {
                for rv in points.iter() {
                    let _ = arena_kdt.knn(k, rv.as_slice().unwrap(), 0f64);
                }
            })
        },
    );
}

fn knn_queries_10d(c: &mut Criterion) {
    let k: usize = 10usize;
    let dim: usize = 10usize;
    let (matrix, points) = set_up_data(dim, 200);
    let values = (0..matrix.nrows()).collect::<Vec<_>>();

    let binding = matrix.view();
    let mut leaf_elements = matrix_to_leaves(&binding, &values);
    let leaf_elements2 = matrix_to_leaves_owned(&binding, &values);
    // For random uniform data, doesn't matter which method to choose

    let tree = KDT::from_leaves(&mut leaf_elements, DIST::SQL2).unwrap();
    let tree_owned = OwnedKDT::from_leaves(leaf_elements2, DIST::SQL2, SplitMethod::MIDPOINT).unwrap();

    let mut kd_tree = kd::KdTree::with_capacity(dim, suggest_capacity(dim));
    for (i, row) in matrix.rows().into_iter().enumerate() {
        let sl = row.to_slice().unwrap();
        let _ = kd_tree.add(sl, i);
    }

    c.bench_function(
        &format!("KdTree Package {} 10NN queries (10D)", points.len()),
        |b| {
            b.iter(|| {
                for rv in points.iter() {
                    let point_slice = rv.as_slice().unwrap();
                    let _ = kd_tree.nearest(point_slice, k, &kd::distance::squared_euclidean);
                }
            })
        },
    );
    
    c.bench_function(
        &format!("Arkadia {} 10NN queries (10D), owned Tree", points.len()),
        |b| {
            b.iter(|| {
                for rv in points.iter() {
                    let _ = tree_owned.knn(k, rv.as_slice().unwrap(), 0f64);
                }
            })
        },
    );
    
    c.bench_function(
        &format!("Arkadia {} 10NN queries (10D)", points.len()),
        |b| {
            b.iter(|| {
                for rv in points.iter() {
                    let _ = tree.knn(k, rv.as_slice().unwrap(), 0f64);
                }
            })
        },
    );

}

fn knn_queries_60d(c: &mut Criterion) {
    let k: usize = 10usize;
    let dim: usize = 60usize;
    let (matrix, points) = set_up_data(dim, 10);
    let values = (0..matrix.nrows()).collect::<Vec<_>>();

    let binding = matrix.view();
    let mut leaf_elements = matrix_to_leaves(&binding, &values);
    let mut leaf_elements2 = matrix_to_leaves(&binding, &values);
    // For random uniform data, doesn't matter which method to choose

    let tree = KDT::from_leaves(&mut leaf_elements, DIST::SQL2).unwrap();

    let tree2 =
        ArenaKdtree::from_leaves(&mut leaf_elements2, dim, suggest_capacity(dim), DIST::SQL2);

    let mut kd_tree = kd::KdTree::with_capacity(dim, suggest_capacity(dim));
    for (i, row) in matrix.rows().into_iter().enumerate() {
        let sl = row.to_slice().unwrap();
        let _ = kd_tree.add(sl, i);
    }

    c.bench_function(
        &format!("KdTree Package {} 10NN queries ({}D)", points.len(), dim),
        |b| {
            b.iter(|| {
                for rv in points.iter() {
                    let point_slice = rv.as_slice().unwrap();
                    let _ = kd_tree.nearest(point_slice, k, &kd::distance::squared_euclidean);
                }
            })
        },
    );

    c.bench_function(
        &format!("Arkadia {} 10NN queries ({}D)", points.len(), dim),
        |b| {
            b.iter(|| {
                for rv in points.iter() {
                    let _ = tree.knn(k, rv.as_slice().unwrap(), 0f64);
                }
            })
        },
    );

    c.bench_function(
        &format!("Arkadia ArenaKDT {} 10NN queries ({}D)", points.len(), dim),
        |b| {
            b.iter(|| {
                for rv in points.iter() {
                    let _ = tree2.knn(k, rv.as_slice().unwrap(), 0f64);
                }
            })
        },
    );
}

fn knn_queries_20d(c: &mut Criterion) {
    let k: usize = 10usize;
    let dim: usize = 20usize;
    let (matrix, points) = set_up_data(dim, 10);
    let values = (0..matrix.nrows()).collect::<Vec<_>>();

    let binding = matrix.view();
    let mut leaf_elements = matrix_to_leaves(&binding, &values);
    let mut leaf_elements2 = matrix_to_leaves(&binding, &values);
    // For random uniform data, doesn't matter which method to choose

    let tree = KDT::from_leaves(&mut leaf_elements, DIST::SQL2).unwrap();

    let tree2 =
        ArenaKdtree::from_leaves(&mut leaf_elements2, dim, suggest_capacity(dim), DIST::SQL2);

    let mut kd_tree = kd::KdTree::with_capacity(dim, suggest_capacity(dim));
    for (i, row) in matrix.rows().into_iter().enumerate() {
        let sl = row.to_slice().unwrap();
        let _ = kd_tree.add(sl, i);
    }

    c.bench_function(
        &format!("KdTree Package {} 10NN queries ({}D)", points.len(), dim),
        |b| {
            b.iter(|| {
                for rv in points.iter() {
                    let point_slice = rv.as_slice().unwrap();
                    let _ = kd_tree.nearest(point_slice, k, &kd::distance::squared_euclidean);
                }
            })
        },
    );

    c.bench_function(
        &format!("Arkadia {} 10NN queries ({}D)", points.len(), dim),
        |b| {
            b.iter(|| {
                for rv in points.iter() {
                    let _ = tree.knn(k, rv.as_slice().unwrap(), 0f64);
                }
            })
        },
    );

    c.bench_function(
        &format!("Arkadia ArenaKDT {} 10NN queries ({}D)", points.len(), dim),
        |b| {
            b.iter(|| {
                for rv in points.iter() {
                    let _ = tree2.knn(k, rv.as_slice().unwrap(), 0f64);
                }
            })
        },
    );
}

fn knn_queries_10d_linf(c: &mut Criterion) {
    let k: usize = 10usize;
    let dim: usize = 10usize;
    let (matrix, points) = set_up_data(dim, 200);
    let values = (0..matrix.nrows()).collect::<Vec<_>>();

    let binding = matrix.view();
    let mut leaf_elements = matrix_to_leaves(&binding, &values);
    // For random uniform data, doesn't matter which method to choose
    let tree = KDT::from_leaves(&mut leaf_elements, DIST::LINF).unwrap();

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
                let _ = tree.knn(k, rv.as_slice().unwrap(), 0f64);
            }
        })
    });
}

fn within_queries(c: &mut Criterion) {
    let (matrix, points) = set_up_data(5, 200);
    let values = (0..matrix.nrows()).collect::<Vec<_>>();

    let binding = matrix.view();
    let mut leaf_elements = matrix_to_leaves(&binding, &values);
    // let mut leaf_elements = matrix_to_leaves(&binding, &values);
    // For random uniform data, doesn't matter which method to choose
    let tree = KDT::from_leaves(&mut leaf_elements, DIST::SQL2).unwrap();

    let mut kd_tree = kd::KdTree::with_capacity(5, suggest_capacity(5));
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
                let _ = tree.within(rv.as_slice().unwrap(), 0.29, true);
            }
        })
    });

    c.bench_function("ARKaDia 200 within radius queries (unsorted)", |b| {
        b.iter(|| {
            for rv in points.iter() {
                let _ = tree.within(rv.as_slice().unwrap(), 0.29, false);
            }
        })
    });
}

fn within_count_queries(c: &mut Criterion) {
    let (matrix, points) = set_up_data(5, 200);
    let values = (0..matrix.nrows()).collect::<Vec<_>>();

    let binding = matrix.view();
    let mut leaf_elements = matrix_to_leaves(&binding, &values);
    // For random uniform data, doesn't matter which method to choose
    let tree = KDT::from_leaves(&mut leaf_elements, DIST::SQL2).unwrap();

    let mut kd_tree = kd::KdTree::with_capacity(5, suggest_capacity(5));
    for (i, row) in matrix.rows().into_iter().enumerate() {
        let sl = row.to_slice().unwrap();
        let _ = kd_tree.add(sl, i);
    }

    c.bench_function("ARKaDia 200 within radius count queries", |b| {
        b.iter(|| {
            for rv in points.iter() {
                let _ = tree.within_count(rv.as_slice().unwrap(), 0.29);
            }
        })
    });
}

criterion_group!(
    benches,
    knn_queries_10d,
    // knn_10d_tree_construction,
    // knn_queries_3d,
    // knn_queries_3d_2,
    // knn_queries_5d,
    // knn_queries_20d,
    // knn_queries_60d,
    // knn_queries_5d_linf,
    // knn_queries_10d_linf,
    // within_queries,
    // within_count_queries
);
criterion_main!(benches);
