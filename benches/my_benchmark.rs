use arkadia::{
    arena_kdt::ArenaKdtree, kdt::{OwnedKDT, DIST, KDT}, slice_to_empty_leaves, slice_to_leaves, slice_to_owned_leaves, suggest_capacity, utils::SplitMethod, SpacialQueries
};
use criterion::{criterion_group, criterion_main, Criterion};
use kdtree as kd;
use simsimd::SpatialSimilarity;

// All tests with 50_000 rows of data

fn linf_dist_slice(a1: &[f64], a2: &[f64]) -> f64 {
    a1.iter()
        .copied()
        .zip(a2.iter().copied())
        .fold(0., |acc, (x, y)| acc.max((x - y).abs()))
}

fn set_up_data_for_construction(dim: usize, nrows:usize) -> (Vec<f64>, usize) {
    let mut v = Vec::new();

    for _ in 0..nrows {
        let data = (0..dim).map(|_| rand::random::<f64>()).collect::<Vec<_>>();
        v.extend_from_slice(&data);
    }
    (v, dim)
}

fn set_up_data(dim: usize, n: usize) -> ((Vec<f64>, usize), Vec<Vec<f64>>) {
    let rows = 50_000usize;
    let matrix_slice = set_up_data_for_construction(dim, rows);

    let mut points = Vec::new();
    for _ in 0..n {
        let random_vec = (0..dim).map(|_| rand::random::<f64>()).collect::<Vec<_>>();
        points.push(random_vec);
    }
    (matrix_slice, points)
}

fn knn_10d_tree_construction(c: &mut Criterion) {

    let dim: usize = 10usize;
    let (matrix_slice, dim) = set_up_data_for_construction(dim, 50_000);

    c.bench_function("Kdtree package tree construction", |b| {
        b.iter(|| {
            let mut kd_tree = kd::KdTree::with_capacity(dim, suggest_capacity(dim));
            for (i, row) in matrix_slice.chunks_exact(dim).into_iter().enumerate() {
                let _ = kd_tree.add(row, i);
            }
        })
    });

    c.bench_function("Arkadia package tree construction", |b| {
        b.iter(|| {
            let mut leaf_elements = slice_to_empty_leaves(&matrix_slice, dim);
            let tree = KDT::from_leaves(&mut leaf_elements, DIST::SQL2).unwrap();
        })
    });

    c.bench_function("Arkadia package unchecked tree construction", |b| {
        b.iter(|| {
            let mut leaf_elements = slice_to_empty_leaves(&matrix_slice, dim);
            let tree = KDT::from_leaves_unchecked(&mut leaf_elements, DIST::SQL2);
        })
    });

    c.bench_function("Arkadia package bulk load tree construction", |b| {
        b.iter(|| {
            let mut leaf_elements = slice_to_empty_leaves(&matrix_slice, dim);
            let tree = KDT::from_leaves_bulk_load(&mut leaf_elements, dim, suggest_capacity(dim), 0, DIST::SQL2);
        })
    });

    c.bench_function("Arkadia ArenaKDT package tree construction", |b| {
        b.iter(|| {
            let mut leaf_elements = slice_to_empty_leaves(&matrix_slice, dim);
            let arena_kdt = ArenaKdtree::from_leaves(
                &mut leaf_elements,
                dim,
                suggest_capacity(dim),
                DIST::SQL2,
            );
        })
    });
}


fn knn_queries_3d(c: &mut Criterion) {
    let k: usize = 10usize;
    let dim: usize = 3usize;
    let ((matrix_slice, dim), points) = set_up_data(dim, 200);
    let values = (0..(matrix_slice.len() / dim)).collect::<Vec<_>>();

    let mut leaf_elements = slice_to_leaves(&matrix_slice, dim, &values);
    // For random uniform data, doesn't matter which method to choose. The kdtree package also uses midpoint

    let tree = KDT::from_leaves(&mut leaf_elements, DIST::SQL2).unwrap();

    // suggest_capacity(dim)
    let mut kd_tree = kd::KdTree::with_capacity(dim, suggest_capacity(dim));
    for (i, row) in matrix_slice.chunks_exact(dim).enumerate() {
        let _ = kd_tree.add(row, i);
    }

    c.bench_function("KdTree Package 200 10NN queries (3D)", |b| {
        b.iter(|| {
            for rv in points.iter() {
                let _ = kd_tree.nearest(rv, k, &kd::distance::squared_euclidean);
            }
        })
    });

    c.bench_function("ARKaDia 200 10NN queries (3D)", |b| {
        b.iter(|| {
            for rv in points.iter() {
                let _ = tree.knn(k, rv, 0f64);
            }
        })
    });
}

fn knn_queries_5d_linf(c: &mut Criterion) {
    let k: usize = 10usize;
    let dim: usize = 5usize;
    let ((matrix_slice, dim), points) = set_up_data(dim, 200);
    let values = (0..(matrix_slice.len() / dim)).collect::<Vec<_>>();

    let mut leaf_elements = slice_to_leaves(&matrix_slice, dim, &values);
    // For random uniform data, doesn't matter which method to choose. The kdtree package also uses midpoint

    let tree = KDT::from_leaves(&mut leaf_elements, DIST::LINF).unwrap();

    // suggest_capacity(dim)
    let mut kd_tree = kd::KdTree::with_capacity(dim, suggest_capacity(dim));
    for (i, row) in matrix_slice.chunks_exact(dim).enumerate() {
        let _ = kd_tree.add(row, i);
    }

    c.bench_function("KdTree Package 200 10NN queries (5D)", |b| {
        b.iter(|| {
            for rv in points.iter() {
                let _ = kd_tree.nearest(rv, k, &linf_dist_slice);
            }
        })
    });

    c.bench_function("ARKaDia 200 10NN queries (5D)", |b| {
        b.iter(|| {
            for rv in points.iter() {
                let _ = tree.knn(k, rv, 0f64);
            }
        })
    });
}

fn knn_queries_10d(c: &mut Criterion) {
    let k: usize = 10usize;
    let dim: usize = 10usize;
    let ((matrix_slice, dim), points) = set_up_data(dim, 200);
    let values = (0..(matrix_slice.len() / dim)).collect::<Vec<_>>();

    let mut leaf_elements = slice_to_leaves(&matrix_slice, dim, &values);
    let leaf_elements2 = slice_to_owned_leaves(&matrix_slice, dim, &values);
    // For random uniform data, doesn't matter which method to choose. The kdtree package also uses midpoint

    let tree = KDT::from_leaves(&mut leaf_elements, DIST::SQL2).unwrap();
    let tree2 = OwnedKDT::from_leaves(leaf_elements2, DIST::SQL2, SplitMethod::MIDPOINT).unwrap();

    // suggest_capacity(dim)
    let mut kd_tree = kd::KdTree::with_capacity(dim, suggest_capacity(dim));
    for (i, row) in matrix_slice.chunks_exact(dim).enumerate() {
        let _ = kd_tree.add(row, i);
    }

    c.bench_function("KdTree Package 200 10NN queries (10D)", |b| {
        b.iter(|| {
            for rv in points.iter() {
                let _ = kd_tree.nearest(rv, k, &kd::distance::squared_euclidean);
            }
        })
    });

    c.bench_function("ARKaDia 200 10NN queries (10D)", |b| {
        b.iter(|| {
            for rv in points.iter() {
                let _ = tree.knn(k, rv, 0f64);
            }
        })
    });


    c.bench_function("ARKaDia owned tree 200 10NN queries (10D)", |b| {
        b.iter(|| {
            for rv in points.iter() {
                let _ = tree2.knn(k, rv, 0f64);
            }
        })
    });
}

fn knn_queries_20d(c: &mut Criterion) {
    let k: usize = 10usize;
    let dim: usize = 20usize;
    let ((matrix_slice, dim), points) = set_up_data(dim, 10);
    let values = (0..(matrix_slice.len() / dim)).collect::<Vec<_>>();

    let mut leaf_elements = slice_to_leaves(&matrix_slice, dim, &values);
    // For random uniform data, doesn't matter which method to choose. The kdtree package also uses midpoint

    let tree = KDT::from_leaves(&mut leaf_elements, DIST::SQL2).unwrap();

    // suggest_capacity(dim)
    let mut kd_tree = kd::KdTree::with_capacity(dim, suggest_capacity(dim));
    for (i, row) in matrix_slice.chunks_exact(dim).enumerate() {
        let _ = kd_tree.add(row, i);
    }

    c.bench_function("KdTree Package 200 10NN queries (20D)", |b| {
        b.iter(|| {
            for rv in points.iter() {
                let _ = kd_tree.nearest(rv, k, &kd::distance::squared_euclidean);
            }
        })
    });

    c.bench_function("ARKaDia 200 10NN queries (20D)", |b| {
        b.iter(|| {
            for rv in points.iter() {
                let _ = tree.knn(k, rv, 0f64);
            }
        })
    });

}



fn knn_queries_60d(c: &mut Criterion) {

    let k: usize = 10usize;
    let dim: usize = 60usize;
    let ((matrix_slice, dim), points) = set_up_data(dim, 10);
    let values = (0..(matrix_slice.len() / dim)).collect::<Vec<_>>();

    let mut leaf_elements = slice_to_leaves(&matrix_slice, dim, &values);
    let leaf_elements2 = slice_to_owned_leaves(&matrix_slice, dim, &values);
    // For random uniform data, doesn't matter which method to choose. The kdtree package also uses midpoint

    let tree = KDT::from_leaves(&mut leaf_elements, DIST::SQL2).unwrap();
    let tree2 = OwnedKDT::from_leaves(leaf_elements2, DIST::SQL2, SplitMethod::MIDPOINT).unwrap();

    // suggest_capacity(dim)
    let mut kd_tree = kd::KdTree::with_capacity(dim, suggest_capacity(dim));
    for (i, row) in matrix_slice.chunks_exact(dim).enumerate() {
        let _ = kd_tree.add(row, i);
    }

    c.bench_function(
        &format!("KdTree Package {} 10NN queries ({}D) no simd", points.len(), dim),
        |b| {
            b.iter(|| {
                for rv in points.iter() {
                    let _ = kd_tree.nearest(
                        rv, 
                        k, 
                        &kd::distance::squared_euclidean
                    );
                }
            })
        },
    );

    c.bench_function(
        &format!("KdTree Package {} 10NN queries ({}D) simsimd", points.len(), dim),
        |b| {
            b.iter(|| {
                for rv in points.iter() {
                    let _ = kd_tree.nearest(
                        rv, 
                        k, 
                        &|a1, a2| f64::sqeuclidean(a1, a2).unwrap()
                    );
                }
            })
        },
    );

    c.bench_function(
        &format!("KdTree Package {} 10NN queries ({}D) cfavml", points.len(), dim),
        |b| {
            b.iter(|| {
                for rv in points.iter() {
                    let _ = kd_tree.nearest(
                        rv, 
                        k, 
                        &cfavml::squared_euclidean
                    );
                }
            })
        },
    );

    c.bench_function(
        &format!("Arkadia {} 10NN queries ({}D)", points.len(), dim),
        |b| {
            b.iter(|| {
                for rv in points.iter() {
                    let _ = tree.knn(k, rv, 0f64);
                }
            })
        },
    );

    c.bench_function(
        &format!("Arkadia ArenaKDT {} 10NN queries ({}D)", points.len(), dim),
        |b| {
            b.iter(|| {
                for rv in points.iter() {
                    let _ = tree2.knn(k, rv, 0f64);
                }
            })
        },
    );
}


fn within_queries(c: &mut Criterion) {

    let dim: usize = 5usize;
    let ((matrix_slice, dim), points) = set_up_data(dim, 200);
    let values = (0..(matrix_slice.len() / dim)).collect::<Vec<_>>();

    let mut leaf_elements = slice_to_leaves(&matrix_slice, dim, &values);
    // For random uniform data, doesn't matter which method to choose. The kdtree package also uses midpoint

    let tree = KDT::from_leaves(&mut leaf_elements, DIST::SQL2).unwrap();
    // suggest_capacity(dim)
    let mut kd_tree = kd::KdTree::with_capacity(dim, suggest_capacity(dim));
    for (i, row) in matrix_slice.chunks_exact(dim).enumerate() {
        let _ = kd_tree.add(row, i);
    }

    c.bench_function("KdTree Package 200 within radius queries (sorted)", |b| {
        b.iter(|| {
            for rv in points.iter() {
                let _ = kd_tree.within(rv, 0.29, &kd::distance::squared_euclidean);
            }
        })
    });

    c.bench_function("ARKaDia 200 within radius queries (sorted)", |b| {
        b.iter(|| {
            for rv in points.iter() {
                let _ = tree.within(rv, 0.29, true);
            }
        })
    });

    c.bench_function("ARKaDia 200 within radius queries (unsorted)", |b| {
        b.iter(|| {
            for rv in points.iter() {
                let _ = tree.within(rv, 0.29, false);
            }
        })
    });
}

fn within_count_queries(c: &mut Criterion) {

    let dim: usize = 5usize;
    let ((matrix_slice, dim), points) = set_up_data(dim, 200);
    let values = (0..(matrix_slice.len() / dim)).collect::<Vec<_>>();

    let mut leaf_elements = slice_to_leaves(&matrix_slice, dim, &values);
    // For random uniform data, doesn't matter which method to choose. The kdtree package also uses midpoint

    let tree = KDT::from_leaves(&mut leaf_elements, DIST::SQL2).unwrap();
    // suggest_capacity(dim)
    let mut kd_tree = kd::KdTree::with_capacity(dim, suggest_capacity(dim));
    for (i, row) in matrix_slice.chunks_exact(dim).enumerate() {
        let _ = kd_tree.add(row, i);
    }

    c.bench_function("ARKaDia 200 within radius count queries", |b| {
        b.iter(|| {
            for rv in points.iter() {
                let _ = tree.within_count(rv, 0.29);
            }
        })
    });
}

criterion_group!(
    benches,
    knn_queries_10d,
    knn_10d_tree_construction,
    knn_queries_3d,
    knn_queries_20d,
    knn_queries_5d_linf,
    within_queries,
    within_count_queries,
    knn_queries_60d,
);
criterion_main!(benches);
