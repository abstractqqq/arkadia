# ARKaDia

**ARKaDia** is an ARray (ndarray) backed KD-tree that is more suitable for data science use cases. E.g. entropy calculations, KNN searches on a fixed dataset, etc.

It is built mostly to serve the Python package [polars_ds](https://github.com/abstractqqq/polars_ds_extension), which is a data science package that is dataframe-centric, and supports complex queries in a dataframe using Polar's syntax. This explains most of the unusual design decisions of the package, e.g. using ArrayViews as input arguments instead of &[T].

It is slightly faster than the other kdtree package in Rust [`kdtree`](https://crates.io/crates/kdtree). I am not comparing with other kdtree packages that require fixed length array at compile time. For data science use cases, data dimension has to be a variable. The code in here only tries to support L2, L Infinity, and maybe L1 distances. It has optimized implementation for each distance metric here. I do not intend to support Haversine distance or arbitrary distances. 

You can see benchmark logs below.

The most common use case of this package is to query 100s, 1000s, or 100k many nearest distances queries. E.g. in a dataframe with feature 1, 2, 3, you want to know the nearest neighbor for each row.

This project may never be published, because it needs to be tightly integrated with the aforementioned Python package, and therefore may not have the API or user-friendliess of a standalone package. This repo serves more as an experimental playground.

## Warning 

No matter what you do, single-machine Kdtree cannot perform vector search for LLMs in a scalable way. Kdtrees suffer from the curse of dimensionality greatly and will be very slow once dimension is > 12.

## Plans

1. Support L Infinity distance
2. Support Approximation. The simplest approximation scheme can be done by adding an epsilon in the criterion that checks whether we should check the opposite branch. More sophisciated approximation methods will be considered.
3. Range search support for L2.

## Benchmark Log

Local Run Log: cargo bench

```
KdTree Package 200 10NN queries (3D)
                        time:   [590.58 µs 590.85 µs 591.14 µs]
Found 9 outliers among 100 measurements (9.00%)
  1 (1.00%) low mild
  4 (4.00%) high mild
  4 (4.00%) high severe

ARKaDia 200 10NN queries (3D)
                        time:   [375.79 µs 376.63 µs 377.60 µs]
Found 14 outliers among 100 measurements (14.00%)
  2 (2.00%) high mild
  12 (12.00%) high severe

Benchmarking KdTree Package 200 10NN queries (5D): Warming up for 3.0000 s
Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 9.1s, enable flat sampling, or reduce sample count to 50.
KdTree Package 200 10NN queries (5D)
                        time:   [1.7917 ms 1.7921 ms 1.7925 ms]
Found 8 outliers among 100 measurements (8.00%)
  2 (2.00%) low mild
  2 (2.00%) high mild
  4 (4.00%) high severe

Benchmarking ARKaDia 200 10NN queries (5D): Warming up for 3.0000 s
Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 6.3s, enable flat sampling, or reduce sample count to 60.
ARKaDia 200 10NN queries (5D)
                        time:   [1.2434 ms 1.2471 ms 1.2508 ms]
Found 18 outliers among 100 measurements (18.00%)
  12 (12.00%) high mild
  6 (6.00%) high severe

KdTree Package 200 10NN queries (10D)
                        time:   [18.292 ms 18.382 ms 18.549 ms]
Found 22 outliers among 100 measurements (22.00%)
  4 (4.00%) low mild
  1 (1.00%) high mild
  17 (17.00%) high severe

ARKaDia 200 10NN queries (10D)
                        time:   [14.750 ms 14.780 ms 14.811 ms]

Benchmarking KdTree Package 200 within radius queries (sorted): Warming up for 3.0000 s
Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 12.1s, or reduce sample count to 40.
KdTree Package 200 within radius queries (sorted)
                        time:   [120.88 ms 120.93 ms 121.00 ms]
Found 8 outliers among 100 measurements (8.00%)
  7 (7.00%) high mild
  1 (1.00%) high severe

Benchmarking ARKaDia 200 within radius queries (sorted): Warming up for 3.0000 s
Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 5.3s, or reduce sample count to 90.
ARKaDia 200 within radius queries (sorted)
                        time:   [53.533 ms 53.579 ms 53.628 ms]
Found 38 outliers among 100 measurements (38.00%)
  16 (16.00%) low mild
  3 (3.00%) high mild
  19 (19.00%) high severe

ARKaDia 200 within radius queries (unsorted)
                        time:   [34.073 ms 34.188 ms 34.357 ms]
Found 1 outliers among 100 measurements (1.00%)
  1 (1.00%) high severe

ARKaDia 200 within radius count queries (sorted)
                        time:   [14.735 ms 14.808 ms 14.882 ms]
```