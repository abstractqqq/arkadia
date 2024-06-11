# ARKaDia

**ARKaDia** is an ARray (ndarray) backed KD-tree that is more suitable for data science use cases. E.g. entropy calculations, KNN searches on a fixed dataset, etc.

It is built mostly to serve the Python package [polars_ds](https://github.com/abstractqqq/polars_ds_extension), which is a data science package that is dataframe-centric, and supports complex queries in a dataframe using Polar's syntax. This explains most of the unusual design decisions of the package, e.g. using ArrayViews as input arguments instead of &[T].

It is slightly faster than the other kdtree package in Rust [`kdtree`](https://crates.io/crates/kdtree). I am not comparing with other kdtree packages that require fixed length array at compile time. For data science use cases, data dimension has to be a variable. The code in here only tries to support L2, L Infinity, and maybe L1 distances. It has optimized implementation for each distance metric here. I do not intend to support Haversine distance or arbitrary distances. 

![benchmark](./pictures/bench.png)

The most common use case of this package is to query 100s, 1000s, or 100k many nearest distances queries. E.g. in a dataframe with feature 1, 2, 3, you want to know the nearest neighbor for each row.

This project may never be published, because it needs to be tightly integrated with the aforementioned Python package, and therefore may not have the API or user-friendliess of a standalone package. This repo serves more as an experimental playground.