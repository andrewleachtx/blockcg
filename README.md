# Block Conjugate Gradient for Multiple RHS

Be sure submodules are pulled (Eigen).
```sh
git submodule update --init --recursive
```

# Building + Usage

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

```sh
./build/blockcg <mode> <matrix.mtx>
```

where `<mode>` is one of `cg`, `pcg`, `bcg`, `pbcg`, or `test_all`.

## Using SuiteSparse Matrices

You can use real sparse matrices from [SuiteSparse](https://sparse.tamu.edu/) see [downloading](#downloading). This provides more fair runtime outputs than default random.

### Downloading

Use the provided download script with `<group>/<name>` from the SuiteSparse website. We used:

```sh
./scripts/download_mtx.sh GHS_psdef/gridgena
```

This downloads and extracts the `.mtx` file into `data/`.

# References

1. Petr Tichý, Gérard Meurant, and Dorota Šimonová. *Block CG algorithms revisited*. arXiv:2502.16998, 2025. https://arxiv.org/abs/2502.16998
2. Timothy A. Davis and Yifan Hu. 2011. The University of Florida Sparse Matrix Collection. ACM Transactions on Mathematical Software 38, 1, Article 1 (December 2011), 25 pages. DOI: https://doi.org/10.1145/2049662.2049663
