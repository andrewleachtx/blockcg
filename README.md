# Block Conjugate Gradient for Multiple RHS

We used WSL2 for this project.

Be sure submodules are pulled (Eigen).
```sh
git submodule update --init --recursive
```

# Building + Usage

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Using SuiteSparse Matrices

You can use real sparse matrices from [SuiteSparse](https://sparse.tamu.edu/) see [downloading](#downloading). This provides more fair runtime outputs than default random.

You can test a download with:

```sh
./build/spmm data/<group>/<name> <ncols in B>
./build/spmm data/audikw_1/audikw_1.mtx 128
```

To run all data downloaded in `data/` and store in a text file:

```sh
./scripts/bench_mtx_all.sh > test.txt
```

### Downloading

Use the provided download script with `<group>/<name>` from the SuiteSparse website. These are the ones we used:

```sh
./scripts/download_mtx.sh HB/bcsstk01
./scripts/download_mtx.sh SNAP/roadNet-CA
./scripts/download_mtx.sh GHS_psdef/audikw_1
./scripts/download_mtx.sh Hamm/hcircuit
./scripts/download_mtx.sh Andrianov/lpl1
./scripts/download_mtx.sh Andrianov/mip1
./scripts/download_mtx.sh Rothberg/gearbox
./scripts/download_mtx.sh Chen/pkustk12
./scripts/download_mtx.sh Janna/Flan_1565
```

This downloads and extracts the `.mtx` file into `data/`.

# References
1. Timothy A. Davis and Yifan Hu. 2011. The University of Florida Sparse Matrix Collection. ACM Transactions on Mathematical Software 38, 1, Article 1 (December 2011), 25 pages. DOI: https://doi.org/10.1145/2049662.2049663
