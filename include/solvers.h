#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

#include "defines.h"

// Based on Algorithm 2
// A in nxn, b is nxm, x is nxm
Eigen::MatrixXd solve_cg_per_b(const Eigen::SparseMatrix<double>& A,
                               const Eigen::MatrixXd& B, double tol);

// Based on Algorithm 4
Eigen::MatrixXd solve_bcg(const Eigen::SparseMatrix<double>& A,
                          const Eigen::MatrixXd& B, double tol);

// Not block at all - b is a single vector
Eigen::VectorXd cg_solve(const Eigen::SparseMatrix<double>& A,
                         const Eigen::VectorXd& b,
                         const Eigen::SparseMatrix<double>& M_inv, double tol);
