#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

#include "defines.h"

Eigen::VectorXd cg_solve(
    const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b,
    const Eigen::SparseMatrix<double>& M_inv, double tol
);

Eigen::MatrixXd
solve_cg_per_b(const Eigen::SparseMatrix<double>& A, const Eigen::MatrixXd& B, double tol);

Eigen::MatrixXd
solve_bcg(const Eigen::SparseMatrix<double>& A, const Eigen::MatrixXd& B, double tol);

// input: A, b, x0, M = LL^T
Eigen::MatrixXd solve_preconditioned_bcg(
    const Eigen::SparseMatrix<double>& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& X_0,
    const Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>& LLT
);
