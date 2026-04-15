#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <chrono>

#include "defines.h"

Eigen::VectorXd cg_solve(
    const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, double tol
);

Eigen::VectorXd preconditioned_cg_solve(
    const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b,
    const Eigen::SparseMatrix<double>& M_inv, double tol
);

Eigen::MatrixXd
solve_cg_per_b(const Eigen::SparseMatrix<double>& A, const Eigen::MatrixXd& B, double tol, std::filesystem::path log_dir);


Eigen::MatrixXd
solve_pcg_per_b(const Eigen::SparseMatrix<double>& A, const Eigen::MatrixXd& B, double tol, std::filesystem::path log_dir);

Eigen::MatrixXd
solve_bcg(const Eigen::SparseMatrix<double>& A, const Eigen::MatrixXd& B, double tol, std::filesystem::path log_dir);

// input: A, b, x0, M = LL^T
Eigen::MatrixXd solve_preconditioned_bcg(
    const Eigen::SparseMatrix<double>& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& X_0,
    const Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>& LLT, std::filesystem::path log_dir
);
