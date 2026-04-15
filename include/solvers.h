#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <chrono>

#include "defines.h"

std::pair<Eigen::VectorXd, std::vector<std::pair<double, long>>> cg_solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, double tol);

std::pair<Eigen::VectorXd, std::vector<std::pair<double, long>>> preconditioned_cg_solve(
    const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, const Eigen::IncompleteCholesky<double, Eigen::Lower, Eigen::NaturalOrdering<int>>& IC, double tol
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
    const Eigen::IncompleteCholesky<double, Eigen::Lower, Eigen::NaturalOrdering<int>>& IC, std::filesystem::path log_dir
);
