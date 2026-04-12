#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>


// Based on Algorithm 2
// A in nxn, b is nxm, x is nxm
void solve_cg_per_b(const Eigen::SparseMatrix<double>& A,
                    const Eigen::MatrixXd& B, double tol); //, const Eigen::MatrixXd& x0

// Based on Algorithm 4
Eigen::MatrixXd solve_bcg(const Eigen::SparseMatrix<double>& A, const Eigen::MatrixXd& B, double tol); //, const Eigen::MatrixXd& x0


// Not block at all - b is a single vector
Eigen::VectorXd cg_solve(const Eigen::SparseMatrix<double>& A,
                          const Eigen::VectorXd& b,
                          const Eigen::SparseMatrix<double>& M_inv,
                          double tol = 1e-6);