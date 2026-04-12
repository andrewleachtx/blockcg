#include <Eigen/Dense>
#include <Eigen/Sparse>


// Based on Algorithm 2
void solve_cg_per_b(const Eigen::SparseMatrix<double>& A,
                    const Eigen::MatrixXd& B, const Eigen::VectorXd x0);

// Based on Algorithm 4
void solve_bcg(const Eigen::SparseMatrix<double>& A, const Eigen::MatrixXd& B,
               const Eigen::VectorXd x0);

// Not block at all - b is a single vector
Eigen::VectorXd cg_solve(const Eigen::SparseMatrix<double>& A,
                          const Eigen::VectorXd& b,
                          const Eigen::SparseMatrix<double>& M_inv,
                          double tol = 1e-6);