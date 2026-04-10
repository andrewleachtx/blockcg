#include <Eigen/Dense>
#include <Eigen/Sparse>

// Based on Algorithm 2
void solve_cg_per_b(const Eigen::SparseMatrix<double>& A,
                    const Eigen::MatrixXd& B, const Eigen::VectorXd x0);

// Based on Algorithm 4
void solve_bcg(const Eigen::SparseMatrix<double>& A, const Eigen::MatrixXd& B,
               const Eigen::VectorXd x0);
