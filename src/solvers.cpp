#include "solvers.h"
using namespace Eigen;

// Based on Algorithm 2
void solve_cg_per_b(const Eigen::SparseMatrix<double>& A,
                    const Eigen::MatrixXd& B, const Eigen::VectorXd x0) { /*TODO*/     (void)A; (void)B; (void)x0; }
// Based on Algorithm 4
void solve_bcg(const Eigen::SparseMatrix<double>& A, const Eigen::MatrixXd& B,
               const Eigen::VectorXd x0) {   /*TODO*/    (void)A; (void)B; (void)x0; }

// Not block at all - b is a single vector
VectorXd cg_solve(const Eigen::SparseMatrix<double>& A, const VectorXd& b, const Eigen::SparseMatrix<double>& M_inv, double tol) {
    int n = A.cols();
    VectorXd x_k = VectorXd::Zero(n);
    VectorXd r_k = b - A * x_k;
    VectorXd h_k = M_inv * r_k;
    double delta_0 = (r_k.transpose()*h_k).value();
    double delta_k = delta_0;
    VectorXd p_k = h_k;
    while (delta_k > tol * tol *delta_0) {
        //A*search direction - used for step size
        VectorXd s_k = A*p_k;
        // step size to get as close as possible along the search direction p_k
        double alpha_k = delta_k/(p_k.transpose()*s_k).value();
        // take step (core cg step, stepping conjugate to all other steps)
        VectorXd x_kp1 = x_k + alpha_k*p_k;
        // residual update without recomputing A
        VectorXd r_kp1 = r_k - alpha_k * s_k;
        VectorXd h_kp1 = M_inv * r_kp1;
        double delta_kp1 = (r_kp1.transpose()*h_kp1).value();
        //update next search direction
        VectorXd p_kp1 = h_kp1 + delta_kp1/delta_k*p_k;
        // update to new k
        p_k = p_kp1;
        delta_k = delta_kp1;
        x_k = x_kp1;
        r_k = r_kp1;

    }

    return x_k;
}
