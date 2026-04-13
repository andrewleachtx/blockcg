#include "solvers.h"
using namespace Eigen;

// Not block at all - b is a single vector
VectorXd cg_solve(const SparseMatrix<double>& A, const VectorXd& b,
                  const SparseMatrix<double>& M_inv, double tol) {
    int n = A.cols();
    VectorXd x_k = VectorXd::Zero(n);
    VectorXd r_k = b - A * x_k;
    VectorXd h_k = M_inv * r_k;
    double delta_0 = (r_k.transpose() * h_k).value();
    double delta_k = delta_0;
    VectorXd p_k = h_k;
    while (delta_k > tol * tol * delta_0) {
        //A*search direction - used for step size
        VectorXd s_k = A * p_k;
        // step size to get as close as possible along the search direction p_k
        double alpha_k = delta_k / (p_k.transpose() * s_k).value();
        // take step (core cg step, stepping conjugate to all other steps)
        VectorXd x_kp1 = x_k + alpha_k * p_k;
        // residual update without recomputing A
        VectorXd r_kp1 = r_k - alpha_k * s_k;
        VectorXd h_kp1 = M_inv * r_kp1;
        double delta_kp1 = (r_kp1.transpose() * h_kp1).value();
        //update next search direction
        VectorXd p_kp1 = h_kp1 + delta_kp1 / delta_k * p_k;
        // update to new k
        p_k = p_kp1;
        delta_k = delta_kp1;
        x_k = x_kp1;
        r_k = r_kp1;
    }

    return x_k;
}

// For each column b in B, run cg_solve(..., b, ...)
MatrixXd solve_cg_per_b(const SparseMatrix<double>& A, const MatrixXd& B,
                        double tol) {
    const int n { static_cast<int>(A.rows()) },
        m { static_cast<int>(B.cols()) };
    MatrixXd X(n, m);

    // No preconditioner == let Minv = I
    SparseMatrix<double> I(n, n);
    I.setIdentity();

    for (int i = 0; i < m; i++) {
        VectorXd b = B.col(i);
        VectorXd x_b = cg_solve(A, b, I, tol);

        /*
            Block CG gives us an X of form x0 | x1 | x2 | ... | x_m
            so we can just iteratively form the same matrix as we walk
            through each column b
        */
        X.col(i) = x_b;
    }

    return X;
}

// Based on Algorithm 4
/// no preconditioning
MatrixXd solve_bcg(const SparseMatrix<double>& A, const MatrixXd& B,
                   double tol) { //, const Eigen::MatrixXd& x0

    int n = A.cols();
    int m = B.cols();
    ///k=0
    MatrixXd x_k = MatrixXd::Zero(n, m);
    MatrixXd r_k = B - A * x_k;
    // phi = I: Hestenes and Stiefel version. Choosing a better phi can avoid breakdown in rank deficient cases
    MatrixXd phi_k = MatrixXd::Identity(m, m);
    MatrixXd p_k = r_k * phi_k;
    while (r_k.squaredNorm() > tol * tol * B.squaredNorm()) {
        //step size matrix mxm because coupling along search directions
        // solve is more efficient than inverting
        MatrixXd r_km1 = r_k;
        MatrixXd gamma_k =
            (p_k.transpose() * A * p_k)
                .lu()
                .solve(phi_k.transpose() * r_km1.transpose() * r_km1);
        // update x_k
        x_k = x_k + p_k * gamma_k;
        /// update residual
        r_k = r_k - A * p_k * gamma_k;
        //compute next search directions
        MatrixXd rtr_km1 = r_km1.transpose() * r_km1; // m×m
        MatrixXd rtr_k = r_k.transpose() * r_k;       // m×m
        MatrixXd delta_k = phi_k.lu().solve(rtr_km1.lu().solve(rtr_k));
        p_k = (r_k + p_k * delta_k) * phi_k;
    }
    return x_k;
}
