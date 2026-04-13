#include "solvers.h"
using Eigen::SparseMatrix, Eigen::VectorXd, Eigen::MatrixXd;

// Standard preconditioned conjugate gradient from Chen's pseudocode. x_0 is always 0.
VectorXd cg_solve(const SparseMatrix<double>& A, const VectorXd& b,
                  const SparseMatrix<double>& M_inv, double tol) {
    const int n = A.cols();

    // Setup
    VectorXd x_k = VectorXd::Zero(n); // current guess
    VectorXd r_k = b - A * x_k;       // current residual, always b - Ax_k
    VectorXd h_k = M_inv * r_k;       // preconditioned residual
    double delta_0 =
        (r_k.transpose() * h_k).value(); // basically r_k dot preconditioned r_k
    double delta_k = delta_0;
    VectorXd p_k = h_k; // current search direction

    // Iterate until error delta_k reduces below tolerance
    while (delta_k > tol * tol * delta_0) {
        // We need to know how far along our search direction p_k to step
        // A * search direction gives us a vector s_k to be used in step size calc
        VectorXd s_k = A * p_k;

        // If we do p_k^T * A * p_k and get a small value, we know there isn't much
        // change in the linear system for this step, and so we can take a large
        // step. That said, if p_k^T * A * p_k is large, it means A * p_k has diverged
        // such that we should take smaller steps here. Note that if p_k and s_k were
        // at a 90 degree angle, this would be a big difference, and cause a zero.
        // But this is supposed to be positive definite, and that would be a violation:
        // x^T Ax > 0 is the requirement.
        double alpha_k = delta_k / (p_k.transpose() * s_k).value();

        // Apply step, x_{k+1} = x_k + step size * search direction
        VectorXd x_kp1 = x_k + alpha_k * p_k;

        // Update the residual, b - Ax_{k+1} would work, but this requires a spmv
        // so simplify: r_{k+1} = b - A(x_k + a_k * p_k) = (b - Ax_k) - a_k * A * p_k
        // = r_k - a_k * s_k
        VectorXd r_kp1 = r_k - alpha_k * s_k;

        // Apply the preconditioner again to get our preconditioned residual
        // we need a "preconditioned residual" in the first place because
        // r_k is the negative gradient, and thus iterating based on it would
        // try to take us straight to the path of minimizing error. The idea is
        // the preconditioner (idk how) gives us more information about our optimization
        // step, so that we do not zigzag
        VectorXd h_kp1 = M_inv * r_kp1;

        // Update err
        double delta_kp1 = (r_kp1.transpose() * h_kp1).value();

        // Update the search space using the new mixed residual search direction h_pk1.
        // This also uses the old search direction p_k to some extent, which is
        // weighted by how much our error has changed. If the new error is much less,
        // we want to follow the h_kp1 more - but if the new error is actually more, we
        // do not want to follow this search direction.
        VectorXd p_kp1 = h_kp1 + ((delta_kp1 / delta_k) * p_k);

        // Update for next iteration
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
    const int n = A.rows();
    const int m = B.cols();
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

// Based on Algorithm 4 - no preconditioner.
MatrixXd solve_bcg(const SparseMatrix<double>& A, const MatrixXd& B,
                   double tol) {
    const int n = A.cols();
    const int m = B.cols();

    // Basically, now that we have multiple righthand sides, we extend our vectors in basic CG to
    // n x m or m x m size. For each of m columns, there is one RHS / soln vector. This makes sense:
    // x_k, r_k, p_k are now n x m instead of just n x 1
    // phi_k, gamma_k=alpha_k, and delta_k are all m x m, which is a bit confusing, but when we do
    // X_{k+1} = X_k + P_k * bigalpha_k we have
    // n x m = n x m * ??? => we need ??? to be m x m - but also another reason, which is that Block CG
    // will mix m columns together, which I'm not sure on why or how

    // Setup
    MatrixXd x_k = MatrixXd::Zero(n, m);
    MatrixXd r_k = B - A * x_k;

    // phi = I (no update): Hestenes and Stiefel version. Choosing a better phi can avoid breakdown in rank deficient cases
    MatrixXd phi_k = MatrixXd::Identity(m, m);
    MatrixXd p_k = r_k * phi_k;

    // Same idea as CG in matrix form, but .squaredNorm() is basically just our Frobenius norm squared.
    // The Frobenius norm being the sum squared residual per entry i, j in the matrix.
    while (r_k.squaredNorm() > tol * tol * B.squaredNorm()) {
        // Save old, r_k minus 1
        MatrixXd r_km1 = r_k;

        // Block version of step size computation. Phi_k = I, so nothing really happens here.
        // Solving is better than doing (denominator inverse) * numerator
        MatrixXd gamma_k =
            (p_k.transpose() * A * p_k)
                .lu()
                .solve(phi_k.transpose() * r_km1.transpose() * r_km1);

        // Same CG updates, matrix form. Move the whole block along based on the new alpha block.
        x_k = x_k + p_k * gamma_k;
        r_k = r_k - A * p_k * gamma_k;

        // R_{k-1}^T * R_{k-1} and R_k^T * R_k
        MatrixXd rtr_km1 = r_km1.transpose() * r_km1;
        MatrixXd rtr_k = r_k.transpose() * r_k;

        MatrixXd delta_k = phi_k.lu().solve(rtr_km1.lu().solve(rtr_k));
        p_k = (r_k + p_k * delta_k) * phi_k;
    }

    return x_k;
}
