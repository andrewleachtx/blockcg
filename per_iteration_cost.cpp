// This file has the per-iteration costs for our implementations of various code.
// It is not intended to compile, but to allow us to convey this point easier than the
// raw ugly C++ code in src/solvers.cpp.

// (P)CG //
// Parenthesized line is only in PCG, ignore it for plain CG.
// n = dim of A, j = avg nnz per row of A
// todo: oliver sanity chk
while (delta_k > tol * tol * delta_0) {
    VectorXd s_k = A * p_k;                                         // jn   (SpMV)
    double alpha_k = delta_k / (p_k.transpose() * s_k).value();     // n
    VectorXd x_kp1 = x_k + alpha_k * p_k;                           // 2n
    VectorXd r_kp1 = r_k - alpha_k * s_k;                           // 2n
    (VectorXd h_kp1 = M_inv * r_kp1;)                               // (jn) PCG only
    double delta_kp1 = (r_kp1.transpose() * h_kp1).value();         // n
    VectorXd p_kp1 = h_kp1 + ((delta_kp1 / delta_k) * p_k);         // 2n

    // Overall: (2)jn + 8n  (CG: jn + 8n, PCG: 2jn + 8n)
}

// BCG //
// n = dim of A, m = block width, j = avg nnz per row of A
// todo: oliver sanity chk
while (r_k.squaredNorm() > tol * tol * B.squaredNorm()) {
    MatrixXd gamma_k =
        (p_k.transpose() * A * p_k)
            .lu()
            .solve(phi_k.transpose() * r_km1.transpose() * r_km1); // njm + nm^2 + (2/3)m^3 + m^2 + 2nm^2

    x_k = x_k + p_k * gamma_k;                                     // nm + nm^2
    r_k = r_k - A * p_k * gamma_k;                                 // nm + njm + nm^2

    MatrixXd rtr_km1 = r_km1.transpose() * r_km1;                  // nm^2
    MatrixXd rtr_k   = r_k.transpose() * r_k;                      // nm^2

    MatrixXd delta_k = phi_k.lu().solve(rtr_km1.lu().solve(rtr_k));// 2m^3 + 2m^2
    p_k = (r_k + p_k * delta_k) * phi_k;                           // nm + 2nm^2

    // Overall: 2njm + 9nm^2 + 3nm + (8/3)m^3 + 3m^2
}

// PBCG //
// n = dim of A, m = block width, j = avg nnz per row of A, nnz(L) = nnz of Cholesky factor
while (R_k.squaredNorm() > EPSILON * EPSILON * b_sqnorm) {     // 2nm
    MatrixXd As_k = A * s_k;                                   // 2jnm  (SpMM)
    MatrixXd xi_k = (s_k.transpose() * As_k).inverse();        // 2nm^2 + m^3

    X_k = X_k + s_k * xi_k * sigma_k;                          // 4nm^2 + nm

    MatrixXd L_inv_As = L.triangularView<Lower>().solve(As_k); // 2 * nnz(L) * m
    MatrixXd w_kp1, zeta_k;
    compute_qr(w_k - L_inv_As * xi_k, w_kp1, zeta_k);          // 2nm^2 + O(nm^2) QR

    MatrixXd s_kp1 =
        L.transpose().triangularView<Upper>().solve(w_kp1)
        + s_k * zeta_k.transpose();                            // 2 * nnz(L) * m + 2nm^2 + nm

    MatrixXd sigma_kp1 = zeta_k * sigma_k;                     // 2m^3

    R_k = R_k - As_k * xi_k * sigma_k;                         // 4nm^2 + nm

    w_k = w_kp1;  s_k = s_kp1;  sigma_k = sigma_kp1;

    // Overall: 2jnm + 4 * nnz(L) * m + 14nm^2 + 3m^3 + 5nm
}
