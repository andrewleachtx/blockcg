#include "matrix_market.h"

#include <iostream>
#include <solvers.h>
#include <stdexcept>
#include <string>

using Eigen::MatrixXd, Eigen::VectorXd, Eigen::SparseMatrix;
using std::cerr, std::cout, std::cin, std::endl, std::string, std::exception;

static bool ends_with_mtx(const string& s)
{
    return s.size() >= 4 && s.compare(s.size() - 4, 4, ".mtx") == 0;
}

static bool is_valid_mode(const string& mode)
{
    return mode == "cg" ||mode == "pcg" || mode == "bcg" || mode == "pbcg";
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        cerr << "Usage: ./blockcg <mode=cg|bcg|pbcg> <matrix.mtx>\n";
        return 1;
    }

    const string mode = argv[1];
    const string matrix_path = argv[2];

    if (!is_valid_mode(mode)) {
        cerr << "Error: mode must be one of cg, bcg, or pbcg\n";
        return 1;
    }
    if (!ends_with_mtx(matrix_path)) {
        cerr << "Error: matrix path must end in .mtx\n";
        return 1;
    }

    try {
        const SparseMatrix<double> A = load_matrix_market(matrix_path);
        const int n = static_cast<int>(A.cols());
        const int m = 3; // Just hardcoded for now

        printf("Running mode: %s\n", mode.c_str());
        if (mode == "cg") {
            const MatrixXd B = MatrixXd::Random(n, m);
            const MatrixXd X = solve_cg_per_b(A, B, EPSILON);

            // fun fact %e is exponential or scientific notation, e.g., 2.943301e-08
            printf("CG residual norm: %e\n", (A * X - B).norm());
        }
        else if (mode == "pcg") {
            const MatrixXd B = MatrixXd::Random(n, m);
            const MatrixXd X = solve_pcg_per_b(A, B, EPSILON);

            printf("PCG residual norm: %e\n", (A * X - B).norm());
        }
        else if (mode == "bcg") {
            const MatrixXd B = MatrixXd::Random(n, m);
            const MatrixXd X = solve_bcg(A, B, EPSILON);
            printf("BCG residual norm: %e\n", (A * X - B).norm());
        }
        else if (mode == "pbcg") {
            const MatrixXd B = MatrixXd::Random(n, m);
            const MatrixXd X_0 = MatrixXd::Zero(n, m);

            // TODO: pick a better preconditioner
            Eigen::SimplicialLLT<SparseMatrix<double>> LLT(A);

            const MatrixXd X = solve_preconditioned_bcg(A, B, X_0, LLT);
            printf("PBCG residual norm: %e\n", (A * X - B).norm());
        }
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    // TODO write fn for and run cg
    // TODO write fn for and run bcg
    // TODO compare cg, bcg

    // TODO if time write fn for preconditioned bcg
    // TODO compare cg, bcg, pbcg
    return 0;
}
