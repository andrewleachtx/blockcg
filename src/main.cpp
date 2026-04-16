#include "matrix_market.h"

#include <iostream>
#include <solvers.h>
#include <stdexcept>
#include <string>
#include <filesystem>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>

using Eigen::MatrixXd, Eigen::VectorXd, Eigen::SparseMatrix;
using std::cerr, std::cout, std::cin, std::endl, std::string, std::exception;

static bool ends_with_mtx(const string& s)
{
    return s.size() >= 4 && s.compare(s.size() - 4, 4, ".mtx") == 0;
}

static bool is_valid_mode(const string& mode)
{
    return mode == "cg" ||mode == "pcg" || mode == "bcg" || mode == "pbcg" || mode == "test_all";
}

void test_all(const SparseMatrix<double>& A) {
    //potential experiments: 
    // vary m
    // Vary A
    // singular B
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);

    std::ostringstream oss;
    oss << "./logs/XCG_batches_" << std::put_time(std::localtime(&t), "%Y%m%d_%H%M%S");

    std::filesystem::create_directories(oss.str());
    std::filesystem::path output_dir = oss.str();

    const int n = static_cast<int>(A.cols());
    const int m = 3; // Just hardcoded for now
    printf("Using m = %d right-hand sides\n", m);
    const MatrixXd B = MatrixXd::Random(n, m);


    const MatrixXd X_0 = solve_cg_per_b(A, B, EPSILON, output_dir);
    const MatrixXd X_1 = solve_pcg_per_b(A, B, EPSILON, output_dir);
    const MatrixXd X_2 = solve_bcg(A, B, EPSILON, output_dir);
    const MatrixXd X_init = MatrixXd::Zero(n, m);

    // TODO: pick a better preconditioner
    Eigen::IncompleteCholesky<double, Eigen::Lower, Eigen::NaturalOrdering<int>> IC(A);

    const MatrixXd X_3 = solve_preconditioned_bcg(A, B, X_init, IC, output_dir);
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
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);

        std::ostringstream oss;
        oss << "./logs/single_run/" << mode <<"/" << std::put_time(std::localtime(&t), "%Y%m%d_%H%M%S");

        if (mode != "test_all") {
            std::filesystem::create_directories(oss.str());
        }
        std::filesystem::path output_dir = oss.str();

        printf("Running mode: %s\n", mode.c_str());
        if (mode != "test_all") {
            printf("Using m = %d right-hand sides\n", m);
        }
        if (mode == "cg") {
            const MatrixXd B = MatrixXd::Random(n, m);
            const MatrixXd X = solve_cg_per_b(A, B, EPSILON, output_dir);

            // fun fact %e is exponential or scientific notation, e.g., 2.943301e-08
            printf("CG residual norm: %e\n", (A * X - B).norm());
        }
        else if (mode == "pcg") {
            const MatrixXd B = MatrixXd::Random(n, m);
            const MatrixXd X = solve_pcg_per_b(A, B, EPSILON, output_dir);

            printf("PCG residual norm: %e\n", (A * X - B).norm());
        }
        else if (mode == "bcg") {
            const MatrixXd B = MatrixXd::Random(n, m);
            const MatrixXd X = solve_bcg(A, B, EPSILON, output_dir);
            printf("BCG residual norm: %e\n", (A * X - B).norm());
        }
        else if (mode == "pbcg") {
            const MatrixXd B = MatrixXd::Random(n, m);
            const MatrixXd X_0 = MatrixXd::Zero(n, m);

            // TODO: pick a better preconditioner
            Eigen::IncompleteCholesky<double, Eigen::Lower, Eigen::NaturalOrdering<int>> IC(A);

            const MatrixXd X = solve_preconditioned_bcg(A, B, X_0, IC, output_dir);
            printf("PBCG residual norm: %e\n", (A * X - B).norm());
        }
        else if (mode == "test_all") {
            test_all(A);
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
