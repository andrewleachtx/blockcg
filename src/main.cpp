#include "matrix_market.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <solvers.h>

static bool ends_with_mtx(const std::string& s) {
    return s.size() >= 4 && s.compare(s.size() - 4, 4, ".mtx") == 0;
}

static bool is_valid_mode(const std::string& mode) {
    return mode == "cg" || mode == "bcg" || mode == "pbcg";
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./blockcg <mode=cg|bcg|pbcg> <matrix.mtx>\n";
        return 1;
    }

    const std::string mode = argv[1];
    const std::string matrix_path = argv[2];

    if (!is_valid_mode(mode)) {
        std::cerr << "Error: mode must be one of cg, bcg, or pbcg\n";
        return 1;
    }

    if (!ends_with_mtx(matrix_path)) {
        std::cerr << "Error: matrix path must end in .mtx\n";
        return 1;
    }

    try {
        const Eigen::SparseMatrix<double> A = load_matrix_market(matrix_path);
        // printf("Mode: %s\n", mode);
        std::cout << "Mode: " << mode << '\n';
        std::cout << "Matrix: " << A.rows() << "x" << A.cols() << ", "
                  << A.nonZeros() << " nonzeros\n";

        //silly lil run with some prints to show it works
        const Eigen::VectorXd b = Eigen::VectorXd::Ones(A.cols()); // or load your rhs
        Eigen::SparseMatrix<double> I(A.cols(), A.cols());
        I.setIdentity();
        const Eigen::VectorXd x = cg_solve(A, b, I);
        std::cout << "CG solution norm: " << x.norm() << '\n';
        for (int i=0; i<10;i++) {
            std::cout << "value "<< i << ": " << x[i] << '\n';
        }
        Eigen::VectorXd calc_b = A*x;
        std::cout << "computed b";
        for (int i=0; i<10;i++) {
            std::cout << "value "<< i << ": " << calc_b[i] << '\n';
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }



    // TODO write fn for and run cg
    // TODO write fn for and run bcg
    // TODO compare cg, bcg

    // TODO if time write fn for preconditioned bcg
    // TODO compare cg, bcg, pbcg
    return 0;
}
