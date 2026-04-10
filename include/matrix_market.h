#pragma once

#include <Eigen/Sparse>
#include <string>

Eigen::SparseMatrix<double> load_matrix_market(const std::string& path);
