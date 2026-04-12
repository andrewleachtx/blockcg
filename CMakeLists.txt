cmake_minimum_required(VERSION 3.20)
project(blockcg LANGUAGES C CXX)

add_executable(blockcg
    src/main.cpp
    src/solvers.cpp
    src/matrix_market.cpp
    vendor/mmio/mmio.c
)
target_include_directories(blockcg PRIVATE include)
target_include_directories(blockcg SYSTEM PRIVATE vendor vendor/mmio)

target_compile_options(blockcg PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror -pedantic>
)

add_subdirectory(vendor/eigen EXCLUDE_FROM_ALL)
target_link_libraries(blockcg PRIVATE Eigen3::Eigen)
