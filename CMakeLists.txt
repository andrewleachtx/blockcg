cmake_minimum_required(VERSION 3.20)
project(blockcg LANGUAGES C CXX CUDA)

add_executable(blockcg
    src/main.cpp
    vendor/mmio/mmio.c
)
target_include_directories(blockcg PRIVATE include)
target_include_directories(blockcg SYSTEM PRIVATE vendor vendor/mmio)

target_compile_options(blockcg PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror -pedantic>
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(blockcg PRIVATE)

# add_subdirectory(external/eigen-5.0.0)
# target_link_libraries(blockcg PRIVATE Eigen3::Eigen)
add_subdirectory(vendor/eigen EXCLUDE_FROM_ALL)
target_link_libraries(blockcg PRIVATE Eigen3::Eigen)
