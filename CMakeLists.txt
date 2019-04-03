cmake_minimum_required(VERSION 3.12)
project(openmptest CXX)
set (CMAKE_CXX_STANDARD 11)

add_executable(bmp BMP.cpp BMP_test.cpp)

# Find OpenMP
find_package(OpenMP REQUIRED)

if(NOT TARGET OpenMP::OpenMP_CXX)
    add_library(OpenMP_TARGET INTERFACE)
    add_library(OpenMP::OpenMP_CXX ALIAS OpenMP_TARGET)
    target_compile_options(OpenMP_TARGET INTERFACE ${OpenMP_CXX_FLAGS})
    find_package(Threads REQUIRED)
    target_link_libraries(OpenMP_TARGET INTERFACE Threads::Threads)
    target_link_libraries(OpenMP_TARGET INTERFACE ${OpenMP_CXX_FLAGS})
endif()

target_link_libraries(bmp PRIVATE OpenMP::OpenMP_CXX)
