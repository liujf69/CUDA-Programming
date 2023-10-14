cmake_minimum_required(VERSION 3.10) # cmake version
project(test) # project name
set(CMAKE_CXX_STANDARD 11) # c++11
set(CMAKE_BUILD_TYPE Debug) # Debug

# cuda
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc) 
enable_language(CUDA)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# opnecv
find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS} ./include)

add_executable(test test.cpp ${PROJECT_SOURCE_DIR}/src/warpaffine.cu) # executable
target_link_libraries(test ${OpenCV_LIBS} cudart) # lib