cmake_minimum_required(VERSION 3.14)
project(Test1 LANGUAGES CUDA) # 添加支持CUDA语言

add_executable(sumMatrixOnGPU-2D-grid-2D-block sumMatrixOnGPU-2D-grid-2D-block.cu)
target_link_libraries(sumMatrixOnGPU-2D-grid-2D-block -lglut -lGLU -lGL)

add_executable(checkThreadIndex checkThreadIndex.cu)
target_link_libraries(checkThreadIndex -lglut -lGLU -lGL)