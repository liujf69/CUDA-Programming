cmake_minimum_required(VERSION 3.14)

project(Test1 LANGUAGES CUDA) # 添加支持CUDA语言
add_executable(checkDimension checkDimension.cu)
target_link_libraries(checkDimension -lglut -lGLU -lGL)

add_executable(sumArraysOnGPU-timer sumArraysOnGPU-timer.cu)
target_link_libraries(sumArraysOnGPU-timer -lglut -lGLU -lGL)