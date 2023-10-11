cmake_minimum_required(VERSION 3.14)

project(Test1 LANGUAGES CUDA) # 添加支持CUDA语言
add_executable(main hello.cu)
target_link_libraries(main -lglut -lGLU -lGL)