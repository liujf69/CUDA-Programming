// nvcc -arch sm_20 hello.cu -o hello // 开关语句-arch sm_20 使编译器为Fermi架构生成设备代码

#include "./common/common.h"
#include <stdio.h>

__global__ void helloFromGPU(){ // 修饰符__global__告诉编译器从CPU调用函数，然后在GPU上执行 
    printf("Hello Wrold from GPU thread %d!\n", threadIdx.x); // threadIdx.x表示线程在当前块中的位置索引
}

int main(int argc, char **argv){
    printf("Hello World from CPU!\n");
    helloFromGPU<<<1, 10>>>(); // 使用1个线程块，10个GPU线程来调用
    CHECK(cudaDeviceReset()); // 显示释放和清空与当前进程中与当前设备有关的所有资源
    return 0;
}
