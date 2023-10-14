// nvcc checkDimension.cu -o checkDimension

#include "./common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

// 在设备端检查网格和线程块的索引和维度
__global__ void checkIndex(void){
    printf("threadIdx:(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
    printf("blockIdx:(%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z);
    printf("blockDim:(%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
    printf("gridDim:(%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);
}

int main(int argc, char **argv){
    int nElem = 6; // 数据量
    dim3 block(3); // 主机端定义块的维度，即一个块有3个线程
    dim3 grid((nElem + block.x - 1) / block.x); // 基于数据量和块来计算网格的维度

    // 在主机端检查网格和线程块的索引
    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);

    checkIndex<<<grid, block>>>();     // 主机端调用核函数
    CHECK(cudaDeviceReset()); // 释放

    return(0);
}
