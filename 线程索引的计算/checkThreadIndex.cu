#include "./common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

// 打印矩阵
void printMatrix(int *C, const int nx, const int ny){
    int *ic = C;
    printf("\nMatrix: (%d.%d)\n", nx, ny);

    for (int iy = 0; iy < ny; iy++){
        for (int ix = 0; ix < nx; ix++){
            printf("%3d", ic[ix]);
        }
        ic += nx;
        printf("\n");
    }
    printf("\n");
    return;
}

__global__ void printThreadIndex(int *A, const int nx, const int ny){
    // 二维网格和二维线程块
    int threadidx = threadIdx.x + threadIdx.y*blockDim.x; // 1.计算单个线程在线程块的索引 threadidx
    int blockidx = blockIdx.x + blockIdx.y*gridDim.x; // 2.计算线程所在线程块在全部线程块的索引 blockidx
    unsigned int idx = threadidx + blockidx*(blockDim.x*blockDim.y); // 3.线程的全局索引: idx = threadidx + blockidx*单个线程块的线程数
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    // unsigned int idx = iy * nx + ix; // 源计算代码

    printf("thread_id (%d,%d) block_id (%d,%d) coordinate (%d,%d) global index"
           " %2d ival %2d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y,
           ix, iy, idx, A[idx]);
}

int main(int argc, char **argv){
    // 设置使用的GPU
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // 初始化矩阵参数
    int nx = 8;
    int ny = 6;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    int *h_A;
    h_A = (int *)malloc(nBytes);
    for (int i = 0; i < nxy; i++){
        h_A[i] = i;
    }
    printMatrix(h_A, nx, ny);

    // 设备端
    int *d_MatA;
    CHECK(cudaMalloc((void **)&d_MatA, nBytes));
    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    dim3 block(4, 2);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    printThreadIndex<<<grid, block>>>(d_MatA, nx, ny);
    CHECK(cudaGetLastError());

    // 释放
    CHECK(cudaFree(d_MatA));
    free(h_A);
    CHECK(cudaDeviceReset());
    return (0);
}
