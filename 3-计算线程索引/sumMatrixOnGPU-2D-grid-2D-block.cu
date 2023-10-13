#include "./common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

void initialData(float *ip, const int size){
    int i;
    for(i = 0; i < size; i++){
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
    return;
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny){
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy < ny; iy++){
        for (int ix = 0; ix < nx; ix++){
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx;
        ib += nx;
        ic += nx;
    }
    return;
}

void checkResult(float *hostRef, float *gpuRef, const int N){
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < N; i++){
        if (abs(hostRef[i] - gpuRef[i]) > epsilon){
            match = 0;
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
            break;
        }
    }
    if (match) printf("Arrays match.\n\n");
    else printf("Arrays do not match.\n\n");
}

__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx, int ny){
    // 源计算代码
    // unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    // unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    // unsigned int idx = iy * nx + ix;
    // if (ix < nx && iy < ny) MatC[idx] = MatA[idx] + MatB[idx];

    // 使用基本准则计算线程索引
    int threadidx = threadIdx.x + threadIdx.y*blockDim.x; // 1.计算单个线程在线程块的索引 threadidx
    int blockidx = blockIdx.x + blockIdx.y*gridDim.x; // 2.计算线程所在线程块在全部线程块的索引 blockidx
    unsigned int idx = threadidx + blockidx*(blockDim.x*blockDim.y); // 3.线程的全局索引: idx = threadidx + blockidx*单个线程块的线程数
    MatC[idx] = MatA[idx] + MatB[idx];
}

int main(int argc, char **argv){

    // 设置使用的GPU
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // 设置测试的矩阵数据
    int nx = 1 << 14;
    int ny = 1 << 14;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // 申请CPU内存
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // 初始化数据
    double iStart = seconds();
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    double iElaps = seconds() - iStart;
    printf("Matrix initialization elapsed %f sec\n", iElaps);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // 在主机端生成测试结果，便于和设备端的结果比较
    iStart = seconds();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    iElaps = seconds() - iStart;
    printf("sumMatrixOnHost elapsed %f sec\n", iElaps);

    // 设备端
    float *d_MatA, *d_MatB, *d_MatC;
    CHECK(cudaMalloc((void **)&d_MatA, nBytes));
    CHECK(cudaMalloc((void **)&d_MatB, nBytes));
    CHECK(cudaMalloc((void **)&d_MatC, nBytes));
    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    iStart = seconds();
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x, grid.y, block.x, block.y, iElaps);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

    // 检查结果
    checkResult(hostRef, gpuRef, nxy);

    // 释放
    CHECK(cudaFree(d_MatA));
    CHECK(cudaFree(d_MatB));
    CHECK(cudaFree(d_MatC));
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    CHECK(cudaDeviceReset());

    return (0);
}
