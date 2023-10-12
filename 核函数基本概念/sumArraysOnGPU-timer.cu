// nvcc sumArraysOnGPU-timer.cu -o sumArraysOnGPU-timer

#include "./common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

// 检查执行结果
void checkResult(float *hostRef, float *gpuRef, const int N){
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < N; i++){
        if (abs(hostRef[i] - gpuRef[i]) > epsilon){
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
                   gpuRef[i], i);
            break;
        }
    }
    if (match) printf("Arrays match.\n\n");
    return;
}

// 随机初始化数据
void initialData(float *ip, int size){
    time_t t;
    srand((unsigned) time(&t));
    for (int i = 0; i < size; i++){
        ip[i] = (float)( rand() & 0xFF ) / 10.0f;
    }
    return;
}

// 主机端数组加法函数
void sumArraysOnHost(float *A, float *B, float *C, const int N){
    for (int idx = 0; idx < N; idx++){
        C[idx] = A[idx] + B[idx];
    }
}

// 设备端数组加法核函数 
__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x; // 计算每个线程的全局索引
    if (i < N) C[i] = A[i] + B[i];
}

int main(int argc, char **argv){
    printf("%s Starting...\n", argv[0]);

    // 设置使用的GPU
    int dev = 0; // GPU 0
    cudaDeviceProp deviceProp; // GPU设备的信息
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name); // 打印GPU的信息
    CHECK(cudaSetDevice(dev));

    // 设置数据量
    int nElem = 1 << 24;
    printf("Vector size %d\n", nElem);

    // 申请CPU内存
    size_t nBytes = nElem * sizeof(float);
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A     = (float *)malloc(nBytes);
    h_B     = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef  = (float *)malloc(nBytes);

    double iStart, iElaps; // 开始时间和结束时间

    // 在主机端初始化数据
    iStart = seconds(); // 计时函数
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    iElaps = seconds() - iStart;
    printf("initialData Time elapsed %f sec\n", iElaps); // 打印初始化数据的时间
    memset(hostRef, 0, nBytes);
    memset(gpuRef,  0, nBytes);

    // 在主机端调用检查代码
    iStart = seconds();
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    iElaps = seconds() - iStart;
    printf("sumArraysOnHost Time elapsed %f sec\n", iElaps); // 打印时间

    // 申请GPU内存
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // 数据从主机端拷贝到设备端
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));

    // 设置网格和线程块的维度
    int iLen = 512;
    dim3 block (iLen);
    dim3 grid  ((nElem + block.x - 1) / block.x);

    // 调用核函数执行
    iStart = seconds();
    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumArraysOnGPU <<<  %d, %d  >>>  Time elapsed %f sec\n", grid.x,
           block.x, iElaps);

    // 检查核函数是否错误
    CHECK(cudaGetLastError()) ;

    // 将核函数执行结果从GPU拷贝回CPU
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // 检查执行结果
    checkResult(hostRef, gpuRef, nElem);

    // 释放GPU内存
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    // 释放CPU内存
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return(0);
}
