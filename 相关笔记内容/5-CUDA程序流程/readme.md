# 1-CUDA程序流程
```
1.设置GPU设备
2.分配主机和设备内存
3.初始化主机中的数据
4.数据从主机复制到设备
5.调用核函数在设备中进行计算
6.计算结果从设备复制到主机
7.释放主机和设备内存
```
# 2-使用的运行时API
1.设置GPU设备
```C++
// 获取GPU设备的数量
// __host__ __device__cudaError_t cudaGetDeviceCount(int *count)
int iDeviceCount = 0;
cudaGetDeviceCount(&iDeviceCount);
// 设置使用的GPU设备
// __host__ cudaError_t cudaSetDevice(int device)
int iDev = 0; // 使用GPU0
cudaSetDevice(iDev);
```

2.内存管理
```C++
// 分配设备内存
// __host__ __device__ cudaError_t cudaMalloc(void** devPtr, size_t size)
float *fpDevice_A;
cudaMalloc((float**)&fpDevice_A, nBytes);

// 数据拷贝
// __host__cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
cudaMemcpy(Device_A, Host_A, nBytes, cudaMemcpyHostToHost);
// kind
// cudaMemcpyHostToHost 主机→主机
// cudaMemcpyHostToDevice 主机→设备
// cudaMemcpyDeviceToHost 设备→主机
// cudaMemcpyDeviceToDevice 设备→设备
// cudaMemcpyDefault 默认方式，根据提供的指针自动判断，只允许在支持统一虚拟寻址的系统上使用
```

3.内存初始化
```C++
// 设备内存初始化
// __host__ cudaError_t cudaMemset(void* devPtr, int value, size_t count)
cudaMemset(fpDevice_A, 0, nBytes);
```

4.内存释放
```C++
// 释放设备内存
cudaFree(pDevice_A);
// __host__ __device__ cudaError_t cudaFree(void* devPtr)
```

5.函数类型
```
1.设备函数
    定义只能执行在GPU设备上的函数为设备函数。
    设备函数只能被核函数或其他设备函数调用。
    设备函数用__device__修饰
2.核函数
    用__global__修饰的函数称为核函数，一般由主机调用，在设备中执行。
    __global__修饰符既不能和__host__同时使用，也不可与__device__同时使用
3.主机函数
    主机端的普通C++函数可用__host__修饰。
    对于主机端的函数，__host__修饰符可省略。
    可以用__host__和__device__同时修饰一个函数减少冗余代码。编译器会针对主机和设备分别编译该函数。
```