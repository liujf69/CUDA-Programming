# 线程束
线程束是SM中基本的执行单元，一个线程束由32个连续的线程组成。在一个线程束中，所有的线程都按照单指令多线程（SIMT）的方式执行。  
如果线程块的大小（线程块中线程的总数）不是线程束的整数倍，则最后的线程束里会有些线程就不会活跃。

# 线程束分化
```C++
if(cond){
    // ...
}
else{
    // ...
}
```
假设在一个线程束中有16个线程的cond判定为true，剩下的16个线程的cond判定为false，则各有一半的线程执行if语句和else语句的语句块。这种在一个线程束中的线程执行不同的指令，被称为线程束分化。  
由于同一个线程束中所有线程在每个周期中必须执行相同的指令，则当上述线程束分化时，会导致只有一半的线程会活跃，另一半的线程会被禁用，从而导致并行效率的降低。 
使用线程束（而不是线程）来交叉存取数据，可以避免线程束分化，即以线程束为分支粒度，例如以下代码:
```C++
// 使用线程作为分支粒度会导致线程分化
__global__ void mathKernel1(float* c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    if(tid % 2 == 0){
        a = 100.0f
    }
    else{
        b = 200.0f;
    }
    c[tid] = a + b;
}

// 使用线程束作为分支粒度来避免线程分化
__global__ void mathKernel2(float* c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    if((tid / warpSize) % 2 == 0){ // 使同一个线程束的线程都执行相同的语句体
        a = 100.0f
    }
    else{
        b = 200.0f;
    }
    c[tid] = a + b;
}
```