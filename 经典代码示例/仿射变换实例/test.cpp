#include "warpaffine.h"
#include "cuda_utils.h"

int main(int argc, char *argv[]){
    cv::Mat src_img = cv::imread("../test1.jpg"); // 读取测试图片

    int dst_width = 640, dst_height = 640; // 设置仿射变换后图片的尺寸
    cv::Mat dst_img(cv::Size(dst_width, dst_height), CV_8UC3); // 仿射变换后保存的图片
    
    // 申请CUDA内存
    uint8_t* psrc_device = nullptr; // 输入数据内存
    uint8_t* pdst_device = nullptr; // 输出数据内存
    size_t src_size = src_img.rows * src_img.cols * 3;
    size_t dst_size = dst_width * dst_height * 3;
    CUDA_CHECK(cudaMalloc(&psrc_device, src_size));
    CUDA_CHECK(cudaMalloc(&pdst_device, dst_size));
    CUDA_CHECK(cudaMemcpy(psrc_device, src_img.data, src_size, cudaMemcpyHostToDevice));

    // 执行仿射变换
    warpaffine(psrc_device, src_img.cols, src_img.rows, pdst_device, dst_width, dst_height);

    // 将变换结果拷贝回CPU
    CUDA_CHECK(cudaMemcpy(dst_img.data, pdst_device, dst_size, cudaMemcpyDeviceToHost));

    // 释放内存
    CUDA_CHECK(cudaFree(psrc_device));
    CUDA_CHECK(cudaFree(pdst_device));

    // 展示仿射变换后的结果
    cv::imshow("output", dst_img);
    cv::waitKey(0);
    return 0;
}

