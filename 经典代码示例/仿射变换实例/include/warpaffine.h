#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <opencv2/opencv.hpp>

void warpaffine(uint8_t* src, 
                int src_width, 
                int src_height, 
                uint8_t* dst, 
                int dst_width, 
                int dst_height);