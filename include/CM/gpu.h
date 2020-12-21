#pragma once
#include <cuda_runtime.h>

struct Helper {
    float upperX;
    float lowerX;
    float upperY;
    float lowerY;
    float rangeX;
    float rangeY;
    int width;
    int height;
    float zoomLevel;
    int color;

    double V_x = 0.70710678118;
    double V_y = 0.70710678118;

};

void checkCuda();

__global__ void compute(unsigned char* out, Helper helper);

void doCalc(unsigned char* out, Helper helper);

