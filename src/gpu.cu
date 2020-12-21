#include <CM/gpu.h>
#include <CM/utils.h>
#include <iostream>
#include <cuda_runtime.h>

#define LOG2 0.301029996

__global__ void compute(unsigned char* out, Helper helper) {
    double threadRowID = blockIdx.x * blockDim.x + threadIdx.x;
    double threadColID = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = threadRowID * helper.width + threadColID;

    double C_x = helper.lowerX + threadColID / double(helper.width) * helper.rangeX;
    double C_y = helper.lowerY + threadRowID / double(helper.height) * helper.rangeY;

    // do mandel comp. here
    double Z_x = 0.0;
    double Z_y = 0.0;

    double abs_ = 0.0;
    char n = 0;

    double dc_x = 1;
    double dc_y = 0;
    double der_x = dc_x;
    double der_y = dc_y;

    double Znew_x;
    double Znew_y;

    double dernew_x;
    double dernew_y;

    // check for abs squared
    while(abs_ <= 10000 && n < 64) {

        Znew_x = Z_x * Z_x - Z_y * Z_y + C_x;
        Znew_y = 2. * Z_x * Z_y + C_y;

        dernew_x = (der_x * Z_x - der_y * Z_y)*2 + dc_x;
        dernew_y = (der_y * Z_x + der_x * Z_y)*2 + dc_y;

        Z_x = Znew_x;
        Z_y = Znew_y;

        der_x = dernew_x;
        der_y = dernew_y;

        abs_ = Z_x * Z_x + Z_y * Z_y;
        n++;
    }
    if(helper.color == 1) {
        out[3*idx + 0] = n*4;
        out[3*idx + 1] = n*4;
        out[3*idx + 2] = n*4;
    }
    else if(helper.color == 2) {
        double k = LOG2;
        double x = std::log(std::log(abs_) / std::pow(2, n)) / k;

        out[3*idx+0] = 255 * (1-std::cos(1.0 / LOG2 * x * 1.0)) * 0.5;
        out[3*idx+1] = 255 * (1-std::cos(1.0 / LOG2 * x * (1.0 / (3.0 / 1.41421356))));
        out[3*idx+2] = 255 * (1-std::cos(2.0 / LOG2 * x * 0.12452650612));
    } else if(helper.color == 3) {
        double x = std::log(std::log(abs_) / std::pow(2, n)) / LOG2;
        unsigned char val = 255 * (1 + std::cos(2 * 3.141592653589 * x))*0.5;
        out[3*idx + 0] = val;
        out[3*idx + 1] = val;
        out[3*idx + 2] = val;
    } else if(helper.color == 4) {
        if(n == 64) {
            // not enough iteratores
            // inside
            out[3*idx + 0] = 0;
            out[3*idx + 1] = 0;
            out[3*idx + 2] = 255;
        } else {
            // z / der
            double u_x = (Z_x * der_x + Z_y * der_y);
            double u_y = (der_y * Z_x - der_x * Z_y);
            
            double u_norm = sqrt(u_x * u_x + u_y * u_y);
            u_x = u_x / u_norm;
            u_y = u_y / u_norm;

            double t = u_x / 1.41421356 + u_y / 1.41421356 + 1.5;
            t = t/ (1. + 1.5);
            if(t < 0) t = 0;

            out[3*idx + 0] = 255 * t;
            out[3*idx + 1] = 255 * t;
            out[3*idx + 2] = 255 * t;
        }
    } else if(helper.color == 5) {
        if(n == 64) {
            // not enough iteratores
            // inside
            out[3*idx + 0] = 0;
            out[3*idx + 1] = 0;
            out[3*idx + 2] = 255;
        } else {
            double u_x = (Z_x * der_x + Z_y * der_y);
            double u_y = (der_y * Z_x - der_x * Z_y);
            
            double u_norm = sqrt(u_x * u_x + u_y * u_y);
            u_x = u_x / u_norm;
            u_y = u_y / u_norm;

            double t = u_x * helper.V_x + u_y * helper.V_y + 1.5;
            t = t/ (1. + 1.5);
            if(t < 0) t = 0;

            out[3*idx + 0] = 255 * t;
            out[3*idx + 1] = 255 * t;
            out[3*idx + 2] = 255 * t;
        }
    } else {
        out[3*idx + 0] = n;
        out[3*idx + 1] = n;
        out[3*idx + 2] = n;
    }
}
void doCalc(unsigned char* out, Helper helper) {
    dim3 block(helper.height, 1);
    dim3 grid(1, helper.width);
    compute<<<grid, block>>>(out, helper);
}

void checkCuda()
{
    std::cout << "CUDA Compiled version: " << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << std::endl;

    int runtime_ver;
    cudaRuntimeGetVersion(&runtime_ver);
    std::cout << "CUDA Runtime version: " << runtime_ver << std::endl;

    int driver_ver;
    cudaDriverGetVersion(&driver_ver);
    std::cout << "CUDA Driver version: " << driver_ver << std::endl;
}