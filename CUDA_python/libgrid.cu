#include <cuda_runtime.h>
#include <complex>
#include <cmath>
#include <cstdio>

#include "libgrid.h"


__global__ void gridding_cuda_kernel(double *grid_real, double *grid_imag, double *uvw_data, double *visibility_real, double *visibility_imag, double *frequency_data) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    int k;

    for (k = 0; k < FREQUENCS; k++) {
        double vis_real = visibility_real[(i * BASELINES * FREQUENCS) + (j * FREQUENCS) + k];
        double vis_imag = visibility_imag[(i * BASELINES * FREQUENCS) + (j * FREQUENCS) + k];
        double freq = frequency_data[k];

        int iu = (int)round(THETA_OVER_C * uvw_data[(i * BASELINES * 3) + (j * 3) + 0] * freq);
        int iv = (int)round(THETA_OVER_C * uvw_data[(i * BASELINES * 3) + (j * 3) + 1] * freq);
        int iu_idx = iu + IMAGE_SIZE / 2;
        int iv_idx = iv + IMAGE_SIZE / 2;

        atomicAdd(&(grid_real[iu_idx * IMAGE_SIZE + iv_idx]), vis_real);
        atomicAdd(&(grid_imag[iu_idx * IMAGE_SIZE + iv_idx]), vis_imag);
    }
}


#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <complex>
#include "gridding.h"

#define IMAGE_SIZE 2048
#define IMAGE_SIZE_HALF 1024
#define THETA 0.0125
#define C 299792458
#define THETA_OVER_C 4.16955e-11

#define TIMESTEPS 512
#define BASELINES 351
#define FREQUENCS 256


__global__ void gridding_cuda_kernel(double *grid_real, double *grid_imag, double *uvw_data, double *visibility_real, double *visibility_imag, double *frequency_data) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    int k;

    for (k = 0; k < FREQUENCS; k++) {
        double vis_real = visibility_real[(i * BASELINES * FREQUENCS) + (j * FREQUENCS) + k];
        double vis_imag = visibility_imag[(i * BASELINES * FREQUENCS) + (j * FREQUENCS) + k];
        double freq = frequency_data[k];

        int iu = (int)round(THETA_OVER_C * uvw_data[(i * BASELINES * 3) + (j * 3) + 0] * freq);
        int iv = (int)round(THETA_OVER_C * uvw_data[(i * BASELINES * 3) + (j * 3) + 1] * freq);
        int iu_idx = iu + IMAGE_SIZE / 2;
        int iv_idx = iv + IMAGE_SIZE / 2;

        atomicAdd(&(grid_real[iu_idx * IMAGE_SIZE + iv_idx]), vis_real);
        atomicAdd(&(grid_imag[iu_idx * IMAGE_SIZE + iv_idx]), vis_imag);
    }
}

extern "C" void gridding_cuda(std::complex<double> *grid, double *uvw_data, std::complex<double> *visibility_data, double *frequency_data) {
    double *d_grid_real, *d_grid_imag;
    double *d_uvw_data;
    double *d_visibility_real, *d_visibility_imag;
    double *d_frequency_data;

    cudaMalloc(&d_grid_real, IMAGE_SIZE * IMAGE_SIZE * sizeof(double));
    cudaMalloc(&d_grid_imag, IMAGE_SIZE * IMAGE_SIZE * sizeof(double));
    cudaMalloc(&d_uvw_data, TIMESTEPS * BASELINES * 3 * sizeof(double));
    cudaMalloc(&d_visibility_real, TIMESTEPS * BASELINES * FREQUENCS * sizeof(double));
    cudaMalloc(&d_visibility_imag, TIMESTEPS * BASELINES * FREQUENCS * sizeof(double));
    cudaMalloc(&d_frequency_data, FREQUENCS * sizeof(double));

    double *grid_real = (double*) malloc(IMAGE_SIZE * IMAGE_SIZE * sizeof(double));
    double *grid_imag = (double*) malloc(IMAGE_SIZE * IMAGE_SIZE * sizeof(double));
    for (int idx = 0; idx < IMAGE_SIZE * IMAGE_SIZE; ++idx) {
        grid_real[idx] = std::real(grid[idx]);
        grid_imag[idx] = std::imag(grid[idx]);
    }

    double *visibility_real = (double*) malloc(TIMESTEPS * BASELINES * FREQUENCS * sizeof(double));
    double *visibility_imag = (double*) malloc(TIMESTEPS * BASELINES * FREQUENCS * sizeof(double));
    for (int idx = 0; idx < TIMESTEPS * BASELINES * FREQUENCS; ++idx) {
        visibility_real[idx] = std::real(visibility_data[idx]);
        visibility_imag[idx] = std::imag(visibility_data[idx]);
    }

    cudaMemcpy(d_grid_real, grid_real, IMAGE_SIZE * IMAGE_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid_imag, grid_imag, IMAGE_SIZE * IMAGE_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uvw_data, uvw_data, TIMESTEPS * BASELINES * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_visibility_real, visibility_real, TIMESTEPS * BASELINES * FREQUENCS * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_visibility_imag, visibility_imag, TIMESTEPS * BASELINES * FREQUENCS * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frequency_data, frequency_data, FREQUENCS * sizeof(double), cudaMemcpyHostToDevice);

    gridding_cuda_kernel<<<TIMESTEPS, BASELINES>>>(d_grid_real, d_grid_imag, d_uvw_data, d_visibility_real, d_visibility_imag, d_frequency_data);

    cudaMemcpy(grid_real, d_grid_real, IMAGE_SIZE * IMAGE_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(grid_imag, d_grid_imag, IMAGE_SIZE * IMAGE_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    for (int idx = 0; idx < IMAGE_SIZE * IMAGE_SIZE; ++idx) {
        grid[idx] = std::complex<double>(grid_real[idx], grid_imag[idx]);
    }

    cudaFree(d_grid_real);
    cudaFree(d_grid_imag);
    cudaFree(d_uvw_data);
    cudaFree(d_visibility_real);
    cudaFree(d_visibility_imag);
    cudaFree(d_frequency_data);

    free(grid_real);
    free(grid_imag);
    free(visibility_real);
    free(visibility_imag);
}

