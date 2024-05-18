#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>
#include <cstdio>

#include "libgrid.h"


__global__ void gridding_cuda_kernel(cuDoubleComplex *grid, double *uvwt, cuDoubleComplex *vist, double *freq) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    int k;

    for (k = 0; k < FREQUENCS; k++) {
        cuDoubleComplex vis = vist[(i * BASELINES * FREQUENCS) + (j * FREQUENCS) + k];
        double f = freq[k];

        int iu = (int)round(THETA_OVER_C * uvwt[(i * BASELINES * 3) + (j * 3) + 0] * f);
        int iv = (int)round(THETA_OVER_C * uvwt[(i * BASELINES * 3) + (j * 3) + 1] * f);
        int iu_idx = iu + IMAGE_SIZE_HALF;
        int iv_idx = iv + IMAGE_SIZE_HALF;

        atomicAdd(&(grid[iu_idx * IMAGE_SIZE + iv_idx].x), cuCreal(vis));
        atomicAdd(&(grid[iu_idx * IMAGE_SIZE + iv_idx].y), cuCimag(vis));

    }
}

extern "C" void gridding_cuda(cuDoubleComplex *grid, double *uvwt, cuDoubleComplex *vist, double *freq) {
    cuDoubleComplex *d_grid;
    double *d_uvwt;
    cuDoubleComplex *d_vist;
    double *d_freq;

    cudaMalloc(&d_grid, IMAGE_SIZE * IMAGE_SIZE * sizeof(cuDoubleComplex));
    cudaMalloc(&d_uvwt, TIMESTEPS * BASELINES * 3 * sizeof(double));
    cudaMalloc(&d_vist, TIMESTEPS * BASELINES * FREQUENCS * sizeof(cuDoubleComplex));
    cudaMalloc(&d_freq, FREQUENCS * sizeof(double));

    cudaMemcpy(d_grid, grid, IMAGE_SIZE * IMAGE_SIZE * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uvwt, uvwt, TIMESTEPS * BASELINES * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vist, vist, TIMESTEPS * BASELINES * FREQUENCS * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_freq, freq, FREQUENCS * sizeof(double), cudaMemcpyHostToDevice);

    gridding_cuda_kernel<<<TIMESTEPS, BASELINES>>>(d_grid, d_uvwt, d_vist, d_freq);

    cudaMemcpy(grid, d_grid, IMAGE_SIZE * IMAGE_SIZE * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    cudaFree(d_grid);
    cudaFree(d_uvwt);
    cudaFree(d_vist);
    cudaFree(d_freq);
}

