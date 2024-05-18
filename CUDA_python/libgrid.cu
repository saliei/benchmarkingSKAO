#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cuComplex.h>
#include <cuda_runtime.h>

#include "libgrid.h"

__global__ void gridding_kernel(cuDoubleComplex *grid, double *uvwt, cuDoubleComplex *vist, double *freq) {
    int timestep = blockIdx.x;
    int baseline = blockIdx.y;
    int fq = threadIdx.x;

    cuDoubleComplex vis = vist[(timestep * BASELINES * FREQUENCS) + (baseline * FREQUENCS) + fq];
    double f = freq[fq];

    int iu = (int)round(THETA_OVER_C * uvwt[(timestep * BASELINES * 3) + (baseline * 3) + 0] * f);
    int iv = (int)round(THETA_OVER_C * uvwt[(timestep * BASELINES * 3) + (baseline * 3) + 1] * f);
    int iu_idx = iu + IMAGE_SIZE_HALF;
    int iv_idx = iv + IMAGE_SIZE_HALF;

    atomicAdd(&(grid[iu_idx * IMAGE_SIZE + iv_idx].x), cuCreal(vis));
    atomicAdd(&(grid[iu_idx * IMAGE_SIZE + iv_idx].y), cuCimag(vis));
}

__global__ void gridding_kernel_mpi(cuDoubleComplex *grid, double *uvwt, cuDoubleComplex *vist, double *freq, int timesteps_start, int timesteps_end) {
    int timestep = timesteps_start + blockIdx.x;
    int baseline = blockIdx.y;
    int fq = threadIdx.x;

    cuDoubleComplex vis = vist[(timestep * BASELINES * FREQUENCS) + (baseline * FREQUENCS) + fq];
    double f = freq[fq];

    int iu = (int)round(THETA_OVER_C * uvwt[(timestep * BASELINES * 3) + (baseline * 3) + 0] * f);
    int iv = (int)round(THETA_OVER_C * uvwt[(timestep * BASELINES * 3) + (baseline * 3) + 1] * f);
    int iu_idx = iu + IMAGE_SIZE_HALF;
    int iv_idx = iv + IMAGE_SIZE_HALF;

    atomicAdd(&(grid[iu_idx * IMAGE_SIZE + iv_idx].x), cuCreal(vis));
    atomicAdd(&(grid[iu_idx * IMAGE_SIZE + iv_idx].y), cuCimag(vis));
}

void gridding_cuda(std::complex<double> *grid, double *uvwt, std::complex<double> *vist, double *freq) {
    cuDoubleComplex *d_grid;
    double *d_uvwt;
    cuDoubleComplex *d_vist;
    double *d_freq;

    cudaMalloc(&d_grid, IMAGE_SIZE * IMAGE_SIZE * sizeof(cuDoubleComplex));
    cudaMalloc(&d_uvwt, TIMESTEPS * BASELINES * 3 * sizeof(double));
    cudaMalloc(&d_vist, TIMESTEPS * BASELINES * FREQUENCS * sizeof(cuDoubleComplex));
    cudaMalloc(&d_freq, FREQUENCS * sizeof(double));

    // no need to copy the grid since it's zero initialized on the host
    cudaMemset(d_grid, 0, IMAGE_SIZE * IMAGE_SIZE * sizeof(cuDoubleComplex));
    cudaMemcpy(d_uvwt, uvwt, TIMESTEPS * BASELINES * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vist, vist, TIMESTEPS * BASELINES * FREQUENCS * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_freq, freq, FREQUENCS * sizeof(double), cudaMemcpyHostToDevice);

    dim3 gridDim(TIMESTEPS, BASELINES);
    dim3 blockDim(FREQUENCS);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    gridding_kernel<<<gridDim, blockDim>>>(d_grid, d_uvwt d_vist, d_freq);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    //printf("kernel execution time: %f ms\n", milliseconds);

    cudaMemcpy(grid, d_grid, IMAGE_SIZE * IMAGE_SIZE * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    cudaFree(d_grid);
    cudaFree(d_uvwt);
    cudaFree(d_vist);
    cudaFree(d_freq);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// we assume a CUDA aware MPI implementation
void gridding_cuda_mpi(std::complex<double> *grid, double *uvwt, std::complex<double> *vist, double *freq) {
    int rank, size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int timesteps_per_rank = TIMESTEPS / size;
    int timesteps_start = rank * timesteps_per_rank;
    int timesteps_end = (rank == size - 1) ? TIMESTEPS : timesteps_start + timesteps_per_rank;

    cuDoubleComplex *d_grid;
    double *d_uvwt;
    cuDoubleComplex *d_vist;
    double *d_freq;

    cudaMemset(d_grid, 0, IMAGE_SIZE * IMAGE_SIZE * sizeof(cuDoubleComplex));
    cudaMalloc(&d_grid, IMAGE_SIZE * IMAGE_SIZE * sizeof(cuDoubleComplex));
    cudaMalloc(&d_uvwt, TIMESTEPS * BASELINES * 3 * sizeof(double));
    cudaMalloc(&d_vist, TIMESTEPS * BASELINES * FREQUENCS * sizeof(cuDoubleComplex));
    cudaMalloc(&d_freq, FREQUENCS * sizeof(double));

    cudaMemcpy(d_uvwt, uvwt, TIMESTEPS * BASELINES * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vist, vist, TIMESTEPS * BASELINES * FREQUENCS * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_freq, freq, FREQUENCS * sizeof(double), cudaMemcpyHostToDevice);

    dim3 gridDim(timesteps_per_rank, BASELINES);
    dim3 blockDim(FREQUENCS);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    gridding_kernel_mpi<<<gridDim, blockDim>>>(d_grid, d_uvwt, d_vist, d_freq, timesteps_start, timesteps_end);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    //printf("rank: %d, kernel execution time: %f ms\n", rank, milliseconds);

    //cuDoubleComplex *h_grid = (cuDoubleComplex*) malloc(IMAGE_SIZE * IMAGE_SIZE * sizeof(cuDoubleComplex));
    //cudaMemcpy(h_grid, d_grid, IMAGE_SIZE * IMAGE_SIZE * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    // reduce the results across all ranks
    cuDoubleComplex *d_grid_global = nullptr;
    if (rank == 0) {
        cudaMalloc(&d_grid_global, IMAGE_SIZE * IMAGE_SIZE * sizeof(cuDoubleComplex));
    }
    //cuDoubleComplex *grid_global = (rank == 0) ? (cuDoubleComplex*) malloc(IMAGE_SIZE * IMAGE_SIZE * sizeof(cuDoubleComplex)) : nullptr;

    MPI_Reduce(d_grid, d_grid_global, IMAGE_SIZE * IMAGE_SIZE * sizeof(cuDoubleComplex), MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cudaMemcpy(grid, d_grid_global, IMAGE_SIZE * IMAGE_SIZE * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost)
    }

    cudaFree(d_grid_global);
    cudaFree(d_grid);
    cudaFree(d_uvwt);
    cudaFree(d_vist);
    cudaFree(d_freq);

    //free(h_grid);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    MPI_Finalize();
}

