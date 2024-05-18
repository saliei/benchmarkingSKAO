#include <mpi.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>
#include <cstdio>

#include "libgrid.h"

__global__ void gridding_cuda_kernel(cuDoubleComplex *grid, double *uvw_data, cuDoubleComplex * visibility_data, double * frequency_data, int timesteps_start, int timesteps_end) {
    int timestep = timesteps_start + blockIdx.x;
    int baseline = blockIdx.y;
    int freq = threadIdx.x;

    cuDoubleComplex vis = visibility_data[(timestep * BASELINES * FREQUENCS) + (baseline * FREQUENCS) + freq];
    double frequency = frequency_data[freq];

    int iu = (int)round(THETA_OVER_C * uvw_data[(timestep * BASELINES * 3) + (baseline * 3) + 0] * frequency);
    int iv = (int)round(THETA_OVER_C * uvw_data[(timestep * BASELINES * 3) + (baseline * 3) + 1] * frequency);
    int iu_idx = iu + IMAGE_SIZE / 2;
    int iv_idx = iv + IMAGE_SIZE / 2;

    atomicAdd(&(grid[iu_idx * IMAGE_SIZE + iv_idx].x), cuCreal(vis));
    atomicAdd(&(grid[iu_idx * IMAGE_SIZE + iv_idx].y), cuCimag(vis));
}

extern "C" void gridding_cuda_mpi(std::complex<double> * grid, double * uvw_data, std::complex<double> * visibility_data, double * frequency_data) {
    int rank, size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int timesteps_per_rank = TIMESTEPS / size;
    int timesteps_start = rank * timesteps_per_rank;
    int timesteps_end = (rank == size - 1) ? TIMESTEPS : timesteps_start + timesteps_per_rank;

    cuDoubleComplex *d_grid;
    double *d_uvw_data;
    cuDoubleComplex *d_visibility_data;
    double *d_frequency_data;

    // Allocate memory on the GPU
    cudaMalloc(&d_grid, IMAGE_SIZE * IMAGE_SIZE * sizeof(cuDoubleComplex));
    cudaMalloc(&d_uvw_data, TIMESTEPS * BASELINES * 3 * sizeof(double));
    cudaMalloc(&d_visibility_data, TIMESTEPS * BASELINES * FREQUENCS * sizeof(cuDoubleComplex));
    cudaMalloc(&d_frequency_data, FREQUENCS * sizeof(double));

    // Initialize grid on the GPU
    cudaMemset(d_grid, 0, IMAGE_SIZE * IMAGE_SIZE * sizeof(cuDoubleComplex));

    // Copy data to the GPU
    cudaMemcpy(d_uvw_data, uvw_data, TIMESTEPS * BASELINES * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_visibility_data, visibility_data, TIMESTEPS * BASELINES * FREQUENCS * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frequency_data, frequency_data, FREQUENCS * sizeof(double), cudaMemcpyHostToDevice);

    // Launch the CUDA kernel
    dim3 gridDim(timesteps_per_rank, BASELINES);
    dim3 blockDim(FREQUENCS);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    gridding_cuda_kernel<<<gridDim, blockDim>>>(d_grid, d_uvw_data, d_visibility_data, d_frequency_data, timesteps_start, timesteps_end);

    // Record the stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    //printf("Rank %d: CUDA kernel execution time: %f ms\n", rank, milliseconds);

    // Copy results back to the CPU
    cuDoubleComplex *h_grid = (cuDoubleComplex*) malloc(IMAGE_SIZE * IMAGE_SIZE * sizeof(cuDoubleComplex));
    cudaMemcpy(h_grid, d_grid, IMAGE_SIZE * IMAGE_SIZE * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    // Reduce the results across all ranks
    cuDoubleComplex *global_grid = (rank == 0) ? (cuDoubleComplex*) malloc(IMAGE_SIZE * IMAGE_SIZE * sizeof(cuDoubleComplex)) : nullptr;
    MPI_Reduce(h_grid, global_grid, IMAGE_SIZE * IMAGE_SIZE * sizeof(cuDoubleComplex), MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);

    // Combine the final grid on the root rank
    if (rank == 0) {
        #pragma omp parallel for
        for (int idx = 0; idx < IMAGE_SIZE * IMAGE_SIZE; ++idx) {
            grid[idx] = std::complex<double>(cuCreal(global_grid[idx]), cuCimag(global_grid[idx]));
        }

        // Free global grid memory
        free(global_grid);
    }

    // Free GPU memory
    cudaFree(d_grid);
    cudaFree(d_uvw_data);
    cudaFree(d_visibility_data);
    cudaFree(d_frequency_data);

    // Free CPU memory
    free(h_grid);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    MPI_Finalize();
}

