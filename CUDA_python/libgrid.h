#ifndef LIBGRID_H
#define LIBGRID_H

#include <complex>

#define IMAGE_SIZE 2048
#define IMAGE_SIZE_HALF 1024
#define THETA 0.0125
#define C 299792458
#define THETA_OVER_C 4.16955e-11

#define TIMESTEPS 512
#define BASELINES 351
#define FREQUENCS 256

extern "C" void gridding_cuda(std::complex<double> *grid, double *uvw_data, std::complex<double> *visibility_data, double *frequency_data);

#endif // LIBGRID_H
