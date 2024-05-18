#ifndef LIBGRID_H
#define LIBGRID_H

#include <cuComplex.h>

#define IMAGE_SIZE 2048
#define IMAGE_SIZE_HALF 1024
#define THETA 0.0125
#define C 299792458
#define THETA_OVER_C 4.16955e-11

#define TIMESTEPS 512
#define BASELINES 351
#define FREQUENCS 256

extern "C" void gridding_cuda(cuDoubleComplex *grid, double *uvwt, cuDoubleComplex *vist, double *freq);

#endif // LIBGRID_H
