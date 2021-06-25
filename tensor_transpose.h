#include <hip/hip_runtime.h>

#define T3D_INDEX(i,j,k,Nx,Ny,Nz) i+Nx*(j+Ny*k)

#define BLOCK_DIM 16
