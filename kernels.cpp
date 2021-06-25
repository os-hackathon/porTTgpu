
#include <hip/hip_runtime.h>
#include "tensor_transpose.h"
#include <cuComplex.h>


// ************** Float Real ************** //

__global__ void transpose_021_fr(float *dIn, float *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(i,k,j,Nx,Nz,Ny)];

  }

}

__global__ void transpose_102_fr(float *dIn, float *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(j,i,k,Ny,Nx,Nz)];

  }

}

__global__ void transpose_120_fr(float *dIn, float *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(j,k,i,Ny,Nz,Nx)];

  }

}

__global__ void transpose_210_fr(float *dIn, float *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(k,j,i,Nz,Ny,Nx)];

  }

}
/*
__global__ void transpose_210_fr_block(float *dIn, float *dOut, int Nx, int Ny, int Nz){

  // Each thread is responsible for transposing an 8x8x8 block.
  // The start of a block is calculated from the blockIdx.[x,y,z]
  // The i0, j0, k0 correspond to the starting index for the block
  size_t i0 = threadIdx.x + 8*blockIdx.x;
  size_t j0 = threadIdx.y + 8*blockIdx.y;
  size_t k0 = threadIdx.z + 8*blockIdx.z;
  int width = gridDim.x*8;

  dIn(
/*
 * do k = 0, nBz
 *   do j = 0, nBy
 *     do i = 0, nBx
 *       do lk = 0,7
 *
 *       dout(i,j,k) = din(k,j,i)
 *     enddo
 *   enddo
 * enddo
 *
  dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(k,j,i,Nz,Ny,Nx)];


}
*/
__global__ void transpose_201_fr(float *dIn, float *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(k,i,j,Nz,Nx,Ny)];

  }

}


__global__ void copy_fr(float *dIn, float *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(i,j,k,Nz,Ny,Nx)];

  }

}

// ************** Double Real ************** //

__global__ void transpose_021_dr(double *dIn, double *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(i,k,j,Nx,Nz,Ny)];

  }

}

__global__ void transpose_102_dr(double *dIn, double *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(j,i,k,Ny,Nx,Nz)];

  }

}

__global__ void transpose_120_dr(double *dIn, double *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(j,k,i,Ny,Nz,Nx)];

  }

}

__global__ void transpose_210_dr(double *dIn, double *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(k,j,i,Nz,Ny,Nx)];

  }

}

__global__ void transpose_201_dr(double *dIn, double *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(k,i,j,Nz,Nx,Ny)];

  }

}

// ************** Float Complex ************** //

__global__ void transpose_021_fc(cuFloatComplex *dIn, cuFloatComplex *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(i,k,j,Nx,Nz,Ny)];

  }

}

__global__ void transpose_102_fc(cuFloatComplex *dIn, cuFloatComplex *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(j,i,k,Ny,Nx,Nz)];

  }

}

__global__ void transpose_120_fc(cuFloatComplex *dIn, cuFloatComplex *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(j,k,i,Ny,Nz,Nx)];

  }

}

__global__ void transpose_210_fc(cuFloatComplex *dIn, cuFloatComplex *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(k,j,i,Nz,Ny,Nx)];

  }

}

__global__ void transpose_201_fc(cuFloatComplex *dIn, cuFloatComplex *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(k,i,j,Nz,Nx,Ny)];

  }

}

// ************** Double Complex ************** //

__global__ void transpose_021_dc(cuDoubleComplex *dIn, cuDoubleComplex *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(i,k,j,Nx,Nz,Ny)];

  }

}

__global__ void transpose_102_dc(cuDoubleComplex *dIn, cuDoubleComplex *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(j,i,k,Ny,Nx,Nz)];

  }

}

__global__ void transpose_120_dc(cuDoubleComplex *dIn, cuDoubleComplex *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(j,k,i,Ny,Nz,Nx)];

  }

}

__global__ void transpose_210_dc(cuDoubleComplex *dIn, cuDoubleComplex *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(k,j,i,Nz,Ny,Nx)];

  }

}

__global__ void transpose_201_dc(cuDoubleComplex *dIn, cuDoubleComplex *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(k,i,j,Nz,Nx,Ny)];

  }

}
