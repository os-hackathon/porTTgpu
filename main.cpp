
#include "tensor_transpose.h"
#include "kernels.h"
#include <cutensor.h>

int main() {

  int Nx = 512;
  int Ny = 512;
  int Nz = 512;

  // Calculate grid size
  int bSize = 8;
  int nBx = Nx/bSize;
  int nBy = Ny/bSize;
  int nBz = Nz;
  printf("n blocks : %d\n",nBx);
  
  float *dIn_cpu;
  float *dOut_cpu;
  float *dIn_gpu;
  float *dOut_gpu;

  // Set up host data
  dIn_cpu = (float*)malloc(Nx*Ny*Nz*sizeof(float));
  dOut_cpu = (float*)malloc(Nx*Ny*Nz*sizeof(float));

  // Initialize input
  for( int k=0; k<Nz; k++ ){
    for( int j=0; j<Ny; j++ ){
      for( int i=0; i<Nx; i++ ){
         dIn_cpu[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = T3D_INDEX(i,j,k,Nx,Ny,Nz);
      }
    }
  }
  
  // Set up device data
  hipMalloc(&dIn_gpu, Nx*Ny*Nz*sizeof(float));
  hipMalloc(&dOut_gpu, Nx*Ny*Nz*sizeof(float));

  // Copy dIn from host to device
  hipMemcpy(dIn_gpu,dIn_cpu,Nx*Ny*Nz*sizeof(float),hipMemcpyHostToDevice);

  // Call the transpose routine
  
  // Loop to obtain timing results
  for( int i=0; i<1000; i++ ){
    transpose_021_fr<<<dim3(nBx,nBy,nBz),dim3(bSize,bSize,1)>>>(dIn_gpu,dOut_gpu,Nx,Ny,Nz);
  }

  for( int i=0; i<1000; i++ ){
    transpose_102_fr<<<dim3(nBx,nBy,nBz),dim3(bSize,bSize,1)>>>(dIn_gpu,dOut_gpu,Nx,Ny,Nz);
  }

  for( int i=0; i<1000; i++ ){
    transpose_120_fr<<<dim3(nBx,nBy,nBz),dim3(bSize,bSize,1)>>>(dIn_gpu,dOut_gpu,Nx,Ny,Nz);
  }

  for( int i=0; i<1000; i++ ){
    transpose_201_fr<<<dim3(nBx,nBy,nBz),dim3(bSize,bSize,1)>>>(dIn_gpu,dOut_gpu,Nx,Ny,Nz);
  }

  for( int i=0; i<1000; i++ ){
    transpose_210_fr<<<dim3(nBx,nBy,nBz),dim3(bSize,bSize,1)>>>(dIn_gpu,dOut_gpu,Nx,Ny,Nz);
  }

  for( int i=0; i<1000; i++ ){
    copy_fr<<<dim3(nBx,nBy,nBz),dim3(bSize,bSize,1)>>>(dIn_gpu,dOut_gpu,Nx,Ny,Nz);
  }
  //
  // Set up cuTensor for a 201 transpose
  cutensorStatus_t err;
  cutensorHandle_t handle;
  cutensorInit(&handle);
  cutensorTensorDescriptor_t descA,descB;
  float one=1.0;
  int64_t strideA[3],strideB[3];
  int64_t extA[3],extB[3];
  int imo1[] = {0,1,2};
  int imc[] = {0,2,1};
  extA[0] = Nx;
  extA[1] = Ny;
  extA[2] = Nz;
  extB[0] = Nz;
  extB[1] = Nx;
  extB[2] = Ny;
  strideA[0] = 1;
  strideA[1] = Nx;
  strideA[2] = Nx*Ny;
  strideB[0] = 1;
  strideB[1] = Nz;
  strideB[2] = Nz*Nx;

  err = cutensorInitTensorDescriptor( &handle, &descA,3,extA,strideA,CUDA_R_32F,CUTENSOR_OP_IDENTITY);
  err = cutensorInitTensorDescriptor( &handle, &descB,3,extB,strideB,CUDA_R_32F,CUTENSOR_OP_IDENTITY);
  if(err != CUTENSOR_STATUS_SUCCESS)
    printf("Error while creating tensor descriptor\n");

  for( int i=0; i<1000; i++ ){
    err = cutensorPermutation( &handle, &one, dIn_gpu, &descA, imo1, dOut_gpu, &descB, imc,CUDA_R_32F, 0);
  }
  if(err != CUTENSOR_STATUS_SUCCESS)
    printf("Error while permuting tensor: %s\n",cutensorGetErrorString(err));


  const char *error = hipGetErrorString(hipGetLastError());
  printf(error);

  // Copy dOut from device to host
  hipMemcpy(dOut_cpu,dOut_gpu,Nx*Ny*Nz*sizeof(float),hipMemcpyDeviceToHost);
 
  hipDeviceSynchronize();

  // Free device pointers
  hipFree(dIn_gpu);
  hipFree(dOut_gpu);

  // Free host pointers
  free(dIn_cpu);
  free(dOut_cpu);
  
  return 0;
}

