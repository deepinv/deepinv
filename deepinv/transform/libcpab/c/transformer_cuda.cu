#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cpab_ops.cuh"

#define DIV_UP(a, b) (((a) + (b)-1) / (b))

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

at::Tensor cpab_cuda_forward(at::Tensor points_in, 
                             at::Tensor trels_in,  
                             at::Tensor nstepsolver_in, 
                             at::Tensor nc_in, 
                             const int broadcast,
							 at::Tensor output){
    // Problem size
    const int ndim = (broadcast) ? points_in.size(1) : points_in.size(0);
    const int nP = (broadcast) ? points_in.size(2) : points_in.size(1);
    const auto batch_size = trels_in.size(0);        
    
    // Kernel configuration
    dim3 bc((int)ceil(nP/256.0), batch_size);
    dim3 tpb(256, 1);
    
    // Launch kernel
    // We do it in this way, since dynamically allocating memory in CUDA sucks!
    if(ndim == 1){
         cpab_cuda_kernel_forward_1D<<<bc, tpb>>>(nP, batch_size,
                                                  output.data<float>(),
                                                  points_in.data<float>(),
                                                  trels_in.data<float>(),
                                                  nstepsolver_in.data<int>(),
                                                  nc_in.data<int>(),
                                                  broadcast);
	}
	if(ndim == 2){
         cpab_cuda_kernel_forward_2D<<<bc, tpb>>>(nP, batch_size,
                                                  output.data<float>(),
                                                  points_in.data<float>(),
                                                  trels_in.data<float>(),
                                                  nstepsolver_in.data<int>(),
                                                  nc_in.data<int>(),
                                                  broadcast);
	}
	if(ndim == 3){
        cpab_cuda_kernel_forward_3D<<<bc, tpb>>>(nP, batch_size,
                                               	 output.data<float>(),
                                                 points_in.data<float>(),
                                                 trels_in.data<float>(),
                                                 nstepsolver_in.data<int>(),
                                                 nc_in.data<int>(),
                                                 broadcast);
    }
    gpuErrchk( cudaPeekAtLastError() );                           
    return output;           
}

at::Tensor cpab_cuda_backward(at::Tensor points_in, 
                              at::Tensor As_in, 
                              at::Tensor Bs_in, 
                              at::Tensor nstepsolver_in,
                              at::Tensor nc_in,
                              const int broadcast,
                              at::Tensor output){
                              
    // Problem size
    const int ndim = (broadcast) ? points_in.size(1) : points_in.size(0);
    const int nP = (broadcast) ? points_in.size(2) : points_in.size(1);
    const auto n_theta = As_in.size(0);
    const auto d = Bs_in.size(0);
    const auto nC = Bs_in.size(1);
    
    // Kernel configuration
    dim3 tpb = dim3(std::min((int)nP, 128), std::min((int)n_theta, 4), std::min((int)d, 1));
    dim3 bc = dim3(DIV_UP(nP, tpb.x), DIV_UP(n_theta, tpb.y), DIV_UP(d, tpb.z));
    dim3 vtc = dim3(nP, n_theta, d);
    
    // Launch kernel
    // We do it in this way, since dynamically allocating memory in CUDA sucks!
	if(ndim == 1){
         cpab_cuda_kernel_backward_1D<<<bc, tpb>>>(vtc, n_theta, d, nP, nC,
                                                   output.data<float>(), 
                                                   points_in.data<float>(), 
                                                   As_in.data<float>(), 
                                                   Bs_in.data<float>(),
                                                   nstepsolver_in.data<int>(), 
                                                   nc_in.data<int>(),
                                                   broadcast);
	}
	if(ndim == 2){
         cpab_cuda_kernel_backward_2D<<<bc, tpb>>>(vtc, n_theta, d, nP, nC,
                                                   output.data<float>(), 
                                                   points_in.data<float>(), 
                                                   As_in.data<float>(), 
                                                   Bs_in.data<float>(),
                                                   nstepsolver_in.data<int>(), 
                                                   nc_in.data<int>(),
                                                   broadcast);
	}
 	if(ndim == 3){
         cpab_cuda_kernel_backward_3D<<<bc, tpb>>>(vtc, n_theta, d, nP, nC,
                                                   output.data<float>(), 
                                                   points_in.data<float>(), 
                                                   As_in.data<float>(), 
                                                   Bs_in.data<float>(),
                                                   nstepsolver_in.data<int>(), 
                                                   nc_in.data<int>(),
                                                   broadcast);
    }
    return output;
}
