#include <torch/extension.h>
#include "cpab_ops.h"
#include <iostream>

at::Tensor cpab_forward(at::Tensor points_in, //[ndim, n_points] or [batch_size, ndim, n_points]
                        at::Tensor trels_in,  //[batch_size, nC, ndim, ndim+1]
                        at::Tensor nstepsolver_in, // scalar
                        at::Tensor nc_in){ // ndim length tensor
    
    // Determine if grid is matrix or tensor
    const int broadcast = (int)(points_in.dim() == 3 & points_in.size(0) == trels_in.size(0));
    
    // Problem size
    const int ndim = (broadcast) ? points_in.size(1) : points_in.size(0);
    const int nP = (broadcast) ? points_in.size(2) : points_in.size(1);
    const auto batch_size = trels_in.size(0);

    // Allocate output
    auto output = torch::zeros({batch_size, ndim, nP}, at::kCPU);
    auto newpoints = output.data<float>();
		
    // Convert to pointers
    const auto points = points_in.data<float>();
    const auto trels = trels_in.data<float>();
    const auto nstepsolver = nstepsolver_in.data<int>();
    const auto nc = nc_in.data<int>();
    
    // Call function
    cpab_forward_op(newpoints, points, trels, nstepsolver, nc,
                    ndim, nP, batch_size, broadcast);
    return output;
}

at::Tensor cpab_backward(at::Tensor points_in, // [ndim, nP] or [batch_size, ndim, n_points]
                         at::Tensor As_in, // [n_theta, nC, ndim, ndim+1]
                         at::Tensor Bs_in, // [d, nC, ndim, ndim+1]
                         at::Tensor nstepsolver_in, // scalar
                         at::Tensor nc_in){ // ndim length tensor
    // Determine if grid is matrix or tensor
    const int broadcast = (int)(points_in.dim() == 3 & points_in.size(0) == As_in.size(0));
    
    // Problem size
    const auto ndim = (broadcast) ? points_in.size(1) : points_in.size(0);
    const auto nP = (broadcast) ? points_in.size(2) : points_in.size(1);
    const auto n_theta = As_in.size(0);
    const auto d = Bs_in.size(0);
    const auto nC = Bs_in.size(1);
    
    // Allocate output
    auto output = torch::zeros({d, n_theta, ndim, nP}, at::kCPU);
    auto grad = output.data<float>();
    
    // Convert to pointers
    const auto points = points_in.data<float>();
    const auto As = As_in.data<float>();
    const auto Bs = Bs_in.data<float>();
    const auto nstepsolver = nstepsolver_in.data<int>();
    const auto nc = nc_in.data<int>();
    
    // Call function
    cpab_backward_op(grad, points, As, Bs, nstepsolver, nc,
                     n_theta, d, ndim, nP, nC, broadcast);
    return output;
}
    

// Binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cpab_forward, "Cpab transformer forward");
    m.def("backward", &cpab_backward, "Cpab transformer backward");
}
