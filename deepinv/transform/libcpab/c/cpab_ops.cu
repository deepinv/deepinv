#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

__device__ int cuda_mymin(int a, double b) {
    return !(b<a)?a:round(b);
}

__device__ double cuda_fmod(double numer, double denom){
    double tquou = floor(numer / denom);
    return numer - tquou * denom;
}

__device__ int cuda_findcellidx_1D(const float* p, const int ncx) {           
    // Floor value to find cell
    int idx = floor(p[0] * ncx);
    idx = max(0, min(idx, ncx-1));
    return idx;                            
}

__device__ int cuda_findcellidx_2D(const float* p, const int ncx, const int ncy) {
    // Copy point
    double point[2];
    point[0] = p[0];
    point[1] = p[1];
    
    // Cell size
    const float inc_x = 1.0 / ncx;
    const float inc_y = 1.0 / ncy;
    
    // Find initial row, col placement
    double p0 = min((ncx * inc_x - 0.000000001), max(0.0, point[0]));
    double p1 = min((ncy * inc_y - 0.000000001), max(0.0, point[1]));

    double xmod = cuda_fmod((double)p0, (double)inc_x);
    double ymod = cuda_fmod((double)p1, (double)inc_y);

    double x = xmod / inc_x;
    double y = ymod / inc_y;
            
    int cell_idx =  cuda_mymin(ncx-1, (p0 - xmod) / inc_x) + 
                    cuda_mymin(ncy-1, (p1 - ymod) / inc_y) * ncx;        
    cell_idx *= 4;
            
    // Out of bound (left)
    if(point[0]<=0){
        if(point[1] <= 0 && point[1]/inc_y<point[0]/inc_x){
            // Nothing to do here
        } else if(point[1] >= ncy * inc_y && point[1]/inc_y-ncy > -point[0]/inc_x) {
            cell_idx += 2;
        } else {
            cell_idx += 3;
        }
        return cell_idx;
    }
            
    // Out of bound (right)
    if(point[0] >= ncx*inc_x){
        if(point[1]<=0 && -point[1]/inc_y > point[0]/inc_x - ncx){
            // Nothing to do here
        } else if(point[1] >= ncy*inc_y && point[1]/inc_y - ncy > point[0]/inc_x-ncx){
            cell_idx += 2;
        } else {
            cell_idx += 1;
        }
        return cell_idx;
    }
            
    // Out of bound (up)
    if(point[1] <= 0){
        return cell_idx;
    }
            
    // Out of bound (bottom)
    if(point[1] >= ncy*inc_y){
        cell_idx += 2;
        return cell_idx;
    }
            
    // OK, we are inbound
    if(x<y){
        if(1-x<y){
            cell_idx += 2;
        } else {
            cell_idx += 3;
        }
    } else if(1-x<y) {
        cell_idx += 1;
    }
                                
    return cell_idx;
    /*
    // Cell size
    const float inc_x = 1.0 / nx;
    const float inc_y = 1.0 / ny;

    // Copy point                        
    float point[2];
    point[0] = p[0];
    point[1] = p[1];
    
    // If point is outside [0, 1]x[0, 1] then we push it inside
    if (point[0] < 0.0 || point[0] > 1.0 || point[1] < 0.0 || point[1] > 1.0) {
        const float half = 0.5;
        point[0] -= half;
        point[1] -= half;
        const float abs_x = abs(point[0]);
        const float abs_y = abs(point[1]);

        const float push_x = (abs_x < abs_y) ? half*inc_x : 0.0;
        const float push_y = (abs_y < abs_x) ? half*inc_y : 0.0;
        if (abs_x > half) {
            point[0] = copysign(half - push_x, point[0]);
        }
        if (abs_y > half) {
            point[1] = copysign(half - push_y, point[1]);
        }
        
        point[0] += half;
        point[1] += half;
    }
    
    // Find initial row, col placement
    const float p0 = min((float)(1.0 - 1e-8), point[0]);
    const float p1 = min((float)(1.0 - 1e-8), point[1]);
    const float p0ncx = p0*nx;
    const float p1ncy = p1*ny;
    const int ip0ncx = p0ncx; // rounds down
    const int ip1ncy = p1ncy; // rounds down
    int cell_idx = 4 * (ip0ncx + ip1ncy * nx);
    
    // Find (sub)triangle
    const float x = p0ncx - ip0ncx;
    const float y = p1ncy - ip1ncy;
    if (x < y) {
        if (1-x < y) {
            cell_idx += 2;
        } else {
            cell_idx += 3;
        }
    } else if (1-x < y) {
        cell_idx += 1;
    }
    
    return cell_idx;
    */
}

__device__ int cuda_findcellidx_3D(const float* p, const int nx, const int ny, const int nz) {
    // Cell size
    const float inc_x = 1.0 / nx;
    const float inc_y = 1.0 / ny;
    const float inc_z = 1.0 / nz;
    
    // Copy point
    float point[3];
    point[0] = p[0];
    point[1] = p[1];
    point[2] = p[2];
    
    // If point is outside [0, 1]x[0, 1]x[0, 1] then we push it inside
    if(point[0] < 0.0 || point[0] > 1.0 || point[1] < 0.0 || point[1] > 1.0) {
        const float half = 0.5;
        point[0] -= half;
        point[1] -= half;
        point[2] -= half;
        const float abs_x = abs(point[0]);
        const float abs_y = abs(point[1]);
        const float abs_z = abs(point[2]);
        
        const float push_x = (abs_x < abs_y && abs_x < abs_z) ? half*inc_x : 0.0;
        const float push_y = (abs_y < abs_x && abs_x < abs_z) ? half*inc_y : 0.0;
        const float push_z = (abs_z < abs_x && abs_x < abs_y) ? half*inc_z : 0.0;
        if(abs_x > half){point[0] = copysign(half - push_x, point[0]);}
        if(abs_y > half){point[1] = copysign(half - push_y, point[1]);}
        if(abs_z > half){point[2] = copysign(half - push_z, point[2]);}
        point[0] += half;
        point[1] += half;
        point[2] += half;
    }
    float zero = 0.0;
    float p0 = min((float)(nx*inc_x-1e-8),max(zero, point[0]));
    float p1 = min((float)(ny*inc_y-1e-8),max(zero, point[1]));       
    float p2 = min((float)(nz*inc_x-1e-8),max(zero, point[2])); 
    
    double xmod = cuda_fmod(p0,inc_x);
    double ymod = cuda_fmod(p1,inc_y);
    double zmod = cuda_fmod(p2,inc_z);
    
    int i = cuda_mymin(nx-1,((p0 - xmod)/inc_x));
    int j = cuda_mymin(ny-1,((p1 - ymod)/inc_y));
    int k = cuda_mymin(nz-1,((p2 - zmod)/inc_z));
    
    int cell_idx = 5*(i + j * nx + k * nx * ny);
    
    double x = xmod/inc_x;
    double y = ymod/inc_y;
    double z = zmod/inc_z;
    
    bool tf = false;              
    if (k%2==0){
        if ((i%2==0 && j%2==1) ||  (i%2==1 && j%2==0)){
            tf = true;
        }
    }
    else if((i%2==0 && j%2==0) ||  (i%2==1 && j%2==1)){
        tf = true;
    }              
    if (tf){
        double tmp = x;
        x = y;
        y = 1-tmp;
    }
    if (-x -y +z  >= 0){
        cell_idx+=1;
    }
    else if (x+y+z - 2 >= 0){
        cell_idx+=2;
    }
    else if (-x+y-z >= 0){
        cell_idx+=3;
    }
    else if (x-y-z >= 0){
        cell_idx+=4;
    }
    return cell_idx;
}


__device__ void A_times_b_1D(float x[], const float* A, float* b) {
    x[0] = A[0]*b[0] + A[1];
    return;
}

__device__ void A_times_b_2D(float x[], const float* A, float* b) {
    x[0] = A[0]*b[0] + A[1]*b[1] + A[2];
    x[1] = A[3]*b[0] + A[4]*b[1] + A[5];
    return;
}

__device__ void A_times_b_3D(float x[], const float* A, float* b) {
    x[0] = A[0]*b[0] + A[1]*b[1] + A[2]*b[2] + A[3];
    x[1] = A[4]*b[0] + A[5]*b[1] + A[6]*b[2] + A[7];
    x[2] = A[8]*b[0] + A[9]*b[1] + A[10]*b[2] + A[11];
    return;
}

__device__ void A_times_b_linear_1D(float x[], const float* A, float* b) {
    x[0] = A[0]*b[0];
    return;
}

__device__ void A_times_b_linear_2D(float x[], const float* A, float* b) {
    x[0] = A[0]*b[0] + A[1]*b[1];
    x[1] = A[3]*b[0] + A[4]*b[1];
    return;
}

__device__ void A_times_b_linear_3D(float x[], const float* A, float* b) {
    x[0] = A[0]*b[0] + A[1]*b[1] + A[2]*b[2];
    x[1] = A[4]*b[0] + A[5]*b[1] + A[6]*b[2];
    x[2] = A[8]*b[0] + A[9]*b[1] + A[10]*b[2];
    return;
}


// Kernel declaration
__global__ void cpab_cuda_kernel_forward_1D(const int nP, const int batch_size,
                                            float* newpoints, const float* points,
                                            const float* Trels, const int* nStepSolver,
                                            const int* nc, const int broadcast) {
    
    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    if(point_index < nP && batch_index < batch_size) {
        // Get point
        float point[1];
        point[0] = points[broadcast*batch_index*nP*1+point_index];
    
        // Define start index for the matrices belonging to this batch
        // batch * 2 params pr cell * cell in x
        int start_idx = batch_index * 2 * nc[0]; 
    
        // Iterate in nStepSolver
        int cellidx;
        for(int n = 0; n < nStepSolver[0]; n++){
            // Find cell idx
            cellidx = cuda_findcellidx_1D(point, nc[0]);
            
            // Extract the mapping in the cell
            const float* Trels_idx = Trels + 2*cellidx + start_idx;                
                     
            // Calculate trajectory of point
            float point_updated[1];                
            A_times_b_1D(point_updated, Trels_idx, point);
            point[0] = point_updated[0];
        }
    
        // Copy to output
        newpoints[nP * batch_index + point_index] = point[0];
    }
    return;                            
}

__global__ void cpab_cuda_kernel_forward_2D(const int nP, const int batch_size,
                                            float* newpoints, const float* points,
                                            const float* Trels, const int* nStepSolver,
                                            const int* nc, const int broadcast) {

    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    if(point_index < nP && batch_index < batch_size) {    
        // Get point
        float point[2];
        point[0] = points[broadcast*batch_index*nP*2+point_index];
        point[1] = points[broadcast*batch_index*nP*2+point_index + nP];
    
        // Define start index for the matrices belonging to this batch
        // batch * num_elem * 4 triangles pr cell * cell in x * cell in y
        int start_idx = batch_index * 6 * 4 * nc[0] * nc[1]; 
        
        // Iterate in nStepSolver
        int cellidx;
        for(int n = 0; n < nStepSolver[0]; n++){
            // Find cell idx
            cellidx = cuda_findcellidx_2D(point, nc[0], nc[1]);
            
            // Extract the mapping in the cell
            const float* Trels_idx = Trels + 6*cellidx + start_idx;                
                     
            // Calculate trajectory of point
            float point_updated[2];                
            A_times_b_2D(point_updated, Trels_idx, point);

            point[0] = point_updated[0];
            point[1] = point_updated[1];
        }
    
        // Copy to output
        newpoints[2 * nP * batch_index + point_index] = point[0];
        newpoints[2 * nP * batch_index + point_index + nP] = point[1];    
    }
    return;                            
}

__global__ void cpab_cuda_kernel_forward_3D(const int nP, const int batch_size,
                                            float* newpoints, const float* points, 
                                            const float* Trels, const int* nStepSolver,
                                            const int* nc, const int broadcast) {
    
    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    if(point_index < nP && batch_index < batch_size) {
        // Get point
        float point[3];
        point[0] = points[broadcast*batch_index*nP*3+point_index];
        point[1] = points[broadcast*batch_index*nP*3+point_index + nP];
        point[2] = points[broadcast*batch_index*nP*3+point_index + 2*nP];
    
        // Define start index for the matrices belonging to this batch
        // batch * 12 params pr cell * 5 triangles pr cell * cell in x * cell in y * cell in z
        int start_idx = batch_index * 12 * 5 * nc[0] * nc[1] * nc[2]; 
    
        // Iterate in nStepSolver
        int cellidx;
        for(int n = 0; n < nStepSolver[0]; n++){
            // Find cell idx
            cellidx = cuda_findcellidx_3D(point, nc[0], nc[1], nc[2]);
            
            // Extract the mapping in the cell
            const float* Trels_idx = Trels + 12*cellidx + start_idx;                
                     
            // Calculate trajectory of point
            float point_updated[3];                
            A_times_b_3D(point_updated, Trels_idx, point);

            point[0] = point_updated[0];
            point[1] = point_updated[1];
            point[2] = point_updated[2];
        }
    
        // Copy to output
        newpoints[3 * nP * batch_index + point_index] = point[0];
        newpoints[3 * nP * batch_index + point_index + nP] = point[1];
        newpoints[3 * nP * batch_index + point_index + 2 * nP] = point[2];    
    }
    return;                            
}

__global__ void cpab_cuda_kernel_backward_1D(dim3 nthreads, const int n_theta, const int d, const int nP, const int nC,
                                             float* grad, const float* points, const float* As, const float* Bs,
                                             const int* nStepSolver, const int* nc, const int broadcast) {
        
        // Allocate memory for computations
        float p[1], v[1], pMid[1], vMid[1], q[1], qMid[1];
        float B_times_T[1], A_times_dTdAlpha[1], u[1], uMid[1];
        float Alocal[2], Blocal[2];
        int cellidx;
        
        // Thread index
        int point_index = threadIdx.x + blockIdx.x * blockDim.x;
        int batch_index = threadIdx.y + blockIdx.y * blockDim.y;
        int dim_index   = threadIdx.z + blockIdx.z * blockDim.z;
        
        // Make sure we are within bounds
        if(point_index < nP && batch_index < n_theta && dim_index < d){
            int index = nP * batch_index + point_index;
            int boxsize = nP * n_theta;
        
            // Define start index for the matrices belonging to this batch
            // batch * 2 params pr cell * cell in x
            int start_idx = batch_index * 2 * nc[0]; 

            // Get point
            p[0] = points[broadcast*batch_index*nP*1+point_index];
            
            // Step size for solver
            double h = (1.0 / nStepSolver[0]);
        
            // Iterate a number of times
            for(int t=0; t<nStepSolver[0]; t++) {
                // Get current cell
                cellidx = cuda_findcellidx_1D(p, nc[0]);
                
                // Get index of A
                int As_idx = 2*cellidx;
                
                // Extract local A
                for(int i = 0; i < 2; i++){
                    Alocal[i] = (As + As_idx + start_idx)[i];
                }
                
                // Compute velocity at current location
                A_times_b_1D(v, Alocal, p);
                
                // Compute midpoint
                pMid[0] = p[0] + h*v[0]/2.0;
                
                // Compute velocity at midpoint
                A_times_b_1D(vMid, Alocal, pMid);
                
                // Get index of B
                int Bs_idx = 2 * dim_index * nC + As_idx;
                
                // Get local B
                for(int i = 0; i < 2; i++){
                    Blocal[i] = (Bs + Bs_idx)[i];
                }
                
                // Copy q
                q[0] = grad[dim_index*boxsize + index];
        
                // Step 1: Compute u using the old location
                // Find current RHS (term 1 + term 2)
                A_times_b_1D(B_times_T, Blocal, p); // Term 1
                A_times_b_linear_1D(A_times_dTdAlpha, Alocal, q); // Term 2
        
                // Sum both terms
                u[0] = B_times_T[0] + A_times_dTdAlpha[0];
        
                // Step 2: Compute mid "point"
                qMid[0] = q[0] + h * u[0]/2.0;
        
                // Step 3: Compute uMid
                A_times_b_1D(B_times_T, Blocal, pMid); // Term 1
                A_times_b_linear_1D(A_times_dTdAlpha, Alocal, qMid); // Term 2
        
                // Sum both terms
                uMid[0] = B_times_T[0] + A_times_dTdAlpha[0];

                // Update q
                q[0] += uMid[0] * h;
        
                // Update gradient
                grad[dim_index * boxsize + index] = q[0];
                
                // Update p
                p[0] += vMid[0]*h;
            }
        }
        return;
}

__global__ void   cpab_cuda_kernel_backward_2D(dim3 nthreads, const int n_theta, const int d, const int nP, const int nC,
                                               float* grad, const float* points, const float* As, const float* Bs,
                                               const int* nStepSolver, const int* nc, const int broadcast) {
        
        // Allocate memory for computations
        float p[2], v[2], pMid[2], vMid[2], q[2], qMid[2];
        float B_times_T[2], A_times_dTdAlpha[2], u[2], uMid[2];
        float Alocal[6], Blocal[6];
        int cellidx;
        
        // Thread index
        int point_index = threadIdx.x + blockIdx.x * blockDim.x;
        int batch_index = threadIdx.y + blockIdx.y * blockDim.y;
        int dim_index   = threadIdx.z + blockIdx.z * blockDim.z;
        
        // Make sure we are within bounds
        if(point_index < nP && batch_index < n_theta && dim_index < d){
            int index = 2 * nP * batch_index + point_index;
            int boxsize = 2 * nP * n_theta;
        
            // Define start index for the matrices belonging to this batch
            // batch * num_elem * 4 triangles pr cell * cell in x * cell in y
            int start_idx = batch_index * 6 * 4 * nc[0] * nc[1]; 

            // Get point
            p[0] = points[broadcast*batch_index*nP*2+point_index];
            p[1] = points[broadcast*batch_index*nP*2+point_index + nP];
            
            // Step size for solver
            double h = (1.0 / nStepSolver[0]);
        
            // Iterate a number of times
            for(int t=0; t<nStepSolver[0]; t++) {
                // Get current cell
                cellidx = cuda_findcellidx_2D(p, nc[0], nc[1]);
                
                // Get index of A
                int As_idx = 6*cellidx;
                
                // Extract local A
                for(int i = 0; i < 6; i++){
                    Alocal[i] = (As + As_idx + start_idx)[i];
                }
                
                // Compute velocity at current location
                A_times_b_2D(v, Alocal, p);
                
                // Compute midpoint
                pMid[0] = p[0] + h*v[0]/2.0;
                pMid[1] = p[1] + h*v[1]/2.0;
                
                // Compute velocity at midpoint
                A_times_b_2D(vMid, Alocal, pMid);
                
                // Get index of B
                int Bs_idx = 6 * dim_index * nC + As_idx;
                
                // Get local B
                for(int i = 0; i < 6; i++){
                    Blocal[i] = (Bs + Bs_idx)[i];
                }
                
                // Copy q
                q[0] = grad[dim_index*boxsize + index];
                q[1] = grad[dim_index*boxsize + index + nP];
        
                // Step 1: Compute u using the old location
                // Find current RHS (term 1 + term 2)
                A_times_b_2D(B_times_T, Blocal, p); // Term 1
                A_times_b_linear_2D(A_times_dTdAlpha, Alocal, q); // Term 2
        
                // Sum both terms
                u[0] = B_times_T[0] + A_times_dTdAlpha[0];
                u[1] = B_times_T[1] + A_times_dTdAlpha[1];
        
                // Step 2: Compute mid "point"
                qMid[0] = q[0] + h * u[0]/2.0;
                qMid[1] = q[1] + h * u[1]/2.0;
        
                // Step 3: Compute uMid
                A_times_b_2D(B_times_T, Blocal, pMid); // Term 1
                A_times_b_linear_2D(A_times_dTdAlpha, Alocal, qMid); // Term 2
        
                // Sum both terms
                uMid[0] = B_times_T[0] + A_times_dTdAlpha[0];
                uMid[1] = B_times_T[1] + A_times_dTdAlpha[1];

                // Update q
                q[0] += uMid[0] * h;
                q[1] += uMid[1] * h;
        
                // Update gradient
                grad[dim_index * boxsize + index] = q[0];
                grad[dim_index * boxsize + index + nP] = q[1];
                
                // Update p
                p[0] += vMid[0]*h;
                p[1] += vMid[1]*h;
            }
        }
        return;
}

__global__ void   cpab_cuda_kernel_backward_3D(dim3 nthreads, const int n_theta, const int d, const int nP, const int nC,
                                               float* grad, const float* points, const float* As, const float* Bs,
                                               const int* nStepSolver, const int* nc, const int broadcast) {
        
        // Allocate memory for computations
        float p[3], v[3], pMid[3], vMid[3], q[3], qMid[3];
        float B_times_T[3], A_times_dTdAlpha[3], u[3], uMid[3];
        float Alocal[12], Blocal[12];
        int cellidx;
        
        // Thread index
        int point_index = threadIdx.x + blockIdx.x * blockDim.x;
        int batch_index = threadIdx.y + blockIdx.y * blockDim.y;
        int dim_index   = threadIdx.z + blockIdx.z * blockDim.z;
        
        // Make sure we are within bounds
        if(point_index < nP && batch_index < n_theta && dim_index < d){
            int index = 3 * nP * batch_index + point_index;
            int boxsize = 3 * nP * n_theta;
        
            // Define start index for the matrices belonging to this batch
            // batch * 12 params pr cell * 6 triangles pr cell * cell in x * cell in y * cell in z
            int start_idx = batch_index * 12 * 5 * nc[0] * nc[1] * nc[2]; 
            
            // Get point
            p[0] = points[broadcast*batch_index*nP*3+point_index];
            p[1] = points[broadcast*batch_index*nP*3+point_index + nP];
            p[2] = points[broadcast*batch_index*nP*3+point_index + 2 * nP];
            
            // Step size for solver
            double h = (1.0 / nStepSolver[0]);
        
            // Iterate a number of times
            for(int t=0; t<nStepSolver[0]; t++) {
                // Get current cell
                cellidx = cuda_findcellidx_3D(p, nc[0], nc[1], nc[2]);
                
                // Get index of A
                int As_idx = 12*cellidx;
                
                // Extract local A
                for(int i = 0; i < 12; i++){
                    Alocal[i] = (As + As_idx + start_idx)[i];
                }
                
                // Compute velocity at current location
                A_times_b_3D(v, Alocal, p);
                
                // Compute midpoint
                pMid[0] = p[0] + h*v[0]/2.0;
                pMid[1] = p[1] + h*v[1]/2.0;
                pMid[2] = p[2] + h*v[2]/2.0;
                
                // Compute velocity at midpoint
                A_times_b_3D(vMid, Alocal, pMid);
                
                // Get index of B
                int Bs_idx = 12 * dim_index * nC + As_idx;
                
                // Get local B
                for(int i = 0; i < 12; i++){
                    Blocal[i] = (Bs + Bs_idx)[i];
                }
                
                // Copy q
                q[0] = grad[dim_index * boxsize + index];
                q[1] = grad[dim_index * boxsize + index + nP];
                q[2] = grad[dim_index * boxsize + index + 2*nP];
        
                // Step 1: Compute u using the old location
                // Find current RHS (term 1 + term 2)
                A_times_b_3D(B_times_T, Blocal, p); // Term 1
                A_times_b_linear_3D(A_times_dTdAlpha, Alocal, q); // Term 2
        
                // Sum both terms
                u[0] = B_times_T[0] + A_times_dTdAlpha[0];
                u[1] = B_times_T[1] + A_times_dTdAlpha[1];
                u[2] = B_times_T[2] + A_times_dTdAlpha[2];
        
                // Step 2: Compute mid "point"
                qMid[0] = q[0] + h * u[0]/2.0;
                qMid[1] = q[1] + h * u[1]/2.0;
                qMid[2] = q[2] + h * u[2]/2.0;
        
                // Step 3: Compute uMid
                A_times_b_3D(B_times_T, Blocal, pMid); // Term 1
                A_times_b_linear_3D(A_times_dTdAlpha, Alocal, qMid); // Term 2
        
                // Sum both terms
                uMid[0] = B_times_T[0] + A_times_dTdAlpha[0];
                uMid[1] = B_times_T[1] + A_times_dTdAlpha[1];
                uMid[2] = B_times_T[2] + A_times_dTdAlpha[2];

                // Update q
                q[0] += uMid[0] * h;
                q[1] += uMid[1] * h;
                q[2] += uMid[2] * h;
        
                // Update gradient
                grad[dim_index * boxsize + index] = q[0];
                grad[dim_index * boxsize + index + nP] = q[1];
                grad[dim_index * boxsize + index + 2 * nP] = q[2];
                
                // Update p
                p[0] += vMid[0]*h;
                p[1] += vMid[1]*h;
                p[2] += vMid[2]*h;
            }
        }
        return;
}
