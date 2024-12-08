#include <math.h>
#include <algorithm>
#include <iostream>

int stride(const int ndim, const int* nc){
    int s;
		if(ndim == 1){ s = 2; } // two parameters per cell
		if(ndim == 2){ s = 6 * 4; } // 6 parameters per triangle, 4 triangles per cell
		if(ndim == 3){ s = 12 * 5; } // 12 parameters per pyramid, 5 pyramids per cell
    for(int j = 0; j < ndim; j++) {
        s *= nc[j];
    }
    return s;
}

int param_pr_cell(const int ndim){
	if(ndim == 1){ return 2; } // two parameters per cell
	if(ndim == 2){ return 6; } // 6 parameters per triangle
	if(ndim == 3){ return 12; } // 12 parameters per pyramid
}

int mymin(const int a, const double b) {
    return !(b<a)?a:round(b);
}
    
int findcellidx_1D(const float* p, const int nx) {
    // Floor value to find cell
    int idx = std::floor(p[0] * nx);
    idx = std::max(0, std::min(idx, nx-1));
    return idx;
}

int findcellidx_2D(const float* p, const int nx, const int ny) {
    // Copy point
    double point[2];
    point[0] = p[0];
    point[1] = p[1];
    
    // Cell size
    const float inc_x = 1.0 / nx;
    const float inc_y = 1.0 / ny;
    
    // Find initial row, col placement
    double p0 = std::min((nx * inc_x - 0.000000001), std::max(0.0, point[0]));
    double p1 = std::min((ny * inc_y - 0.000000001), std::max(0.0, point[1]));

    double xmod = std::fmod((double)p0, (double)inc_x);
    double ymod = std::fmod((double)p1, (double)inc_y);

    double x = xmod / inc_x;
    double y = ymod / inc_y;
            
    int cell_idx =  mymin(nx-1, (p0 - xmod) / inc_x) + 
                    mymin(ny-1, (p1 - ymod) / inc_y) * nx;        
    cell_idx *= 4;
            
    // Out of bound (left)
    if(point[0]<=0){
        if(point[1] <= 0 && point[1]/inc_y<point[0]/inc_x){
            // Nothing to do here
        } else if(point[1] >= ny * inc_y && point[1]/inc_y-ny > -point[0]/inc_x) {
            cell_idx += 2;
        } else {
            cell_idx += 3;
        }
        return cell_idx;
    }
            
    // Out of bound (right)
    if(point[0] >= nx*inc_x){
        if(point[1]<=0 && -point[1]/inc_y > point[0]/inc_x - nx){
            // Nothing to do here
        } else if(point[1] >= ny*inc_y && point[1]/inc_y - ny > point[0]/inc_x-nx){
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
    if(point[1] >= ny*inc_y){
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
}

int findcellidx_3D(const float* p, const int nx, const int ny, const int nz) {
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
        const float abs_x = std::abs(point[0]);
        const float abs_y = std::abs(point[1]);
        const float abs_z = std::abs(point[2]);
        
        const float push_x = (abs_x < abs_y && abs_x < abs_z) ? half*inc_x : 0.0;
        const float push_y = (abs_y < abs_x && abs_x < abs_z) ? half*inc_y : 0.0;
        const float push_z = (abs_z < abs_x && abs_x < abs_y) ? half*inc_z : 0.0;
        if(abs_x > half){point[0] = std::copysign(half - push_x, point[0]);}
        if(abs_y > half){point[1] = std::copysign(half - push_y, point[1]);}
        if(abs_z > half){point[2] = std::copysign(half - push_z, point[2]);}
        point[0] += half;
        point[1] += half;
        point[2] += half;
    }
    float zero = 0.0;
    float p0 = std::min((float)(nx*inc_x-1e-8),std::max(zero, point[0]));
    float p1 = std::min((float)(ny*inc_y-1e-8),std::max(zero, point[1]));       
    float p2 = std::min((float)(nz*inc_x-1e-8),std::max(zero, point[2])); 
    
    double xmod = fmod(p0,inc_x);
    double ymod = fmod(p1,inc_y);
    double zmod = fmod(p2,inc_z);
    
    int i = mymin(nx-1,((p0 - xmod)/inc_x));
    int j = mymin(ny-1,((p1 - ymod)/inc_y));
    int k = mymin(nz-1,((p2 - zmod)/inc_z));
    
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

int findcellidx(int ndim, const float* p, const int* nc){
	if(ndim == 1){ return findcellidx_1D(p, nc[0]);} 
	if(ndim == 2){ return findcellidx_2D(p, nc[0], nc[1]);}
	if(ndim == 3){ return findcellidx_3D(p, nc[0], nc[1], nc[2]); }
}
    
void A_times_b(int ndim, float x[], const float* A, const float* b){
	if(ndim == 1){
		x[0] = A[0]*b[0] + A[1];
	}
	if(ndim == 2){
		x[0] = A[0]*b[0] + A[1]*b[1] + A[2];
    	x[1] = A[3]*b[0] + A[4]*b[1] + A[5];		
	}
	if(ndim == 3){
		x[0] = A[0]*b[0] + A[1]*b[1] + A[2]*b[2] + A[3];
        x[1] = A[4]*b[0] + A[5]*b[1] + A[6]*b[2] + A[7];
        x[2] = A[8]*b[0] + A[9]*b[1] + A[10]*b[2] + A[11];
	}
    return;
}

void A_times_b_linear(int ndim, float x[], const float* A, float* b){
    if(ndim == 1){
		x[0] = A[0]*b[0];
	}
	if(ndim == 2){
		x[0] = A[0]*b[0] + A[1]*b[1];
    	x[1] = A[3]*b[0] + A[4]*b[1];		
	}
	if(ndim == 3){
		x[0] = A[0]*b[0] + A[1]*b[1] + A[2]*b[2];
        x[1] = A[4]*b[0] + A[5]*b[1] + A[6]*b[2];
        x[2] = A[8]*b[0] + A[9]*b[1] + A[10]*b[2];
    }
    return;
}

void cpab_forward_op(  float* newpoints, const float* points, const float* trels,
                       const int* nstepsolver, const int* nc,
                       const int ndim, const int nP, const int batch_size,
                       const int broadcast){
    // Main loop
    for(int t = 0; t < batch_size; t++) { // for all batches
        // Start index for batch
        int start_idx = t * stride(ndim, nc);
        
        for(int i = 0; i < nP; i++) { // for all points
            // Current point
            float point[ndim];
            for(int j = 0; j < ndim; j++){
                point[j] = points[broadcast*t*nP*ndim + i + j*nP];
            }
            // Iterate in nStepSolver
            for(int n = 0; n < nstepsolver[0]; n++){
                // Find cell index
                int idx = findcellidx(ndim, point, nc);
                
                // Get mapping
                const float* tidx = trels + param_pr_cell(ndim)*idx + start_idx;  
								
                // Update points
                float newpoint[ndim];
                A_times_b(ndim, newpoint, tidx, point);
                for(int j = 0; j < ndim; j++){
                    point[j] = newpoint[j];
                }
            }
            // Update output
            for(int j = 0; j < ndim; j++){
                newpoints[t * ndim * nP + i + j * nP] = point[j]; 
            }
        }
    }
}

void cpab_backward_op( float* grad, const float* points, const float* As,
                       const float* Bs, const int* nstepsolver, const int* nc,
                       const int n_theta, const int d,
                       const int ndim, const int nP, const int nC,
                       const int broadcast) {
    // Make data structures for calculations
    float p[ndim], v[ndim], pMid[ndim], vMid[ndim], q[ndim], qMid[ndim];
    float B_times_T[ndim], A_times_dTdAlpha[ndim], u[ndim], uMid[ndim];
    float Alocal[ndim*(ndim+1)], Blocal[ndim*(ndim+1)];

    // Loop over all transformations
    for(int batch_index = 0; batch_index < n_theta; batch_index++){
        // For all points
        for(int point_index = 0; point_index < nP; point_index++){
            // For all parameters in the transformations
            for(int dim_index = 0; dim_index < d; dim_index++){
                int index = ndim * nP * batch_index + point_index;
                int boxsize = ndim * nP * n_theta;
                
                // Define start index for the matrices belonging to this batch
                int start_idx = batch_index * stride(ndim, nc);
                
                // Get point
                for(int j = 0; j < ndim; j++){
                    p[j] = points[broadcast*batch_index*ndim*nP+point_index + j*nP];
                }
                
                double h = (1.0 / nstepsolver[0]);
                
                // Integrate velocity field
                for(int t = 0; t < nstepsolver[0]; t++){
                    // Get current cell
                    int cellidx = findcellidx(ndim, p, nc);
                    
                    // Get index of A
                    int params_size = param_pr_cell(ndim);
                    int As_idx = params_size*cellidx;
                    
                    // Extract local A
                    for(int i = 0; i < params_size; i++){
                        Alocal[i] = (As + As_idx + start_idx)[i];
                    }
                    
                    // Compute velocity at current location
                    A_times_b(ndim, v, Alocal, p);
                    
                    // Compute midpoint
                    for(int j = 0; j < ndim; j++){
                        pMid[j] = p[j] + h*v[j]/2.0;
                    }
                    
                    // Compute velocity at midpoint
                    A_times_b(ndim, vMid, Alocal, pMid);
                    
                    // Get index of B
                    int Bs_idx = params_size * dim_index * nC + As_idx;
                    
                    // Get local B
                    for(int i = 0; i < params_size; i++){
                        Blocal[i] = (Bs + Bs_idx)[i];
                    }
                    
                    // Copy q
                    for(int j = 0; j < ndim; j++){
                        q[j] = grad[dim_index*boxsize + index + j*nP];
                    }
                    
                    // Step 1: Compute u using the old location
                    // Find current RHS (term 1 + trem 2)
                    A_times_b(ndim, B_times_T, Blocal, p); // term 1
                    A_times_b_linear(ndim, A_times_dTdAlpha, Alocal, q); // term 2
                    
                    // Sum both terms
                    for(int j = 0; j < ndim; j++){
                        u[j] = B_times_T[j] + A_times_dTdAlpha[j];
                    }
                    
                    // Step 2: Compute mid point
                    for(int j = 0; j < ndim; j++){
                        qMid[j] = q[j] + h * u[j]/2.0;
                    }
                    
                    // Step 3: Compute uMid
                    A_times_b(ndim, B_times_T, Blocal, pMid);
                    A_times_b_linear(ndim, A_times_dTdAlpha, Alocal, qMid);
                    
                    // Sum both terms
                    for(int j = 0; j < ndim; j++){
                        uMid[j] = B_times_T[j] + A_times_dTdAlpha[j];
                    }
                    
                    // Update q
                    for(int j = 0; j < ndim; j++){
                        q[j] += uMid[j] * h;
                    }
                    
                    // Update gradient
                    for(int j = 0; j < ndim; j++){
                        grad[dim_index * boxsize + index + j*nP] = q[j];
                    }
                    
                    // Update p
                    for(int j = 0; j < ndim; j++){
                        p[j] += vMid[j]*h;
                    }
                    
                }
            }
        }
    }
}