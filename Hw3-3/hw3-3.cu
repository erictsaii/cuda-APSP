#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <omp.h>

#define DEV_NO 0
#define B 64
#define half_B 32
 
cudaDeviceProp prop;
int cpu_cnt;

int* D;
int V;
int V_before_padding;
int E;

void input(const char* filename) {
    // open file
    FILE* file = fopen(filename, "rb");
	fread(&V, sizeof(int), 1, file);
	fread(&E, sizeof(int), 1, file);

    V_before_padding = V;
    int r = V % B;
    if (r != 0) V = V + B - r;
    // initialize matrix
    D = (int *)malloc(V * V * sizeof(int));
    for (int i = 0; i < V; ++i){
        int i_V = i * V;
        for (int j = 0; j < V; ++j){
            if (i == j) D[i_V + j] = 0;
            else D[i_V + j] = 1073741823;
        }
    }
    int tmp[300];
    if (E >= 100){
        int j = 0;
        for (; j < E; j += 100) {
            fread(tmp, sizeof(int), 300, file);
            for (int i = 0; i < 300; i += 3){
                D[tmp[i] * V + tmp[i+1]] = tmp[i+2];
            }
	    }
        for (int i = j - 100; i < E; ++i) {
            fread(tmp, sizeof(int), 3, file);
            D[tmp[0] * V + tmp[1]] = tmp[2];
	    }
    }
    else{
        for (int i = 0; i < E; ++i) {
            fread(tmp, sizeof(int), 3, file);
            D[tmp[0] * V + tmp[1]] = tmp[2];
	    }
    }
	fclose(file);
}


void output(const char* filename) {
    FILE* file = fopen(filename, "w");
    for (int i = 0; i < V_before_padding; ++i){
        fwrite(&D[i * V], sizeof(int), V_before_padding, file);
    }
	fclose(file);
}

__global__ void phase_1(int *d_D, int round, int V) {
    __shared__ int share_D[B * B];
    // for share
    const int s_x = threadIdx.x;
    const int s_y = threadIdx.y;
    // for global
    const int g_x = round * B + threadIdx.x; 
    const int g_y = round * B + threadIdx.y; 
    // block size is 64*64 but we only have 32*32 threads 
    // 1 thread deals with 4 kinds of D[i, j]
    share_D[s_y * B + s_x] = d_D[g_y * V + g_x];
    share_D[s_y * B + (s_x + half_B)] = d_D[g_y * V + (g_x + half_B)];
    share_D[(s_y + half_B) * B + s_x] = d_D[(g_y + half_B) * V + g_x];
    share_D[(s_y + half_B) * B + (s_x + half_B)] = d_D[(g_y + half_B) * V + (g_x + half_B)];

    __syncthreads();

    #pragma unroll 32
	for (int k = 0; k < B; ++k) {
		share_D[s_y * B + s_x] = min(share_D[s_y * B + s_x], share_D[s_y * B + k] + share_D[k * B + s_x]);
        share_D[s_y * B + (s_x + half_B)] = min(share_D[s_y * B + (s_x + half_B)], share_D[s_y * B + k] + share_D[k * B + (s_x + half_B)]);
		share_D[(s_y + half_B) * B + s_x] = min(share_D[(s_y + half_B) * B + s_x], share_D[(s_y + half_B) * B + k] + share_D[k * B + s_x]);
		share_D[(s_y + half_B) * B + (s_x + half_B)] = min(share_D[(s_y + half_B) * B + (s_x + half_B)], share_D[(s_y + half_B) * B + k] + share_D[k * B + (s_x + half_B)]);
		__syncthreads();
	}

    // load back to global
	d_D[g_y * V + g_x] = share_D[s_y * B + s_x];
    d_D[g_y * V + (g_x + half_B)] = share_D[s_y * B + (s_x + half_B)]; 
    d_D[(g_y + half_B) * V + g_x] = share_D[(s_y + half_B) * B + s_x];
    d_D[(g_y + half_B) * V + (g_x + half_B)] = share_D[(s_y + half_B) * B + (s_x + half_B)];
}

__global__ void phase_2(int *d_D, int round, int V) {
    if (blockIdx.x == round) return;
    // init share memory
    __shared__ int pivot_D[B * B];
    __shared__ int row_D[B * B];
    __shared__ int col_D[B * B];
    // load pivot_D
    const int s_x = threadIdx.x;
    const int s_y = threadIdx.y;
    int g_x = round * B + threadIdx.x; 
    int g_y = round * B + threadIdx.y;

    pivot_D[s_y * B + s_x] = d_D[g_y * V + g_x];
    pivot_D[s_y * B + (s_x + half_B)] = d_D[g_y * V + (g_x + half_B)];
    pivot_D[(s_y + half_B) * B + s_x] = d_D[(g_y + half_B) * V + g_x];
    pivot_D[(s_y + half_B) * B + (s_x + half_B)] = d_D[(g_y + half_B) * V + (g_x + half_B)];
    
    // load row_D
    g_x = blockIdx.x * B + threadIdx.x; 
    g_y = round * B + threadIdx.y;

    row_D[s_y * B + s_x] = d_D[g_y * V + g_x];
    row_D[s_y * B + (s_x + half_B)] = d_D[g_y * V + (g_x + half_B)];
    row_D[(s_y + half_B) * B + s_x] = d_D[(g_y + half_B) * V + g_x];
    row_D[(s_y + half_B) * B + (s_x + half_B)] = d_D[(g_y + half_B) * V + (g_x + half_B)];

    // load col_D
    g_x = round * B + threadIdx.x; 
    g_y = blockIdx.x * B + threadIdx.y;

    col_D[s_y * B + s_x] = d_D[g_y * V + g_x];
    col_D[s_y * B + (s_x + half_B)] = d_D[g_y * V + (g_x + half_B)];
    col_D[(s_y + half_B) * B + s_x] = d_D[(g_y + half_B) * V + g_x];
    col_D[(s_y + half_B) * B + (s_x + half_B)] = d_D[(g_y + half_B) * V + (g_x + half_B)];

    __syncthreads();

    // calculate
    #pragma unroll 32
    for (int k = 0; k < B; ++k){
        // row
        row_D[s_y * B + s_x] = min(row_D[s_y * B + s_x], pivot_D[s_y * B + k] + row_D[k * B + s_x]);
        row_D[s_y * B + (s_x + half_B)] = min(row_D[s_y * B + (s_x + half_B)], pivot_D[s_y * B + k] + row_D[k * B + (s_x + half_B)]);
		row_D[(s_y + half_B) * B + s_x] = min(row_D[(s_y + half_B) * B + s_x], pivot_D[(s_y + half_B) * B + k] + row_D[k * B + s_x]);
		row_D[(s_y + half_B) * B + (s_x + half_B)] = min(row_D[(s_y + half_B) * B + (s_x + half_B)], pivot_D[(s_y + half_B) * B + k] + row_D[k * B + (s_x + half_B)]);
        // col        
        col_D[s_y * B + s_x] = min(col_D[s_y * B + s_x], col_D[s_y * B + k] + pivot_D[k * B + s_x]);
        col_D[s_y * B + (s_x + half_B)] = min(col_D[s_y * B + (s_x + half_B)], col_D[s_y * B + k] + pivot_D[k * B + (s_x + half_B)]);
		col_D[(s_y + half_B) * B + s_x] = min(col_D[(s_y + half_B) * B + s_x], col_D[(s_y + half_B) * B + k] + pivot_D[k * B + s_x]);
		col_D[(s_y + half_B) * B + (s_x + half_B)] = min(col_D[(s_y + half_B) * B + (s_x + half_B)], col_D[(s_y + half_B) * B + k] + pivot_D[k * B + (s_x + half_B)]);
    }
    // load col back to global
    d_D[g_y * V + g_x] = col_D[s_y * B + s_x];
    d_D[g_y * V + (g_x + half_B)] = col_D[s_y * B + (s_x + half_B)];
    d_D[(g_y + half_B) * V + g_x] = col_D[(s_y + half_B) * B + s_x];
    d_D[(g_y + half_B) * V + (g_x + half_B)] = col_D[(s_y + half_B) * B + (s_x + half_B)];

    // load row back to global
    g_x = blockIdx.x * B + threadIdx.x; 
    g_y = round * B + threadIdx.y;

    d_D[g_y * V + g_x] = row_D[s_y * B + s_x];
    d_D[g_y * V + (g_x + half_B)] = row_D[s_y * B + (s_x + half_B)];
    d_D[(g_y + half_B) * V + g_x] = row_D[(s_y + half_B) * B + s_x];
    d_D[(g_y + half_B) * V + (g_x + half_B)] = row_D[(s_y + half_B) * B + (s_x + half_B)];
}

__global__ void phase_3(int *d_D, int round, int V, int y_offset) {
    if (blockIdx.x == round || blockIdx.y + y_offset == round) return;
    // init share memory
    __shared__ int row_D[B * B];
    __shared__ int col_D[B * B];
    // load row_D
    const int s_x = threadIdx.x;
    const int s_y = threadIdx.y;
    const int s_y_mul_B = s_y * B;
    const int s_y_add_half_B_mul_B = (s_y + half_B) * B;

    int g_x = blockIdx.x * B + threadIdx.x; 
    int g_y = round * B + threadIdx.y;

    row_D[s_y_mul_B + s_x] = d_D[g_y * V + g_x];
    row_D[s_y_mul_B + (s_x + half_B)] = d_D[g_y * V + (g_x + half_B)];
    row_D[s_y_add_half_B_mul_B + s_x] = d_D[(g_y + half_B) * V + g_x];
    row_D[s_y_add_half_B_mul_B + (s_x + half_B)] = d_D[(g_y + half_B) * V + (g_x + half_B)];

    // load col_D
    g_x = round * B + threadIdx.x; 
    g_y = (blockIdx.y + y_offset) * B + threadIdx.y;

    col_D[s_y_mul_B + s_x] = d_D[g_y * V + g_x];
    col_D[s_y_mul_B + (s_x + half_B)] = d_D[g_y * V + (g_x + half_B)];
    col_D[s_y_add_half_B_mul_B + s_x] = d_D[(g_y + half_B) * V + g_x];
    col_D[s_y_add_half_B_mul_B + (s_x + half_B)] = d_D[(g_y + half_B) * V + (g_x + half_B)];

    // load base
    g_x = blockIdx.x * B + threadIdx.x; 
    g_y = (blockIdx.y + y_offset) * B + threadIdx.y;

    int base_0 = d_D[g_y * V + g_x];
    int base_1 = d_D[g_y * V + (g_x + half_B)];
    int base_2 = d_D[(g_y + half_B) * V + g_x];
    int base_3 = d_D[(g_y + half_B) * V + (g_x + half_B)];

     __syncthreads();

    // calculate
    #pragma unroll 32
	for (int k = 0; k < B; ++k) {
		base_0 = min(base_0, col_D[s_y_mul_B + k] + row_D[k * B + s_x]);
		base_1 = min(base_1, col_D[s_y_mul_B + k] + row_D[k * B + (s_x + half_B)]);
        base_2 = min(base_2, col_D[s_y_add_half_B_mul_B + k] + row_D[k * B + s_x]);
		base_3 = min(base_3, col_D[s_y_add_half_B_mul_B + k] + row_D[k * B + (s_x + half_B)]);
	}
    
    d_D[g_y * V + g_x] = base_0;
    d_D[g_y * V + (g_x + half_B)] = base_1;
    d_D[(g_y + half_B) * V + g_x] = base_2;
    d_D[(g_y + half_B) * V + (g_x + half_B)] = base_3;
}

int main(int argc, char** argv) {
    // get input
    input(argv[1]);

    // cudaGetDeviceProperties(&prop, DEV_NO);
    // printf("maxThreadsPerBlock = %d, sharedMemPerBlock = %d", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);

    int* d_D[2];
    int rounds = V / B;
    size_t total_size = V * V * sizeof(int);
    // pin host D
    cudaHostRegister(D, total_size, cudaHostRegisterDefault);
    dim3 num_threads(32, 32);

    #pragma omp parallel num_threads(2)
    {
        int tid = omp_get_thread_num();
        cudaSetDevice(tid);
		cudaMalloc(&d_D[tid], total_size);
        dim3 phase_3_blocks(rounds, rounds / 2);
        if (tid == 1 && (rounds & 1)) ++phase_3_blocks.y;

        int y_offset = (tid == 0) ? 0 : rounds / 2;

        cudaMemcpy(d_D[tid] + y_offset * B * V, D + y_offset * B * V, phase_3_blocks.y * B * V * sizeof(int), cudaMemcpyHostToDevice);

        for (int i = 0; i < rounds; ++i) {
            if (i >= y_offset && i < y_offset + phase_3_blocks.y) {
				cudaMemcpy(D + i * B * V, d_D[tid] + i * B * V, B * V * sizeof(int), cudaMemcpyDeviceToHost);
			}
            #pragma omp barrier
            if (i < y_offset || i >= y_offset + phase_3_blocks.y) {
				cudaMemcpy(d_D[tid] + i * B * V, D + i * B * V, B * V * sizeof(int), cudaMemcpyHostToDevice);
			}

            phase_1<<<1, num_threads>>>(d_D[tid], i, V);
            phase_2<<<rounds, num_threads>>>(d_D[tid], i, V);
            phase_3<<<phase_3_blocks, num_threads>>>(d_D[tid], i, V, y_offset);
        }
        cudaMemcpy(D + y_offset * B * V, d_D[tid] + y_offset * B * V, phase_3_blocks.y * B * V * sizeof(int), cudaMemcpyDeviceToHost);
    }

    // output
    output(argv[2]);
}