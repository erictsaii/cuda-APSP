#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define DEV_NO 0
#define B 64
#define half_B 32
 
cudaDeviceProp prop;
int cpu_cnt;

int* D;
int V;
int E;

void input(const char* filename) {
    // open file
    FILE* file = fopen(filename, "rb");
	fread(&V, sizeof(int), 1, file);
	fread(&E, sizeof(int), 1, file);

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
    fwrite(D, sizeof(int), V * V, file);
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

    #pragma unroll 32 //necessary?
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
        col_D[s_y * B + s_x] = min(col_D[s_y * B + s_x], pivot_D[s_y * B + k] + col_D[k * B + s_x]);
        col_D[s_y * B + (s_x + half_B)] = min(col_D[s_y * B + (s_x + half_B)], pivot_D[s_y * B + k] + col_D[k * B + (s_x + half_B)]);
		col_D[(s_y + half_B) * B + s_x] = min(col_D[(s_y + half_B) * B + s_x], pivot_D[(s_y + half_B) * B + k] + col_D[k * B + s_x]);
		col_D[(s_y + half_B) * B + (s_x + half_B)] = min(col_D[(s_y + half_B) * B + (s_x + half_B)], pivot_D[(s_y + half_B) * B + k] + col_D[k * B + (s_x + half_B)]);
        // no __syncthreads()?
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

int main(int argc, char** argv) {
    // get input
    input(argv[1]);

    // cudaGetDeviceProperties(&prop, DEV_NO);
    // printf("maxThreasPerBlock = %d, sharedMemPerBlock = %d", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);

    int* d_D;
    // pin host D, maybe cudaHostRegisterReadOnly?
    cudaHostRegister(D, V * V * sizeof(int), cudaHostRegisterDefault);
    cudaMalloc(&d_D, V * V * sizeof(int));
    cudaMemcpy(d_D, D, V * V * sizeof(int), cudaMemcpyHostToDevice);

    // block
    // B 32 or 64?
    int rounds = V / B;
    dim3 num_threads(32, 32); // maybe 64*16 is faster 
    dim3 phase_2_blocks(rounds, 1);
    
    for (int i = 0; i < rounds; ++i) {
        phase_1<<<1, num_threads>>>(d_D, i, V);
        phase_2<<<phase_2_blocks, num_threads>>>(d_D, i, V);
    }

    // output
    output(argv[2]);
}