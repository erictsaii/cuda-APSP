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

__global__ void phase_1(int *d_D, int round) {
    __shared__ int share_D[B * B];
    // for share
    const int s_x = threadIdx.x;
    const int s_y = threadIdx.y;
    // for global
    const int g_x = round * B + threadIdx.x; 
    const int g_y = round * B + threadIdx.y; 
    // block size is 64*64 but we only have 32*32 threads 
    share_D[s_y * B + s_x] = d_D[g_y * V + g_x];
    share_D[(s_y + half_B) * B + s_x] = d_D[(g_y + half_B) * V + g_x];
    share_D[s_y * B + (s_x + half_B)] = d_D[g_y * V + (g_x + half_B)];
    share_D[(s_y + half_B) * B + (s_x + half_B)] = d_D[(g_y + half_B) * V + (g_x + half_B)];

    __syncthreads();

    #pragma unroll 32
	for (int k = 0; k < B; ++k) {
		share_D[s_y * B + s_x] = min(share_D[s_y * B + s_x], share_D[s_y * B + k] + share_D[k * B + s_x]);
		share_D[(s_y + half_B) * B + s_x] = min(share_D[(s_y + half_B) * B + s_x], share_D[(s_y + half_B) * B + k] + share_D[k * B + s_x]);
		share_D[s_y * B + (s_x + half_B)] = min(share_D[s_y * B + (s_x + half_B)], share_D[s_y * B + k] + share_D[k * B + (s_x + half_B)]);
		share_D[(s_y + half_B) * B + (s_x + half_B)] = min(share_D[(s_y + half_B) * B + (s_x + half_B)], share_D[(s_y + half_B) * B + k] + share_D[k * B + (s_x + half_B)]);
		__syncthreads();
	}

    // load back to global
	d_D[g_y * V + g_x] = share_D[s_y * B + s_x];
    d_D[(g_y + half_B) * V + g_x] = share_D[(s_y + half_B) * B + s_x];
    d_D[g_y * V + (g_x + half_B)] = share_D[s_y * B + (s_x + half_B)]; 
    d_D[(g_y + half_B) * V + (g_x + half_B)] = share_D[(s_y + half_B) * B + (s_x + half_B)];
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
    dim3 block(rounds, rounds);
    dim3 num_threads(B, 1024 / B); //?
    dim3 phase_1_num_threads(32, 32);
    
    for (int i = 0; i < rounds; ++i) {
        phase_1<<<1, phase_1_num_threads>>>(d_D, i);
    }

    // output
    output(argv[2]);
}