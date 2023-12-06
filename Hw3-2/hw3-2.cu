#include <immintrin.h>
#include <omp.h>
#include <pthread.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#ifdef DEBUG
#define DEBUG_PRINT(fmt, args...) fprintf(stderr, fmt, ##args);
#define DEBUG_MSG(str) std::cout << str << "\n";
#define CUDA_EXE(F)                                                \
    {                                                              \
        cudaError_t err = F;                                       \
        if ((err != cudaSuccess)) {                                \
            printf("Error %s at %s:%d\n", cudaGetErrorString(err), \
                   __FILE__, __LINE__);                            \
            exit(-1);                                              \
        }                                                          \
    }
#define CUDA_CHECK()                                                                    \
    {                                                                                   \
        cudaError_t err = cudaGetLastError();                                           \
        if ((err != cudaSuccess)) {                                                     \
            printf("Error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(-1);                                                                   \
        }                                                                               \
    }
#else
#define DEBUG_PRINT(fmt, args...)
#define DEBUG_MSG(str)
#define CUDA_EXE(F) F;
#define CUDA_CHECK()
#endif  // DEBUG

#ifdef TIMING
#include <ctime>
#define TIMING_START(arg)          \
    struct timespec __start_##arg; \
    clock_gettime(CLOCK_MONOTONIC, &__start_##arg);
#define TIMING_END(arg)                                                                       \
    {                                                                                         \
        struct timespec __temp_##arg, __end_##arg;                                            \
        double __duration_##arg;                                                              \
        clock_gettime(CLOCK_MONOTONIC, &__end_##arg);                                         \
        if ((__end_##arg.tv_nsec - __start_##arg.tv_nsec) < 0) {                              \
            __temp_##arg.tv_sec = __end_##arg.tv_sec - __start_##arg.tv_sec - 1;              \
            __temp_##arg.tv_nsec = 1000000000 + __end_##arg.tv_nsec - __start_##arg.tv_nsec;  \
        } else {                                                                              \
            __temp_##arg.tv_sec = __end_##arg.tv_sec - __start_##arg.tv_sec;                  \
            __temp_##arg.tv_nsec = __end_##arg.tv_nsec - __start_##arg.tv_nsec;               \
        }                                                                                     \
        __duration_##arg = __temp_##arg.tv_sec + (double)__temp_##arg.tv_nsec / 1000000000.0; \
        printf("%s took %lfs.\n", #arg, __duration_##arg);                                    \
    }
#else
#define TIMING_START(arg)
#define TIMING_END(arg)
#endif  // TIMING

#define TILE 26
#define block_size 78
#define div_block 3
const int INF = ((1 << 30) - 1);

__device__ int blk_idx(int r, int c, int blk_pitch, int nblocks);

__global__ void proc_1_glob(int *blk_dist, int k, int blk_pitch, int nblocks);
__global__ void proc_2_glob(int *blk_dist, int s, int k, int blk_pitch, int nblocks);
__global__ void proc_3_glob(int *blk_dist, int s_i, int s_j, int k, int blk_pitch, int nblocks);

__global__ void init_dist(int *blk_dist, int blk_pitch, int nblocks);
__global__ void build_dist(int *edge, int E, int *blk_dist, int blk_pitch, int nblocks);
__global__ void copy_dist(int *blk_dist, int blk_pitch, int *dist, int pitch, int nblocks);

__global__ void proc_1_blk_glob(int *blk_dist, int k, int pitch);
__global__ void proc_2_blk_glob(int *blk_dist, int s, int k, int pitch);
__global__ void proc_3_blk_glob(int *blk_dist, int s_i, int s_j, int k, int pitch);

__global__ void init_blk_dist(int *blk_dist, int pitch);
__global__ void build_blk_dist(int *edge, int E, int *blk_dist, int pitch);

int main(int argc, char **argv) {
    auto compute_start = std::chrono::steady_clock::now();
    assert(argc == 3);

    char *input_filename = argv[1];
    char *output_filename = argv[2];
    FILE *input_file;
    FILE *output_file;
    int ncpus = omp_get_max_threads();
    int device_cnt;
    int V, E;
    int *edge;
    int *dist;
    int VP;
    int nblocks;
    cudaDeviceProp deviceProp;

    cudaGetDeviceCount(&device_cnt);
    cudaSetDevice(0);
    cudaGetDeviceProperties(&deviceProp, 0);

    TIMING_START(hw3_2);

    /* input */
    TIMING_START(input);
    input_file = fopen(input_filename, "rb");
    assert(input_file);
    fread(&V, sizeof(int), 1, input_file);
    fread(&E, sizeof(int), 1, input_file);
    edge = (int *)malloc(sizeof(int) * 3 * E);
    fread(edge, sizeof(int), 3 * E, input_file);
    dist = (int *)malloc(sizeof(int) * V * V);
    fclose(input_file);
    DEBUG_PRINT("vertices: %d\nedges: %d\n", V, E);
    TIMING_END(input);

    nblocks = (int)ceilf(float(V) / block_size);
    VP = nblocks * block_size;

    /* calculate */
    if ((size_t)VP * VP * 2 + 2 * 3 * E + V * V <= deviceProp.totalGlobalMem / 4) {
        DEBUG_PRINT("Flatten\n");
        int *edge_dev;
        int *dist_dev;
        int *blk_dist_dev;
        size_t blk_pitch;

        TIMING_START(calculate);

        cudaHostRegister(edge, sizeof(int) * 3 * E, cudaHostRegisterReadOnly);
        cudaMalloc(&edge_dev, sizeof(int) * 3 * E);
        cudaHostRegister(dist, sizeof(int) * V * V, cudaHostRegisterDefault);
        cudaMalloc(&blk_dist_dev, sizeof(int) * block_size * block_size * nblocks * nblocks);
        blk_pitch = block_size * block_size;

        cudaMemcpy(edge_dev, edge, sizeof(int) * 3 * E, cudaMemcpyDefault);

        init_dist<<<dim3(VP / TILE, VP / TILE), dim3(TILE, TILE)>>>(blk_dist_dev, blk_pitch, nblocks);
        build_dist<<<(int)ceilf((float)E / (TILE * TILE)), TILE * TILE>>>(edge_dev, E, blk_dist_dev, blk_pitch, nblocks);
        cudaFree(edge_dev);

        dim3 blk(TILE, TILE);
        for (int k = 0, nk = nblocks - 1; k < nblocks; k++, nk--) {
            /* Phase 1 */
            proc_1_glob<<<1, blk>>>(blk_dist_dev, k, blk_pitch, nblocks);
            /* Phase 2 */
            proc_2_glob<<<dim3(nblocks - 1, 2), blk>>>(blk_dist_dev, 0, k, blk_pitch, nblocks);
            /* Phase 3 */
            proc_3_glob<<<dim3(nblocks - 1, nblocks - 1), blk>>>(blk_dist_dev, 0, 0, k, blk_pitch, nblocks);
        }

        cudaMalloc(&dist_dev, sizeof(int) * VP * VP);
        copy_dist<<<dim3(VP / TILE, VP / TILE), dim3(TILE, TILE)>>>(blk_dist_dev, blk_pitch, dist_dev, VP, nblocks);
        cudaMemcpy2D(dist, sizeof(int) * V, dist_dev, sizeof(int) * VP, sizeof(int) * V, V, cudaMemcpyDefault);

        cudaDeviceSynchronize();

        cudaFree(blk_dist_dev);
        cudaFree(dist_dev);

        TIMING_END(calculate);
    } else {
        int *edge_dev;
        int *dist_dev;

        TIMING_START(calculate);

        cudaHostRegister(edge, sizeof(int) * 3 * E, cudaHostRegisterReadOnly);
        cudaMalloc(&edge_dev, sizeof(int) * 3 * E);
        cudaHostRegister(dist, sizeof(int) * V * V, cudaHostRegisterDefault);
        cudaMalloc(&dist_dev, sizeof(int) * VP * VP);

        cudaMemcpy(edge_dev, edge, sizeof(int) * 3 * E, cudaMemcpyDefault);

        init_blk_dist<<<dim3(VP / TILE, VP / TILE), dim3(TILE, TILE)>>>(dist_dev, VP);
        build_blk_dist<<<(int)ceilf((float)E / (TILE * TILE)), TILE * TILE>>>(edge_dev, E, dist_dev, VP);
        cudaFree(edge_dev);

        dim3 blk(TILE, TILE);
        for (int k = 0, nk = nblocks - 1; k < nblocks; k++, nk--) {
            /* Phase 1 */
            proc_1_blk_glob<<<1, blk>>>(dist_dev, k, VP);
            /* Phase 2 */
            proc_2_blk_glob<<<dim3(nblocks - 1, 2), blk>>>(dist_dev, 0, k, VP);
            /* Phase 3 */
            proc_3_blk_glob<<<dim3(nblocks - 1, nblocks - 1), blk>>>(dist_dev, 0, 0, k, VP);
        }

        cudaMemcpy2D(dist, sizeof(int) * V, dist_dev, sizeof(int) * VP, sizeof(int) * V, V, cudaMemcpyDefault);

        cudaDeviceSynchronize();
        cudaFree(dist_dev);

        TIMING_END(calculate);
    }
    /* output */
    TIMING_START(output);
    output_file = fopen(output_filename, "w");
    assert(output_file);
    fwrite(dist, 1, sizeof(int) * V * V, output_file);
    fclose(output_file);
    TIMING_END(output);
    TIMING_END(hw3_2);

    free(edge);
    free(dist);
    auto compute_end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(compute_end - compute_start);
    printf("time %lld ms\n", duration.count());
    return 0;
}

__device__ int blk_idx(int r, int c, int blk_pitch, int nblocks) {
    return ((r / block_size) * nblocks + (c / block_size)) * blk_pitch + (r % block_size) * block_size + (c % block_size);
}

#define _ref(i, j, r, c) blk_dist[(i * nblocks + j) * blk_pitch + (r)*block_size + c]
__global__ void proc_1_glob(int *blk_dist, int k, int blk_pitch, int nblocks) {
    __shared__ int k_k_sm[block_size][block_size];

    int r = threadIdx.y;
    int c = threadIdx.x;

#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            k_k_sm[r + rr * TILE][c + cc * TILE] = _ref(k, k, r + rr * TILE, c + cc * TILE);
        }
    }
    __syncthreads();

#pragma unroll
    for (int b = 0; b < block_size; b++) {
#pragma unroll
        for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
            for (int cc = 0; cc < div_block; cc++) {
                k_k_sm[r + rr * TILE][c + cc * TILE] = min(k_k_sm[r + rr * TILE][c + cc * TILE], k_k_sm[r + rr * TILE][b] + k_k_sm[b][c + cc * TILE]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            _ref(k, k, r + rr * TILE, c + cc * TILE) = k_k_sm[r + rr * TILE][c + cc * TILE];
        }
    }
}
__global__ void proc_2_glob(int *blk_dist, int s, int k, int blk_pitch, int nblocks) {
    __shared__ int k_k_sm[block_size][block_size];
    __shared__ int sm[block_size][block_size];

    int i = s + blockIdx.x;
    int r = threadIdx.y;
    int c = threadIdx.x;

    if (i >= k)
        i++;

#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            k_k_sm[r + rr * TILE][c + cc * TILE] = _ref(k, k, r + rr * TILE, c + cc * TILE);
        }
    }
    if (blockIdx.y == 0) {
        /* rows */
#pragma unroll
        for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
            for (int cc = 0; cc < div_block; cc++) {
                sm[r + rr * TILE][c + cc * TILE] = _ref(i, k, r + rr * TILE, c + cc * TILE);
            }
        }
        __syncthreads();

#pragma unroll
        for (int b = 0; b < block_size; b++) {
#pragma unroll
            for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
                for (int cc = 0; cc < div_block; cc++) {
                    sm[r + rr * TILE][c + cc * TILE] = min(sm[r + rr * TILE][c + cc * TILE], sm[r + rr * TILE][b] + k_k_sm[b][c + cc * TILE]);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
            for (int cc = 0; cc < div_block; cc++) {
                _ref(i, k, r + rr * TILE, c + cc * TILE) = sm[r + rr * TILE][c + cc * TILE];
            }
        }
    } else {
        /* cols */
#pragma unroll
        for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
            for (int cc = 0; cc < div_block; cc++) {
                sm[r + rr * TILE][c + cc * TILE] = _ref(k, i, r + rr * TILE, c + cc * TILE);
            }
        }
        __syncthreads();

#pragma unroll
        for (int b = 0; b < block_size; b++) {
#pragma unroll
            for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
                for (int cc = 0; cc < div_block; cc++) {
                    sm[r + rr * TILE][c + cc * TILE] = min(sm[r + rr * TILE][c + cc * TILE], k_k_sm[r + rr * TILE][b] + sm[b][c + cc * TILE]);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
            for (int cc = 0; cc < div_block; cc++) {
                _ref(k, i, r + rr * TILE, c + cc * TILE) = sm[r + rr * TILE][c + cc * TILE];
            }
        }
    }
}
__global__ void proc_3_glob(int *blk_dist, int s_i, int s_j, int k, int blk_pitch, int nblocks) {
    __shared__ int i_k_sm[block_size][block_size];
    __shared__ int k_j_sm[block_size][block_size];

    int i = s_i + blockIdx.y;
    int j = s_j + blockIdx.x;
    int r = threadIdx.y;
    int c = threadIdx.x;
    int loc[div_block][div_block];

    if (i >= k)
        i++;
    if (j >= k)
        j++;

#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            i_k_sm[r + rr * TILE][c + cc * TILE] = _ref(i, k, r + rr * TILE, c + cc * TILE);
        }
    }
#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            k_j_sm[r + rr * TILE][c + cc * TILE] = _ref(k, j, r + rr * TILE, c + cc * TILE);
        }
    }
    __syncthreads();
#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            loc[rr][cc] = _ref(i, j, r + rr * TILE, c + cc * TILE);
        }
    }

#pragma unroll
    for (int b = 0; b < block_size; b++) {
#pragma unroll
        for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
            for (int cc = 0; cc < div_block; cc++) {
                loc[rr][cc] = min(loc[rr][cc], i_k_sm[r + rr * TILE][b] + k_j_sm[b][c + cc * TILE]);
            }
        }
    }
#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            _ref(i, j, r + rr * TILE, c + cc * TILE) = loc[rr][cc];
        }
    }
}
__global__ void init_dist(int *blk_dist, int blk_pitch, int nblocks) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    blk_dist[blk_idx(r, c, blk_pitch, nblocks)] = (r != c) * INF;
}
__global__ void build_dist(int *edge, int E, int *blk_dist, int blk_pitch, int nblocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < E) {
        int src = *(edge + idx * 3);
        int dst = *(edge + idx * 3 + 1);
        int w = *(edge + idx * 3 + 2);
        blk_dist[blk_idx(src, dst, blk_pitch, nblocks)] = w;
    }
}
__global__ void copy_dist(int *blk_dist, int blk_pitch, int *dist, int pitch, int nblocks) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    dist[r * pitch + c] = blk_dist[blk_idx(r, c, blk_pitch, nblocks)];
}

#define _ref_blk(i, j, r, c) blk_dist[i * block_size * pitch + j * block_size + (r)*pitch + c]
__global__ void proc_1_blk_glob(int *blk_dist, int k, int pitch) {
    __shared__ int k_k_sm[block_size][block_size];

    int r = threadIdx.y;
    int c = threadIdx.x;

#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            k_k_sm[r + rr * TILE][c + cc * TILE] = _ref_blk(k, k, r + rr * TILE, c + cc * TILE);
        }
    }
    __syncthreads();

#pragma unroll
    for (int b = 0; b < block_size; b++) {
#pragma unroll
        for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
            for (int cc = 0; cc < div_block; cc++) {
                k_k_sm[r + rr * TILE][c + cc * TILE] = min(k_k_sm[r + rr * TILE][c + cc * TILE], k_k_sm[r + rr * TILE][b] + k_k_sm[b][c + cc * TILE]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            _ref_blk(k, k, r + rr * TILE, c + cc * TILE) = k_k_sm[r + rr * TILE][c + cc * TILE];
        }
    }
}
__global__ void proc_2_blk_glob(int *blk_dist, int s, int k, int pitch) {
    __shared__ int k_k_sm[block_size][block_size];
    __shared__ int sm[block_size][block_size];

    int i = s + blockIdx.x;
    int r = threadIdx.y;
    int c = threadIdx.x;

    if (i >= k)
        i++;

#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            k_k_sm[r + rr * TILE][c + cc * TILE] = _ref_blk(k, k, r + rr * TILE, c + cc * TILE);
        }
    }
    if (blockIdx.y == 0) {
        /* rows */
#pragma unroll
        for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
            for (int cc = 0; cc < div_block; cc++) {
                sm[r + rr * TILE][c + cc * TILE] = _ref_blk(i, k, r + rr * TILE, c + cc * TILE);
            }
        }
        __syncthreads();

#pragma unroll
        for (int b = 0; b < block_size; b++) {
#pragma unroll
            for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
                for (int cc = 0; cc < div_block; cc++) {
                    sm[r + rr * TILE][c + cc * TILE] = min(sm[r + rr * TILE][c + cc * TILE], sm[r + rr * TILE][b] + k_k_sm[b][c + cc * TILE]);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
            for (int cc = 0; cc < div_block; cc++) {
                _ref_blk(i, k, r + rr * TILE, c + cc * TILE) = sm[r + rr * TILE][c + cc * TILE];
            }
        }
    } else {
        /* cols */
#pragma unroll
        for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
            for (int cc = 0; cc < div_block; cc++) {
                sm[r + rr * TILE][c + cc * TILE] = _ref_blk(k, i, r + rr * TILE, c + cc * TILE);
            }
        }
        __syncthreads();

#pragma unroll
        for (int b = 0; b < block_size; b++) {
#pragma unroll
            for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
                for (int cc = 0; cc < div_block; cc++) {
                    sm[r + rr * TILE][c + cc * TILE] = min(sm[r + rr * TILE][c + cc * TILE], k_k_sm[r + rr * TILE][b] + sm[b][c + cc * TILE]);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
            for (int cc = 0; cc < div_block; cc++) {
                _ref_blk(k, i, r + rr * TILE, c + cc * TILE) = sm[r + rr * TILE][c + cc * TILE];
            }
        }
    }
}
__global__ void proc_3_blk_glob(int *blk_dist, int s_i, int s_j, int k, int pitch) {
    __shared__ int i_k_sm[block_size][block_size];
    __shared__ int k_j_sm[block_size][block_size];

    int i = s_i + blockIdx.y;
    int j = s_j + blockIdx.x;
    int r = threadIdx.y;
    int c = threadIdx.x;
    int loc[div_block][div_block];

    if (i >= k)
        i++;
    if (j >= k)
        j++;

#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            i_k_sm[r + rr * TILE][c + cc * TILE] = _ref_blk(i, k, r + rr * TILE, c + cc * TILE);
        }
    }
#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            k_j_sm[r + rr * TILE][c + cc * TILE] = _ref_blk(k, j, r + rr * TILE, c + cc * TILE);
        }
    }
    __syncthreads();
#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            loc[rr][cc] = _ref_blk(i, j, r + rr * TILE, c + cc * TILE);
        }
    }

#pragma unroll
    for (int b = 0; b < block_size; b++) {
#pragma unroll
        for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
            for (int cc = 0; cc < div_block; cc++) {
                loc[rr][cc] = min(loc[rr][cc], i_k_sm[r + rr * TILE][b] + k_j_sm[b][c + cc * TILE]);
            }
        }
    }
#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            _ref_blk(i, j, r + rr * TILE, c + cc * TILE) = loc[rr][cc];
        }
    }
}
__global__ void init_blk_dist(int *blk_dist, int pitch) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    blk_dist[r * pitch + c] = (r != c) * INF;
}
__global__ void build_blk_dist(int *edge, int E, int *blk_dist, int pitch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < E) {
        int src = *(edge + idx * 3);
        int dst = *(edge + idx * 3 + 1);
        int w = *(edge + idx * 3 + 2);
        blk_dist[src * pitch + dst] = w;
    }
}