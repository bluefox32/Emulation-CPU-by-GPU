import numpy as np
from numba import cuda
import time

# CUDAカーネルの定義
@cuda.jit
def matrix_multiply_kernel(A, B, C):
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp

# 行列サイズの設定
matrix_size = (1000, 1000)
A = np.random.random(matrix_size).astype(np.float32)
B = np.random.random(matrix_size).astype(np.float32)
C = np.zeros(matrix_size, dtype=np.float32)

# 行列をGPUに転送
A_gpu = cuda.to_device(A)
B_gpu = cuda.to_device(B)
C_gpu = cuda.to_device(C)

# 仮想的なコア数を指定するためのスレッドブロックのサイズを動的に設定
def run_with_core_count(core_count):
    threads_per_block = (core_count, core_count)
    blocks_per_grid_x = int(np.ceil(matrix_size[0] / threads_per_block[0]))
    blocks_per_grid_y = int(np.ceil(matrix_size[1] / threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    start_time = time.time()
    matrix_multiply_kernel[blocks_per_grid, threads_per_block](A_gpu, B_gpu, C_gpu)
    cuda.synchronize()
    end_time = time.time()
    
    print(f"コア数（スレッドブロックサイズ）: {core_count} x {core_count}")
    print(f"計算時間: {end_time - start_time:.2f} 秒")

# 例として、1から6のコア数（スレッドブロックサイズ）で計算を実行
for core_count in range(1, 7):
    run_with_core_count(core_count)