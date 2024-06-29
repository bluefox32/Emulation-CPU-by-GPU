import numpy as np
from numba import cuda
import time

# 仮想的なデータ生成関数（実行中のソフトウェアから取得するデータをシミュレート）
def get_data_from_software():
    return np.random.random((1000, 1000)).astype(np.float32), np.random.random((1000, 1000)).astype(np.float32)

# CUDAカーネルの定義
@cuda.jit
def matrix_multiply_kernel(A, B, C):
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp

# メイン処理
def main():
    # 実行中のソフトウェアからデータを取得
    A_data, B_data = get_data_from_software()

    # Unified Memoryを使用して行列を生成
    A = cuda.to_device(A_data)
    B = cuda.to_device(B_data)
    C = cuda.managed_array(A_data.shape, dtype=np.float32)

    # 仮想的なコア数を指定するためのスレッドブロックのサイズを動的に設定
    def run_with_core_count(core_count):
        threads_per_block = (core_count, core_count)
        blocks_per_grid_x = int(np.ceil(A.shape[0] / threads_per_block[0]))
        blocks_per_grid_y = int(np.ceil(A.shape[1] / threads_per_block[1]))
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        
        start_time = time.time()
        matrix_multiply_kernel[blocks_per_grid, threads_per_block](A, B, C)
        cuda.synchronize()
        end_time = time.time()
        
        print(f"コア数（スレッドブロックサイズ）: {core_count} x {core_count}")
        print(f"計算時間: {end_time - start_time:.2f} 秒")

    # 例として、1から6のコア数（スレッドブロックサイズ）で計算を実行
    for core_count in range(1, 7):
        run_with_core_count(core_count)

    # 結果を取得して反映（ここでは結果を表示するのみ）
    result = C.copy_to_host()
    print(result)

if __name__ == '__main__':
    main()