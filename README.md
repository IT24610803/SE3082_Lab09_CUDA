# SE3082 – Parallel Computing
## Lab Sheet 9 – Introduction to CUDA (Google Colab Edition)

**Student ID:** IT24610803  
**Module:** SE3082 Parallel Computing  
**Platform:** Google Colab (NVIDIA Tesla T4 GPU)  
**Compiler:** NVIDIA CUDA Compiler (`nvcc`)  

---

## 📌 Repository Contents

| File | Description |
| :--- | :--- |
| **`SE3082_Lab09_IT24610803.ipynb`** | Main Google Colab Jupyter Notebook containing all exercises, code cells, compilation/execution commands, and answers to questions. |
| **`hello.cu`** | Exercise 1: Smoke test launching 1 block of 4 threads (`hello<<<1, 4>>>`). |
| **`ex2_add.cu`** | Exercise 2: Host/device memory transfer and integer addition on 1 thread (`add<<<1, 1>>>`). |
| **`ex3_blocks.cu`** | Exercise 3: Vector addition using 512 blocks with 1 thread each (`add<<<N, 1>>>`). |
| **`ex4_threads.cu`** | Exercise 4: Vector addition using 1 block with 512 threads (`addT<<<1, N>>>`). |
| **`ex5_vecmul.cu`** | Exercise 5: Element-wise multiplication of two $10,000,000$-element vectors using 19,532 blocks $\times$ 512 threads with boundary guards. |
| **`ex6_matrix.cu`** | Exercise 6: Element-wise multiplication of two $10,000 \times 10,000$ matrices ($100,000,000$ elements) using a 2D grid of $313 \times 625$ blocks with $(32, 16)$ threads per block. |

---

## 🚀 How to Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com).
2. Set Runtime: **Runtime &rarr; Change runtime type &rarr; Hardware accelerator: T4 GPU &rarr; Save**.
3. Open or upload `SE3082_Lab09_IT24610803.ipynb`.
4. Run cells sequentially or click **Runtime &rarr; Run all**.

### Compilation & Execution Pattern
```bash
# Write code to file
%%writefile name.cu
# ... code ...

# Compile using nvcc
!nvcc -arch=native -o name.o name.cu

# Execute binary
!./name.o
```

---

## 📊 Summary of Exercises & Verification

### Exercise 5: Vector Multiplication ($10,000,000$ elements)
- **Threads per block:** `512`
- **Blocks launched:** `19,532` blocks
- **Total threads launched:** `10,000,384` threads
- **Boundary guard:** `if (i < n)` protects against accessing out-of-bounds memory by the last 384 threads.
- **CPU Verification:** `0 mismatches out of 10000000`

### Exercise 6: 2D Matrix Multiplication ($10,000 \times 10,000$)
- **Threads per block:** `dim3 threadsPerBlock(32, 16);` (512 threads/block)
- **Grid size:** `dim3 numBlocks(313, 625);` (195,625 blocks)
- **Total threads launched:** `100,160,000` threads
- **CPU Verification:** `0 mismatches out of 100000000`

---

## 💡 Answers to Conceptual Questions

### Exercise 2 &ndash; THINK
1. **Why do we need separate variables `d_a`, `d_b`, `d_c` on the device?**  
   CPU (host) and GPU (device) maintain distinct physical memory spaces. GPU kernels cannot directly dereference host memory addresses without unified memory. Memory must be allocated on the device via `cudaMalloc`.
2. **Why pass `&a` to `cudaMemcpy` but `d_a` to the kernel launch?**  
   `a` is a local CPU variable of type `int`, so its pointer `&a` (`int*`) is passed to copy its data. `d_a` is already a pointer (`int*`) holding the device memory address, so it is passed directly by value.
3. **How many threads run the `add` kernel?**  
   Exactly 1 thread (`<<<1, 1>>>`).

### Exercise 3 &ndash; PREDICT BEFORE YOU RUN
- Launching `add<<<512, 1>>>` spawns 512 blocks with 1 thread each. Because NVIDIA GPUs schedule execution in warps of 32 threads, 31 lanes per warp remain disabled, yielding a SIMD warp efficiency of only $\approx 3.125\%$.

### Exercise 4 &ndash; EXPERIMENT: Find the Limit
- Setting $N = 2048$ in a single block (`<<<1, 2048>>>`) fails with `Kernel launch failed: invalid argument` (or `invalid configuration argument`).
- Modern NVIDIA GPUs impose a hardware limit of **1024 threads per block** (`maxThreadsPerBlock = 1024`).
