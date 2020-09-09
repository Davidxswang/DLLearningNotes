# CUDA
> This notes is based on Ubuntu 20.04, CUDA 10.1.243, GCC 9.3.0.

## Concepts

### Hardware: SM, Warp, Thread and CUDA Core

- SM: A GPU is formed by multiple units named Streaming Multiprocessors (SMs). Each SM can execute many threads concurrently. 
- Warp (CUDA core): Threads in SMs are group by warps. A warp contains 32 threads, and these 32 threads can only execute the same task at the same time, which is different from CPUs where different threads (on different logic cores) can execute different tasks simultaneously. A warp is a CUDA core.
- Threads: similar to Single Instruction, Multiple Data (SIMD) of CPU.

### Hardware: Compute Capability

The `Compute Capability` uses `SM version` to discriminate hardware capabilities, denoted by `X.Y` where `X` is the major version and `Y` is the minor version. The major version maps to the verion of the hardware architecture, as follow:

- 7: Volta
- 6: Pascal
- 5: Maxwell
- 3: Kepler
- 2: Fermi
- 1: Tesla

### Software: CUDA Version

The CUDA version is different from `Compute Capability`. CUDA version is the version of CUDA software platform. Usually a later CUDA version will support new hardware architectures and add more features.

### Software: Grid, Block, Tread

- Grid: Blocks are organized in grids in 1, 2 or 3 dimentions. A kernel will run on a grid.
- Block: A block can contain up to 1024 threads. Threads in a block can be organized in 1, 2 or 3 dimensions. If a block contains more than 32 threads, multiple warps might be involved. Each block can only be executed within one SM. Every block should be able to run independently so that the kernel can be deployed on the grid with different shapes of block layout.
- Thread: A thread is single process that processes the data.

As far as how many grids on a GPU, how many blocks in a grid, and how many threads in a block, we can refer to the [compute capability table provided by NVIDIA](https://docs.nvidia.com/cuda/archive/10.1/cuda-c-programming-guide/index.html#compute-capabilities).

## Memory Hierarchy and Heterogeneous Programming

### Memory Hierarchy

There are 5 types of memory:

- per-thread local memory: this is the private memory for each thread
- per-block shared memory: this is the memory shared by all the threads inside a block
- global memory: this is the global memory shared by all the threads, persistent across kernel launches by the same application
- context memory (read-only): this is accessible by all the threads, persistent across kernel launches by the same application
- texture memory (read-only): this is accessible by all the threads, persistent across kernel launches by the same application

### Heterogenemous Programming

There are two roles participating in the programming: host and device. Usually host is the CPU that the C program runs on, while device is the GPU that kernels run on. Host and device will have separate memory spaces in DRAM: host memory and device memory.

**Unified Memory** provides *managed memory* to bridge the host and device memory spaces so that we don't need to explicitly mirror data on host and devices.

## Kernel

A function that the GPU can run, is called a kernel. Each kernel uses a grid. A kernel can be executed by multiple equally-shaped thread blocks. A kernel function is indicated by a `__global__` identifier in front of it.

### Access Thread, Block and Grid Information within a Kernel

To access the information about the thread, block and grid within a kernel, use the four variables (all `dim3 structure`):

- `threadIdx`: ID of the current thread
- `blockIdx`: ID of the current block
- `blockDim`: size of each dimension of the current block
- `gridDim`: size of each dimension of the current grid

Use `threadIdx.x` to access the value of x dimension of threadIdx. Same for others. For example, the thread ID of a thread of index `(x, y, z)` can be calculated as: `x + y * blockDim.x + z * blockDim.y * blockDim.x`.

### Kernel Code Example

```CPP (CUDA)
__global__
void add(int n, float *x, float *y)
{
    int t = threadIdx.x;
    int T = blockDim.x;
    for (int i = t; i < n; i += T)
        y[i] = x[i] + y[i]
}
```

### Call a Kernel

`add<<<dim3(1,1,1), dim3(T,1,1)>>>(N, X, Y)`: first `dim3(1,1,1)` specifies the dimensions of the grid and the second dimentions of the block. This line of code will execute the kernel function on 1 block with T threads organized in 1-D.

`add<<<1, T>>>(N, X, Y)`: `1` specifies the number of blocks, `T` specifies number of threads per block.

### Memory Management

- `cudaMallocManaged()`: **allocate** Unified Memory accessible from CPU or GPU
- `cudaFree()`: **free** the memory

### Synchronization

- `__syncthreads`: all the threads in the block will wait before any is allowed to proceed. This is used for threads to synchronize on shared memory.
- `cudaDeviceSynchronize()`: wait for GPU to finish executing a kernel

## Reference

- [Some CUDA concepts explained](https://medium.com/@shiyan/some-cuda-concepts-explained-12ecc390d10f)
- [Parallel Programming With CUDA Tutorial (Part-2: Basics)](https://medium.com/@saadmahmud14/parallel-programming-with-cuda-tutorial-part-2-96f6eaea2832)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/archive/10.1/cuda-c-programming-guide/index.html)