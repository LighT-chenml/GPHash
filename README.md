# GPHash
This is an open source repository for our paper in [FAST 2025](https://www.usenix.org/conference/fast25)

> **GPHash: An Efficient Hash Index for GPU with Byte-Granularity Persistent Memory**
>
> Menglei Chen, Yu Hua, Zhangyu Chen, Ming Zhang, Gen Dong. *Huazhong University of Science and Technology*

## Brief Introduction

GPU with persistent memory (GPM) provides byte-granular persistency for GPU-powered applications. To achieve efficient data management, hash indexes have been widely used in GPU applications. However, conventional hash indexes become inefficient for GPM systems due to warp-agnostic execution manner, high-overhead consistency guarantee, and huge bandwidth gap between PM and GPU. 

We propose GPHash, an efficient hash index for GPM systems with high performance and consistency guarantee. To fully exploit the parallelism of GPU, GPHash executes all index operations in a lock-free and warp-cooperative manner. Moreover, by using CAS primitive and slot states, GPHash ensures consistency guarantee with low overheads. To further bridge the bandwidth gap between PM and GPU, GPHash caches hot items in GPU memory while minimizing the overhead for cache management. Extensive experimental results show that GPHash outperforms state-of-the-art CPU-assisted data management and GPM hash indexes. 

## Platform
We use a GPU server for evaluation, the detailed information of the server:
* 2 \* 26-core Intel Xeon Gold 6230R CPUs, 1 Tesla V100 GPU
* 6 \* 32GB DDR4 DIMMs, 6 \* 128GB Intel Optane DC DIMMs
* Ubuntu 18.04 with Linux kernel version 5.4.0
* CUDA toolkit 11.1 with PyTorch 1.8.1

## Build
Clone this repo
```
git clone https://github.com/LighT-chenml/GPHash.git
cd GPHash 
```

Run the following commands for compilation
```
mkdir build
cd build
cmake ..
make
```
## Run
The basic structure for a running command is:
```
PMEM_MMAP_HINT=7fab00000000 ./ycsb_bench <pm_file_path> <workload_path> <num_load_ops> <num_run_ops> <log_batch_size> <init_level> <cache_size>
```

Please refer to the following list to set the arguments:

| Argument | Usage Description |
|:----------:|:-------------------:|
| `pm_file_path` | The path to PM file that stores hash index data|
| `workload_path` | The path to the workload file (ignored the .load/.run suffix) |
| `num_load_ops` | The number of operations in load phase |
| `num_run_ops` | The number of operations in run phase|
| `log_batch_size` | The logarithm of the batch size (e.g., set this argument to 12 indicates the batch size is 4096) |
| `init_level` | The logarithm of the number of buckets in initial bottom level (e.g., set this argument to 12 indicates the initial bottom level contains 4096 buckets) |
| `cache_size` | The proportion of cached buckets in all buckets (e.g., set this argument to 0.2 indicates 20% buckets are cached) |