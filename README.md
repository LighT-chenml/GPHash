# GPHash
GPHash: An Efficient Hash Index for GPU with Persistent Memory

## Platform
We use a GPU server for evaluation, the detailed information of the server:
* 2 \* 26-core Intel Xeon Gold 6230R CPUs, 1 Tesla V100 GPU
* 6 \* 32GB DDR4 DIMMs, 6 \* 128GB Intel Optane DC DIMMs
* Ubuntu 18.04 with Linux kernel version 5.4.0
* CUDA toolkit 11.1 with PyTorch 1.8.1

## Setup
```
$ mkdir build
$ cd build
$ cmake ..
$ make
```
## Example
```
PMEM_MMAP_HINT=7fab00000000 ./ycsb_bench $pm_file_path $workload_path $num_load_data $num_run_data $log_batch_size $init_level $cache_rate
```