#pragma once

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <libpmem.h>
#include <assert.h>
#include <ctime>
#include <unistd.h>

#include <sys/mman.h>

// global declarations
#include "persist.cuh"
#include "GPHash_global.cuh"

// warp implementations of member functions:
#include "warp/insert.cuh"
#include "warp/update.cuh"
#include "warp/search.cuh"
#include "warp/delete.cuh"
#include "warp/rehash.cuh"
#include "warp/check.cuh"
#include "warp/cache.cuh"

// kernels
#include "concurrent_kernels.cuh"

// implementations
#include "GPHash_implementation.cuh"