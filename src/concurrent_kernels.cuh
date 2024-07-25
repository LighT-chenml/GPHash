#pragma once

#define LATENCY

__global__ void batched_operations(
    uint32_t *d_ops,
    uint64_t *d_keys,
    uint64_t *d_values,
    uint32_t num_operations,
    uint64_t value_offset,
    GPHashContext GPHash_ctx)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    if ((tid - lane_id) >= num_operations)
        return;

    uint32_t my_ops = 0;
    uint64_t *my_key = nullptr;
    uint64_t *my_value = nullptr;
    uint64_t my_address = 0xFFFFFFFFFFFFFFFFLL;

    if (tid < num_operations)
    {
        my_ops = d_ops[tid];
        my_key = d_keys + tid * KEY_SIZE / 8;
        my_value = d_values + tid * max(KEY_SIZE, VALUE_SIZE) / 8;
    }

    bool to_insert = (my_ops == 1) ? true : false;
    bool to_update = (my_ops == 2) ? true : false;
    bool to_delete = (my_ops == 3) ? true : false;
    bool to_search = (my_ops == 4) ? true : false;

    bool ret;

    if (to_insert || to_update)
    {
        my_address = value_offset + tid;
#ifdef INPLACE_KEY
        GPHash_ctx.setValue(GPHash_ctx.values_ + my_address * VALUE_SIZE / 8, my_value);
#else
        GPHash_ctx.setValue(GPHash_ctx.values_ + my_address * (VALUE_SIZE + KEY_SIZE) / 8, my_value);
        GPHash_ctx.setKey(GPHash_ctx.values_ + (my_address * (VALUE_SIZE + KEY_SIZE) + VALUE_SIZE) / 8, my_key);
#endif
        mfence();
    }

#ifdef WARP_COOPERATE
#ifdef BYTE_8
    ret = GPHash_ctx.insert8Byte(to_insert, lane_id, *my_key, my_address, GPHash_ctx.level_);
#else
    ret = GPHash_ctx.insert(to_insert, lane_id, my_key, my_address, GPHash_ctx.level_);
#endif
#else
    ret = GPHash_ctx.insertPerThread(to_insert, lane_id, my_key, my_address, GPHash_ctx.level_);
#endif
    if (my_ops == 1 && ret) d_ops[tid] = 0;

#ifdef WARP_COOPERATE
#ifdef BYTE_8
    ret = GPHash_ctx.update8Byte(to_update, lane_id, *my_key, my_address, GPHash_ctx.level_);
#else
    ret = GPHash_ctx.update(to_update, lane_id, my_key, my_address, GPHash_ctx.level_);
#endif
#else
    ret = GPHash_ctx.updatePerThread(to_update, lane_id, my_key, my_address, GPHash_ctx.level_);
#endif
    if (my_ops == 2 && ret) d_ops[tid] = 0;

#ifdef WARP_COOPERATE
#ifdef BYTE_8
    ret = GPHash_ctx.del8Byte(to_delete, lane_id, *my_key, GPHash_ctx.level_);
#else
    ret = GPHash_ctx.del(to_delete, lane_id, my_key, GPHash_ctx.level_);
#endif
#else
    ret = GPHash_ctx.delPerThread(to_delete, lane_id, my_key, GPHash_ctx.level_);
#endif
    if (my_ops == 3 && ret) d_ops[tid] = 0;

#ifdef WARP_COOPERATE
#ifdef BYTE_8
    ret = GPHash_ctx.search8Byte(to_search, lane_id, *my_key, my_value, GPHash_ctx.level_);
#else
    ret = GPHash_ctx.search(to_search, lane_id, my_key, my_value, GPHash_ctx.level_);
#endif
#else
    ret = GPHash_ctx.searchPerThread(to_search, lane_id, my_key, my_value, GPHash_ctx.level_);
#endif
    if (my_ops == 4 && ret)
    {
        d_ops[tid] = 0;
        GPHash_ctx.setValue(my_value, (uint64_t *)(*my_value));
    }
}

__global__ void check_buckets(uint32_t level, GPHashContext GPHash_ctx)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    if ((tid - lane_id) >= (1 << level))
        return;

    bool to_check = (tid < (1 << level)) ? true : false;

    GPHash_ctx.checkBucket(to_check, lane_id, level, tid);
}

__global__ void fetch_buckets(uint32_t cache_size, uint32_t *level_id, uint32_t *bucket_id, GPHashContext GPHash_ctx)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    if ((tid - lane_id) >= cache_size)
        return;

    uint32_t my_level_id = 0;
    uint32_t my_bucket_id = 0;

    if (tid < cache_size)
    {
        my_level_id = *(level_id + tid);
        my_bucket_id = *(bucket_id + tid);
    }

    bool to_check = (tid < cache_size) ? true : false;

    GPHash_ctx.fetchBucket(to_check, tid, my_level_id, my_bucket_id);
}


__global__ void level_rehash(uint32_t level, GPHashContext GPHash_ctx)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    if ((tid - lane_id) >= (1 << level))
        return;

    bool to_rehash = (tid < (1 << level)) ? true : false;

#ifdef BYTE_8
    GPHash_ctx.rehash8Byte(to_rehash, lane_id, level, tid);
#else
    GPHash_ctx.rehash(to_rehash, lane_id, level, tid);
#endif
}