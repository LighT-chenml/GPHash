#pragma once

__device__ __forceinline__ bool
GPHashContext::del(bool &ongoing,
                   uint32_t lane_id,
                   uint64_t *my_key,
                   uint32_t level)
{
    uint32_t work_queue = 0;
    uint32_t last_work_queue = 0;
    uint32_t src_lane = 0;
    uint64_t src_hash_value = 0;
    uint64_t src_fp = 0;
    uint64_t *src_key_ptr = nullptr;
    uint32_t related_level = (31 - lane_id) / NUM_SLOT_PER_GROUP;
    uint32_t related_slot = (31 - lane_id) - related_level * NUM_SLOT_PER_GROUP;
    uint32_t src_level = related_level + level;
    uint32_t src_bucket = 0;
    uint32_t src_slot = related_slot % NUM_SLOT;

    bool ret = true;

    while ((work_queue = __ballot_sync(0xFFFFFFFF, ongoing)))
    {

        if (last_work_queue != work_queue) // a new operation
        {
            src_lane = __ffs(work_queue) - 1;

            src_key_ptr = (uint64_t *)(((uint64_t)__shfl_sync(0xFFFFFFFF, (uint64_t)my_key >> 32, src_lane, 32) << 32) | __shfl_sync(0xFFFFFFFF, (uint64_t)my_key, src_lane, 32));
            src_hash_value = calHash(src_key_ptr, related_slot / NUM_SLOT);
            src_bucket = computeBucket(src_hash_value, src_level);
            src_fp = ((uint64_t)__shfl_sync(0xFFFFFFFF, src_hash_value >> 32, 31, 32) << 32) | __shfl_sync(0xFFFFFFFF, src_hash_value, 31, 32);
        }

#ifndef INPLACE_KEY
        src_fp <<= 48;
#endif

#ifdef ENABLE_CACHE
        uint32_t cached_bucket = bucket_pos_[src_level][src_bucket];
        bool is_cached = (cached_bucket != NOT_CACHED);
        uint64_t *fp_ptr = is_cached ? getFPPtrFromCache(cached_bucket, src_slot) : getFPPtr(src_level, src_bucket, src_slot);
        uint64_t fp = *fp_ptr;
        uint64_t *key_ptr = is_cached ? getKeyPtrFromCache(cached_bucket, src_slot) : getKeyPtr(src_level, src_bucket, src_slot);

        uint32_t cached_count = __ballot_sync(0xFFFFFFFF, is_cached);
#ifdef COUNT_HIT
        if (is_cached)
        {
            addBucketHit(src_level, src_bucket);
        }

#endif
#ifdef ASYNC_LOADING
        if (is_cached)
        {
            addCachedBucketRef(cached_bucket);
        }
#endif
        addBucketExp(src_level, src_bucket, cached_count);
#else
        uint64_t *fp_ptr = getFPPtr(src_level, src_bucket, src_slot);
        uint64_t fp = *fp_ptr;
        uint64_t *key_ptr = getKeyPtr(src_level, src_bucket, src_slot);
#endif

        uint32_t found_lanes = __ballot_sync(0xFFFFFFFF, compareKey(fp, key_ptr, src_fp, src_key_ptr));
        if (found_lanes)
        {
            if ((found_lanes >> lane_id) & 1 == 1)
            {
#ifdef ENABLE_CACHE
                uint64_t *pm_fp_ptr = !is_cached ? fp_ptr : getFPPtr(src_level, src_bucket, src_slot);
#else
                uint64_t *pm_fp_ptr = fp_ptr;
#endif
                uint64_t old_fp = atomicCAS((unsigned long long int *)pm_fp_ptr, fp, EMPTY_KEY_64);
                if (old_fp == fp)
                {
                    mfence();
                    decBucketCnt(src_level, src_bucket);
#ifdef ENABLE_CACHE
                    if (is_cached)
                    {
                        atomicCAS((unsigned long long int *)fp_ptr, fp, EMPTY_KEY_64);
                    }
#ifdef ASYNC_LOADING
                    else
                    {
                        addBucketVersion(src_level, src_bucket);
                        if (bucket_pos_[src_level][src_bucket] != NOT_CACHED)
                        {
                            cached_bucket = bucket_pos_[src_level][src_bucket];
                            fp_ptr = getFPPtrFromCache(cached_bucket, src_slot);
                            atomicCAS((unsigned long long int *)fp_ptr, fp, EMPTY_KEY_64);
                        }
                    }
#endif
#endif
                }
            }

            if (lane_id == src_lane)
            {
                ongoing = false;
            }
        }
        else
        {
            if (lane_id == src_lane)
            {
                ongoing = false;
            }
        }

#ifdef ENABLE_CACHE
#ifdef ASYNC_LOADING
        if (is_cached)
        {
            decCachedBucketRef(cached_bucket);
        }
#endif
#endif

        last_work_queue = work_queue;
    }
    return ret;
}

__device__ __forceinline__ bool
GPHashContext::del8Byte(bool &ongoing,
                        uint32_t lane_id,
                        uint64_t my_key,
                        uint32_t level)
{
    uint32_t work_queue = 0;
    uint32_t last_work_queue = 0;
    uint32_t src_lane = 0;
    uint64_t src_hash_value = 0;
    uint64_t src_key = 0;
    uint32_t related_level = (31 - lane_id) / NUM_SLOT_PER_GROUP;
    uint32_t related_slot = (31 - lane_id) - related_level * NUM_SLOT_PER_GROUP;
    uint32_t src_level = related_level + level;
    uint32_t src_bucket = 0;
    uint32_t src_slot = related_slot % NUM_SLOT;

    bool ret = true;

    while ((work_queue = __ballot_sync(0xFFFFFFFF, ongoing)))
    {

        if (last_work_queue != work_queue) // a new operation
        {
            src_lane = __ffs(work_queue) - 1;

            src_key = ((uint64_t)__shfl_sync(0xFFFFFFFF, my_key >> 32, src_lane, 32) << 32) | __shfl_sync(0xFFFFFFFF, my_key, src_lane, 32);
            src_hash_value = calHash8Byte(src_key, related_slot / NUM_SLOT);
            src_bucket = computeBucket(src_hash_value, src_level);
        }
#ifdef ENABLE_CACHE
        uint32_t cached_bucket = bucket_pos_[src_level][src_bucket];
        bool is_cached = (cached_bucket != NOT_CACHED);
        uint64_t *key_ptr = is_cached ? getKeyPtrFromCache(cached_bucket, src_slot) : getKeyPtr(src_level, src_bucket, src_slot);
        uint64_t key = *key_ptr;

        uint32_t cached_count = __ballot_sync(0xFFFFFFFF, is_cached);
#ifdef COUNT_HIT
        if (is_cached)
        {
            addBucketHit(src_level, src_bucket);
        }

#endif
#ifdef ASYNC_LOADING
        if (is_cached)
        {
            addCachedBucketRef(cached_bucket);
        }
#endif
        addBucketExp(src_level, src_bucket, cached_count);
#else
        uint64_t *key_ptr = getKeyPtr(src_level, src_bucket, src_slot);
        uint64_t key = *key_ptr;
#endif
        uint32_t found_lanes = __ballot_sync(0xFFFFFFFF, key == src_key);
        if (found_lanes)
        {
            if ((found_lanes >> lane_id) & 1 == 1)
            {
#ifdef ENABLE_CACHE
                uint64_t *pm_key_ptr = !is_cached ? key_ptr : getFPPtr(src_level, src_bucket, src_slot);
#else
                uint64_t *pm_key_ptr = key_ptr;
#endif
                uint64_t old_key = atomicCAS((unsigned long long int *)pm_key_ptr, key, EMPTY_KEY_64);
                if (old_key == key)
                {
                    mfence();
                    decBucketCnt(src_level, src_bucket);
#ifdef ENABLE_CACHE
                    if (is_cached)
                    {
                        atomicCAS((unsigned long long int *)key_ptr, key, EMPTY_KEY_64);
                    }
#ifdef ASYNC_LOADING
                    else
                    {
                        addBucketVersion(src_level, src_bucket);
                        if (bucket_pos_[src_level][src_bucket] != NOT_CACHED)
                        {
                            cached_bucket = bucket_pos_[src_level][src_bucket];
                            key_ptr = getKeyPtrFromCache(cached_bucket, src_slot);
                            atomicCAS((unsigned long long int *)key_ptr, key, EMPTY_KEY_64);
                        }
                    }
#endif
#endif
                }
            }

            if (lane_id == src_lane)
            {
                ongoing = false;
            }
        }
        else
        {
            if (lane_id == src_lane)
            {
                ongoing = false;
            }
        }

#ifdef ENABLE_CACHE
#ifdef ASYNC_LOADING
        if (is_cached)
        {
            decCachedBucketRef(cached_bucket);
        }
#endif
#endif

        last_work_queue = work_queue;
    }
    return ret;
}

__device__ __forceinline__ bool
GPHashContext::delPerThread(bool &ongoing,
                            uint32_t lane_id,
                            uint64_t *my_key,
                            uint32_t level)
{
    uint64_t src_hash_value = 0;
    uint64_t *src_key_ptr = my_key;
    uint32_t related_level = (31 - lane_id) / NUM_SLOT_PER_GROUP;
    uint32_t related_slot = (31 - lane_id) - related_level * NUM_SLOT_PER_GROUP;
    uint32_t src_level = related_level + level;
    uint32_t src_bucket = 0;
    uint32_t src_slot = related_slot % NUM_SLOT;
    uint64_t src_fp = calHash(src_key_ptr, 0) << 48;

    bool ret = true;

    if (ongoing == false)
        return ret;

    for (int i = 0; i < 32; ++i)
    {
        related_level = (31 - i) / NUM_SLOT_PER_GROUP;
        related_slot = (31 - i) - related_level * NUM_SLOT_PER_GROUP;
        src_level = related_level + level;
        src_slot = related_slot % NUM_SLOT;

        src_hash_value = calHash(src_key_ptr, related_slot / NUM_SLOT);
        src_bucket = computeBucket(src_hash_value, src_level);

        uint64_t *fp_ptr = getFPPtr(src_level, src_bucket, src_slot);
        uint64_t fp = *fp_ptr;
        uint64_t *key_ptr = getKeyPtr(src_level, src_bucket, src_slot);

        uint32_t is_found = compareKey(fp, key_ptr, src_fp, src_key_ptr);
        if (is_found)
        {
            uint64_t old_fp = atomicCAS((unsigned long long int *)fp_ptr, fp, EMPTY_KEY_64);
            if (old_fp == fp)
            {
                mfence();
                decBucketCnt(src_level, src_bucket);
            }
        }
    }

    return true;
}