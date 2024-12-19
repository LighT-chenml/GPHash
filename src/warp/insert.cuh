#pragma once

__device__ __forceinline__ bool
GPHashContext::insert(bool &ongoing,
                      uint32_t lane_id,
                      uint64_t *my_key,
                      uint64_t my_value_address,
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
    uint64_t target_fp = 0;

    bool ret = true;

    int count = 0;

    while ((work_queue = __ballot_sync(0xFFFFFFFF, ongoing)))
    {

        if (last_work_queue != work_queue) // a new operation
        {
            count = 0;

            src_lane = __ffs(work_queue) - 1;

            src_key_ptr = (uint64_t *)(((uint64_t)__shfl_sync(0xFFFFFFFF, (uint64_t)my_key >> 32, src_lane, 32) << 32) | __shfl_sync(0xFFFFFFFF, (uint64_t)my_key, src_lane, 32));
            src_hash_value = calHash(src_key_ptr, related_slot / NUM_SLOT);
            src_bucket = computeBucket(src_hash_value, src_level);
            src_fp = ((uint64_t)__shfl_sync(0xFFFFFFFF, src_hash_value >> 32, 31, 32) << 32) | __shfl_sync(0xFFFFFFFF, src_hash_value, 31, 32);
#ifndef INPLACE_KEY
            src_fp <<= 48;
            target_fp = my_value_address | src_fp;
#endif
        }

#ifdef ENABLE_CACHE
        uint32_t cached_bucket = bucket_pos_[src_level][src_bucket];
        bool is_cached = (cached_bucket != NOT_CACHED);

        uint32_t cached_count = __ballot_sync(0xFFFFFFFF, is_cached);
#ifdef COUNT_HIT
        if (count == 0 && is_cached)
        {
            addBucketHit(src_level, src_bucket);
        }

#endif

        if (++count > 2)
            is_cached = false;

        uint64_t *fp_ptr = is_cached ? getFPPtrFromCache(cached_bucket, src_slot) : getFPPtr(src_level, src_bucket, src_slot);
        uint64_t fp = *fp_ptr;
        uint64_t *key_ptr = is_cached ? getKeyPtrFromCache(cached_bucket, src_slot) : getKeyPtr(src_level, src_bucket, src_slot);

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
        uint32_t bucket_cnt = getBucketCnt(src_level, src_bucket);

        uint32_t found_lanes = __ballot_sync(0xFFFFFFFF, compareKey(fp, key_ptr, src_fp, src_key_ptr));
        if (found_lanes)
        {
            if (lane_id == src_lane)
            {
                ongoing = false;
            }
        }
        else
        {
            uint32_t empty_lanes = __ballot_sync(0xFFFFFFFF, fp == EMPTY_KEY_64);

            if (!empty_lanes)
            {
                if (lane_id == src_lane)
                {
                    ongoing = false;
                    ret = false;
                }
            }
            else
            {
                uint32_t dest_lane = 0;
                for (int i = 0; i <= NUM_SLOT; ++i)
                {
                    uint32_t dest_lanes = __ballot_sync(0xFFFFFFFF, bucket_cnt == i) & empty_lanes;
                    if (dest_lanes)
                    {
                        dest_lane = __ffs(dest_lanes) - 1;
                        break;
                    }
                }

                uint32_t dest_level = __shfl_sync(0xFFFFFFFF, src_level, dest_lane, 32);
                uint32_t dest_bucket = __shfl_sync(0xFFFFFFFF, src_bucket, dest_lane, 32);
                uint32_t dest_slot = __shfl_sync(0xFFFFFFFF, src_slot, dest_lane, 32);
                if (lane_id == src_lane)
                {
                    uint64_t *dest_fp_ptr = getFPPtr(dest_level, dest_bucket, dest_slot);
#ifndef INPLACE_KEY
                    uint64_t old_fp = atomicCAS((unsigned long long int *)dest_fp_ptr, EMPTY_KEY_64, target_fp);
                    if (old_fp == EMPTY_KEY_64)
                    {
                        mfence();
                        ongoing = false;
                    }
#else
                    uint64_t old_fp = atomicCAS((unsigned long long int *)dest_fp_ptr, EMPTY_KEY_64, INSERTING_64);
                    if (old_fp == EMPTY_KEY_64)
                    {
                        mfence();
                        addBucketCnt(dest_level, dest_bucket);
                        uint64_t *dest_key_ptr = getKeyPtr(dest_level, dest_bucket, dest_slot);
                        uint64_t *dest_value_address_ptr = getValueAddressPtr(dest_level, dest_bucket, dest_slot);
                        setKey(dest_key_ptr, my_key);
                        *dest_value_address_ptr = my_value_address;
                        mfence();
                        atomicCAS((unsigned long long int *)dest_fp_ptr, INSERTING_64, src_fp);
                        mfence();

#ifdef ENABLE_CACHE
                    INSERT_CACHE_32_BYTE:
                        uint32_t dest_cached_bucket = bucket_pos_[dest_level][dest_bucket];
                        bool is_dest_cached = (dest_cached_bucket != NOT_CACHED);
                        if (is_dest_cached)
                        {
                            dest_fp_ptr = getFPPtrFromCache(dest_cached_bucket, dest_slot);
                            atomicCAS((unsigned long long int *)dest_fp_ptr, EMPTY_KEY_64, INSERTING_64);
                            dest_key_ptr = getKeyPtrFromCache(dest_cached_bucket, dest_slot);
                            dest_value_address_ptr = getValueAddressPtrFromCache(dest_cached_bucket, dest_slot);
                            setKey(dest_key_ptr, my_key);
                            *dest_value_address_ptr = my_value_address;
                            atomicCAS((unsigned long long int *)dest_fp_ptr, INSERTING_64, src_fp);
                        }
#ifdef ASYNC_LOADING
                        else
                        {
                            addBucketVersion(dest_level, dest_bucket);
                            if (bucket_pos_[dest_level][dest_bucket] != NOT_CACHED)
                                goto INSERT_CACHE_32_BYTE;
                        }
#endif
#endif

                        ongoing = false;
                    }
#endif
                }
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
GPHashContext::insert8Byte(bool &ongoing,
                           uint32_t lane_id,
                           uint64_t my_key,
                           uint64_t my_value_address,
                           uint32_t level)
{
    uint32_t work_queue = 0;
    uint32_t last_work_queue = 0;
    uint32_t src_lane = 0;
    uint32_t src_hash_value = 0;
    uint64_t src_key = 0;
    uint32_t related_level = (31 - lane_id) / NUM_SLOT_PER_GROUP;
    uint32_t related_slot = (31 - lane_id) - related_level * NUM_SLOT_PER_GROUP;
    uint32_t src_level = related_level + level;
    uint32_t src_bucket = 0;
    uint32_t src_slot = related_slot % NUM_SLOT;

    bool ret = true;

    int count = 0;

    while ((work_queue = __ballot_sync(0xFFFFFFFF, ongoing)))
    {

        if (last_work_queue != work_queue) // a new operation
        {
            count = 0;

            src_lane = __ffs(work_queue) - 1;

            src_key = ((uint64_t)__shfl_sync(0xFFFFFFFF, my_key >> 32, src_lane, 32) << 32) | __shfl_sync(0xFFFFFFFF, my_key, src_lane, 32);
            src_hash_value = calHash8Byte(src_key, related_slot / NUM_SLOT);
            src_bucket = computeBucket(src_hash_value, src_level);
        }

#ifdef ENABLE_CACHE
        uint32_t cached_bucket = bucket_pos_[src_level][src_bucket];
        bool is_cached = (cached_bucket != NOT_CACHED);

        uint32_t cached_count = __ballot_sync(0xFFFFFFFF, is_cached);
#ifdef COUNT_HIT
        if (count == 0 && is_cached)
        {
            addBucketHit(src_level, src_bucket);
        }

#endif
        if (++count > 2) // retry count
            is_cached = false;

        uint64_t *key_ptr = is_cached ? getKeyPtrFromCache(cached_bucket, src_slot) : getKeyPtr(src_level, src_bucket, src_slot);
        uint64_t key = *key_ptr;

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
        uint32_t bucket_cnt = getBucketCnt(src_level, src_bucket);

        uint32_t found_lanes = __ballot_sync(0xFFFFFFFF, key == src_key);
        if (found_lanes)
        {
            if (lane_id == src_lane)
            {
                ongoing = false;
            }
        }
        else
        {
            uint32_t empty_lanes = __ballot_sync(0xFFFFFFFF, key == EMPTY_KEY_64);

            if (!empty_lanes)
            {
                if (lane_id == src_lane)
                {
                    ongoing = false;
                    ret = false;
                }
            }
            else
            {
                uint32_t dest_lane;
                for (int i = 0; i <= NUM_SLOT; ++i)
                {
                    uint32_t dest_lanes = __ballot_sync(0xFFFFFFFF, bucket_cnt == i) & empty_lanes;
                    if (dest_lanes)
                    {
                        dest_lane = __ffs(dest_lanes) - 1;
                        break;
                    }
                }
                uint32_t dest_level = __shfl_sync(0xFFFFFFFF, src_level, dest_lane, 32);
                uint32_t dest_bucket = __shfl_sync(0xFFFFFFFF, src_bucket, dest_lane, 32);
                uint32_t dest_slot = __shfl_sync(0xFFFFFFFF, src_slot, dest_lane, 32);
                if (lane_id == src_lane)
                {
                    uint64_t *dest_key_ptr = getKeyPtr(dest_level, dest_bucket, dest_slot);
                    uint64_t old_state = atomicCAS((unsigned long long int *)dest_key_ptr, EMPTY_KEY_64, INSERTING_64);
                    if (old_state == EMPTY_KEY_64)
                    {
                        mfence();
                        addBucketCnt(dest_level, dest_bucket);
                        uint64_t *dest_key_ptr = getKeyPtr(dest_level, dest_bucket, dest_slot);
                        uint64_t *dest_value_address_ptr = getValueAddressPtr(dest_level, dest_bucket, dest_slot);
                        *dest_value_address_ptr = my_value_address;
                        atomicCAS((unsigned long long int *)dest_key_ptr, INSERTING_64, src_key);
                        mfence();

#ifdef ENABLE_CACHE
                    INSERT_CACHE_8_BYTE:
                        uint32_t dest_cached_bucket = bucket_pos_[dest_level][dest_bucket];
                        bool is_dest_cached = (dest_cached_bucket != NOT_CACHED);
                        if (is_dest_cached)
                        {
                            dest_key_ptr = getKeyPtrFromCache(dest_cached_bucket, dest_slot);
                            atomicCAS((unsigned long long int *)dest_key_ptr, EMPTY_KEY_64, INSERTING_64);
                            dest_value_address_ptr = getValueAddressPtrFromCache(dest_cached_bucket, dest_slot);
                            *dest_value_address_ptr = my_value_address;
                            atomicCAS((unsigned long long int *)dest_key_ptr, INSERTING_64, src_key);
                        }
#ifdef ASYNC_LOADING
                        else
                        {
                            addBucketVersion(dest_level, dest_bucket);
                            if (bucket_pos_[dest_level][dest_bucket] != NOT_CACHED)
                                goto INSERT_CACHE_8_BYTE;
                        }
#endif
#endif

                        ongoing = false;
                    }
                }
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
GPHashContext::insertPerThread(bool &ongoing,
                               uint32_t lane_id,
                               uint64_t *my_key,
                               uint64_t my_value_address,
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
    uint64_t target_fp = my_value_address | src_fp;

    bool ret = true;

    if (ongoing == false) return ret;

RETRY_INSERT_PER_THREAD:
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
            return true;
        }
    }

    uint32_t dest_lane = 0xFFFFFFFF;
    uint32_t min_bucket_cnt = 0;

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

        uint32_t bucket_cnt = getBucketCnt(src_level, src_bucket);

        uint32_t is_empty = (fp == EMPTY_KEY_64);

        if (is_empty)
        {
            if (dest_lane == 0xFFFFFFFF) dest_lane = i, min_bucket_cnt = bucket_cnt;
            else if (min_bucket_cnt > bucket_cnt) dest_lane = i, min_bucket_cnt = bucket_cnt;
        }
    }

    if (dest_lane == 0xFFFFFFFF) return false;

    related_level = (31 - dest_lane) / NUM_SLOT_PER_GROUP;
    related_slot = (31 - dest_lane) - related_level * NUM_SLOT_PER_GROUP;
    src_level = related_level + level;
    src_slot = related_slot % NUM_SLOT;

    src_hash_value = calHash(src_key_ptr, related_slot / NUM_SLOT);
    src_bucket = computeBucket(src_hash_value, src_level);

    uint32_t dest_level = src_level;
    uint32_t dest_bucket = src_bucket;
    uint32_t dest_slot = src_slot;

    uint64_t *dest_fp_ptr = getFPPtr(dest_level, dest_bucket, dest_slot);
    uint64_t old_fp = atomicCAS((unsigned long long int *)dest_fp_ptr, EMPTY_KEY_64, target_fp);
    if (old_fp == EMPTY_KEY_64)
    {
        mfence();
        addBucketCnt(dest_level, dest_bucket);
    }
    else
    {
        goto RETRY_INSERT_PER_THREAD;
    }

    return true;
}