#pragma once

__device__ __forceinline__ bool
GPHashContext::search(bool &ongoing,
                      uint32_t lane_id,
                      uint64_t *my_key,
                      uint64_t *my_value,
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
            uint32_t dest_lane = __ffs(found_lanes) - 1;
            uint32_t dest_level = __shfl_sync(0xFFFFFFFF, src_level, dest_lane, 32);
            uint32_t dest_bucket = __shfl_sync(0xFFFFFFFF, src_bucket, dest_lane, 32);
            uint32_t dest_slot = __shfl_sync(0xFFFFFFFF, src_slot, dest_lane, 32);
            if (lane_id == src_lane)
            {
#ifdef ENABLE_CACHE
                uint32_t dest_cached_bucket = bucket_pos_[dest_level][dest_bucket];
                bool is_dest_cached = (dest_cached_bucket != NOT_CACHED);
                uint64_t *dest_value_address_ptr = is_dest_cached ? getValueAddressPtrFromCache(dest_cached_bucket, dest_slot)
                                                                  : getValueAddressPtr(dest_level, dest_bucket, dest_slot);
#else
                uint64_t *dest_value_address_ptr = getValueAddressPtr(dest_level, dest_bucket, dest_slot);
#endif
                uint64_t dest_value_address = *dest_value_address_ptr;
#ifndef INPLACE_KEY
                dest_value_address <<= 16;
                dest_value_address >>= 16;
                uint64_t *dest_value_ptr = values_ + dest_value_address * (VALUE_SIZE + KEY_SIZE) / 8;
#else
                uint64_t *dest_value_ptr = values_ + dest_value_address * VALUE_SIZE / 8;
#endif
                *my_value = (uint64_t)dest_value_ptr;
                ongoing = false;
            }
        }
        else
        {
            if (lane_id == src_lane)
            {
                ongoing = false;
                ret = false;
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
GPHashContext::search8Byte(bool &ongoing,
                           uint32_t lane_id,
                           uint64_t my_key,
                           uint64_t *my_value,
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
            uint32_t dest_lane = __ffs(found_lanes) - 1;
            uint32_t dest_level = __shfl_sync(0xFFFFFFFF, src_level, dest_lane, 32);
            uint32_t dest_bucket = __shfl_sync(0xFFFFFFFF, src_bucket, dest_lane, 32);
            uint32_t dest_slot = __shfl_sync(0xFFFFFFFF, src_slot, dest_lane, 32);
            if (lane_id == src_lane)
            {
#ifdef ENABLE_CACHE
                uint32_t dest_cached_bucket = bucket_pos_[dest_level][dest_bucket];
                bool is_dest_cached = (dest_cached_bucket != NOT_CACHED);
                uint64_t *dest_value_address_ptr = is_dest_cached ? getValueAddressPtrFromCache(dest_cached_bucket, dest_slot)
                                                                  : getValueAddressPtr(dest_level, dest_bucket, dest_slot);
#else
                uint64_t *dest_value_address_ptr = getValueAddressPtr(dest_level, dest_bucket, dest_slot);
#endif
                uint64_t *dest_value_ptr = values_ + (*dest_value_address_ptr) * VALUE_SIZE / 8;
                *my_value = (uint64_t)dest_value_ptr;
                ongoing = false;
            }
        }
        else
        {
            if (lane_id == src_lane)
            {
                ongoing = false;
                ret = false;
            }
        }

#ifdef ASYNC_LOADING
        if (is_cached)
        {
            decCachedBucketRef(cached_bucket);
        }
#endif

        last_work_queue = work_queue;
    }
    return ret;
}

__device__ __forceinline__ bool
GPHashContext::searchPerThread(bool &ongoing,
                               uint32_t lane_id,
                               uint64_t *my_key,
                               uint64_t *my_value,
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
            uint64_t dest_value_address = *getValueAddressPtr(src_level, src_bucket, src_slot);
            dest_value_address <<= 16;
            dest_value_address >>= 16;
            uint64_t *dest_value_ptr = values_ + dest_value_address * (VALUE_SIZE + KEY_SIZE) / 8;
            *my_value = (uint64_t)dest_value_ptr;
            return true;
        }
    }

    return false;
}