#pragma once

__device__ __forceinline__ void
GPHashContext::fetchBucket(bool &ongoing,
                           uint32_t cached_pos,
                           uint32_t level_id,
                           uint32_t bucket_id)
{
    uint32_t old_level_id = *(cached_bucket_level_ + cached_pos);
    uint32_t old_bucket_id = *(cached_bucket_id_ + cached_pos);
    if (old_level_id != NOT_CACHED)
    {
        bucket_pos_[old_level_id][old_bucket_id] = 0xFFFFFFFF;
        *(cached_bucket_level_ + cached_pos) = level_id;
        *(cached_bucket_id_ + cached_pos) = bucket_id;
    }
#ifdef ASYNC_LOADING
    int cnt = 0;
RE_FETCH:
    if (getCachedBucketRef(cached_pos) != 0) // wait for old cached bucket is not used
    {
        if (++cnt < 5)
        {
            deviceSleep(1e-6);
            goto RE_FETCH;
        }
        else
            return ;
    }
    uint32_t version = getBucketVersion(level_id, bucket_id);
#endif
    for (int i = 0; i < NUM_SLOT; ++i)
    {
        uint64_t *fp_ptr = getFPPtr(level_id, bucket_id, i);
        uint64_t *key_ptr = getKeyPtr(level_id, bucket_id, i);
        uint64_t *value_address_ptr = getValueAddressPtr(level_id, bucket_id, i);

        uint64_t *cache_fp_ptr = getFPPtrFromCache(cached_pos, i);
        uint64_t *cache_key_ptr = getKeyPtrFromCache(cached_pos, i);
        uint64_t *cache_value_address_ptr = getValueAddressPtrFromCache(cached_pos, i);

        *cache_fp_ptr = *fp_ptr;
        setKey(cache_key_ptr, key_ptr);
        *cache_value_address_ptr = *value_address_ptr;
    }
#ifdef ASYNC_LOADING
    if (version != getBucketVersion(level_id, bucket_id))
    {
        if (++cnt < 5)
            goto RE_FETCH;
        else
            return ;
    }
#endif
    bucket_pos_[level_id][bucket_id] = cached_pos;
}