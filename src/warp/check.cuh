#pragma once

__device__ __forceinline__ void
GPHashContext::checkBucket(bool &ongoing,
                           uint32_t lane_id,
                           uint32_t level,
                           uint32_t bucket)
{
    for (int i = 0; i < NUM_SLOT; ++i)
    {
        uint64_t *fp_ptr = getFPPtr(level, bucket, i);
        uint64_t fp = *fp_ptr;

        if (fp == INSERTING_64)
        {
            atomicCAS((unsigned long long int *)fp_ptr, INSERTING_64, EMPTY_KEY_64);
            mfence();
        }
        else
        {
            addBucketCnt(level, bucket);
        }
    }
}