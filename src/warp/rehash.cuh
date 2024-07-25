#pragma once

__device__ __forceinline__ void
GPHashContext::rehash(bool &ongoing,
                      uint32_t lane_id,
                      uint32_t level,
                      uint32_t bucket)
{
    for (int i = 0; i < NUM_SLOT; ++i)
    {
        bool to_insert = false;
        uint64_t *my_key = nullptr;
        uint64_t my_value_address = 0;

        uint64_t *fp_ptr = getFPPtr(level, bucket, i);
        uint64_t fp = *fp_ptr;

        if (fp != EMPTY_KEY_64)
        {
            to_insert = true;
            my_key = getKeyPtr(level, bucket, i);
            my_value_address = *getValueAddressPtr(level, bucket, i);
        }

        insert(to_insert, lane_id, my_key, my_value_address, level + 1);

        if (fp != EMPTY_KEY_64)
        {
            atomicCAS((unsigned long long int *)fp_ptr, fp, EMPTY_KEY_64);
            mfence();
        }
    }
}

__device__ __forceinline__ void
GPHashContext::rehash8Byte(bool &ongoing,
                      uint32_t lane_id,
                      uint32_t level,
                      uint32_t bucket)
{
    for (int i = 0; i < NUM_SLOT; ++i)
    {
        bool to_insert = false;
        uint64_t *my_key = getKeyPtr(level, bucket, i);
        uint64_t my_value_address = 0;
        uint64_t key = *my_key;

        if (key != EMPTY_KEY_64)
        {
            to_insert = true;
            my_value_address = *getValueAddressPtr(level, bucket, i);
        }

        insert8Byte(to_insert, lane_id, key, my_value_address, level + 1);

        if (key != EMPTY_KEY_64)
        {
            atomicCAS((unsigned long long int *)my_key, key, EMPTY_KEY_64);
            mfence();
        }
    }
}