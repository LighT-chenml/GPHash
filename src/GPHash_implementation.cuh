#pragma once

void GPHash::batchedOperations(uint32_t *d_ops, uint64_t *d_keys, uint64_t *d_values, uint32_t num_ops)
{
    CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    const uint32_t num_blocks = (num_ops + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    batched_operations<<<num_blocks, BLOCKSIZE_, 0, stream>>>(d_ops, d_keys, d_values, num_ops, value_offset_, GPHash_ctx_);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    setValueOffset(value_offset_ + num_ops);
}

void GPHash::checkLevel(uint32_t level)
{
    CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));
    if (level_ptr_[level] == nullptr)
        return;
    const uint32_t num_blocks = ((1 << level_) + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    check_buckets<<<num_blocks, BLOCKSIZE_>>>(level_, GPHash_ctx_);
}

void GPHash::loadCache(uint32_t cache_size)
{
    std::thread t = std::thread(&GPHash::constructCache, this, cache_size);
#ifdef ASYNC_LOADING
    t.detach();
#else
    t.join();
#endif
}

void GPHash::constructCache(uint32_t cache_size)
{
    if (is_constructing_cache)
        return;
    is_constructing_cache = true;

    printf("begin constructing cache!\n");

    uint64_t *cache_ptr = nullptr;
    uint32_t *cache_ref = nullptr;
    uint32_t *cached_bucket_level = nullptr;
    uint32_t *cached_bucket_id = nullptr;
    uint32_t *d_cached_bucket_level = nullptr;
    uint32_t *d_cached_bucket_id = nullptr;
    if (cache_size != cache_size_) // assume new cache_size > 0
    {
        CHECK_CUDA_ERROR(cudaMalloc((void **)&cache_ptr, cache_size * BUCKET_SIZE));
        CHECK_CUDA_ERROR(cudaMemset(cache_ptr, 0xFF, cache_size * BUCKET_SIZE));
        CHECK_CUDA_ERROR(cudaMalloc((void **)&cache_ref, cache_size * sizeof(uint32_t)));
        CHECK_CUDA_ERROR(cudaMemset(cache_ref, 0x00, cache_size * sizeof(uint32_t)));
        CHECK_CUDA_ERROR(cudaMalloc((void **)&cached_bucket_level, cache_size * sizeof(uint32_t)));
        CHECK_CUDA_ERROR(cudaMemset(cached_bucket_level, 0xFF, cache_size * sizeof(uint32_t)));
        CHECK_CUDA_ERROR(cudaMalloc((void **)&cached_bucket_id, cache_size * sizeof(uint32_t)));
        CHECK_CUDA_ERROR(cudaMemset(cached_bucket_id, 0xFF, cache_size * sizeof(uint32_t)));
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_cached_bucket_level, cache_size * sizeof(uint32_t)));
        CHECK_CUDA_ERROR(cudaMemset(d_cached_bucket_level, 0xFF, cache_size * sizeof(uint32_t)));
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_cached_bucket_id, cache_size * sizeof(uint32_t)));
        CHECK_CUDA_ERROR(cudaMemset(d_cached_bucket_id, 0xFF, cache_size * sizeof(uint32_t)));
        if (cache_ptr_ != nullptr)
        {
            CHECK_CUDA_ERROR(cudaFree(cache_ptr_));
            CHECK_CUDA_ERROR(cudaFree(cache_ref_));
            CHECK_CUDA_ERROR(cudaFree(cached_bucket_level_));
            CHECK_CUDA_ERROR(cudaFree(cached_bucket_id_));
            delete h_cached_bucket_level;
            delete h_cached_bucket_id;
        }
        cache_size_ = cache_size;
        cache_ptr_ = cache_ptr;
        cache_ref_ = cache_ref;
        cached_bucket_level_ = cached_bucket_level;
        cached_bucket_id_ = cached_bucket_id;

        GPHash_ctx_.cache_size_ = cache_size_;
        GPHash_ctx_.cache_ptr_ = cache_ptr_;
        GPHash_ctx_.cache_ref_ = cache_ref_;
        GPHash_ctx_.cached_bucket_level_ = cached_bucket_level_;
        GPHash_ctx_.cached_bucket_id_ = cached_bucket_id_;
        h_cached_bucket_level = new uint32_t[cache_size_];
        h_cached_bucket_id = new uint32_t[cache_size_];
        d_cached_bucket_level_ = d_cached_bucket_level;
        d_cached_bucket_id_ = d_cached_bucket_id;
    }

    struct Bucket
    {
        uint32_t level_id;
        uint32_t bucket_id;
        uint32_t value;
        bool operator<(const Bucket &t) const
        {
            if (value != t.value)
                return value > t.value;
            if (level_id != t.level_id)
                return level_id < t.level_id;
            return bucket_id < t.bucket_id;
        }
    };
    std::vector<Bucket> A;

    cudaDeviceSynchronize();

    for (int i = level_; i < level_ + NUM_LEVEL; ++i)
    {
        CHECK_CUDA_ERROR(cudaMemcpy(h_exp, bucket_exp_[i], (1 << i) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        for (int j = 0; j < (1 << i); ++j)
        {
            Bucket a;
            a.level_id = i;
            a.bucket_id = j;
            a.value = h_exp[j];
            A.push_back(a);
        }
    }

    // random_shuffle(A.begin(), A.end());
    nth_element(A.begin(), A.begin() + cache_size_, A.end());

    for (int i = 0; i < cache_size_; ++i)
    {
        h_cached_bucket_level[i] = A[i].level_id;
        h_cached_bucket_id[i] = A[i].bucket_id;
    }

    fetchBuckets();

    is_constructing_cache = false;

    printf("finish constructing cache!\n");
}

void GPHash::fetchBuckets()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // clock_t clock_rate = prop.clockRate;
    // printf("%u\n", clock_rate);
    // printf("%s\n", prop.name);

    if (!prop.deviceOverlap)
    {
        printf("No device will handle overlaps. so no speed up from stream.\n");
        return;
    }

    for (int i = level_; i < level_ + NUM_LEVEL; ++i)
    {
        CHECK_CUDA_ERROR(cudaMemset(bucket_pos_[i], 0xFF, (1LL << i) * sizeof(uint32_t)));
    }

    CHECK_CUDA_ERROR(cudaMemcpy(d_cached_bucket_level_, h_cached_bucket_level, cache_size_ * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_cached_bucket_id_, h_cached_bucket_id, cache_size_ * sizeof(uint32_t), cudaMemcpyHostToDevice));

    printf("begin concurrent fetch!\n");

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    const uint32_t num_blocks = (cache_size_ + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    fetch_buckets<<<num_blocks, BLOCKSIZE_, 0, stream>>>(cache_size_, d_cached_bucket_level_, d_cached_bucket_id_, GPHash_ctx_);

    cudaStreamSynchronize(stream);

    float temp_time = 0.0;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temp_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaStreamDestroy(stream);

    printf("finish concurrent fetch!\n");
    printf("concurrent fetch time: %.3f\tmsec\n", temp_time);
}

uint64_t GPHash::hitCount()
{
    uint64_t sum = 0;
    for (int i = 0; i < MAX_LEVEL; ++i)
        if (bucket_hit_[i] != nullptr)
        {
            uint32_t *h_hit = new uint32_t[1 << i];
            CHECK_CUDA_ERROR(cudaMemcpy(h_hit, bucket_hit_[i], (1 << i) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            cudaDeviceSynchronize();
            for (int j = 0; j < (1 << i); ++j)
            {
                sum += h_hit[j];
            }
        }
    return sum;
}

void GPHash::resize()
{
    *((uint64_t *)pm_base_ + 1) = (uint64_t)1;
    fence();

    CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));

    if (level_ptr_[level_ + NUM_LEVEL] != nullptr)
    {
        *((uint64_t *)pm_base_ + 1) = (uint64_t)0;
        fence();
        return;
    }

    memset(pm_ptr_, 0xFF, (1LL << (level_ + NUM_LEVEL)) * BUCKET_SIZE);
    cudaDeviceSynchronize();
    level_ptr_[level_ + NUM_LEVEL] = (uint64_t *)pm_ptr_;
    pm_ptr_ = (void *)((uint64_t)pm_ptr_ + (1LL << (level_ + NUM_LEVEL)) * BUCKET_SIZE);
    *((uint64_t *)pm_base_) = (uint64_t)pm_ptr_;
    fence();
    *((uint64_t *)pm_base_ + 4 + level_ + NUM_LEVEL) = (uint64_t)level_ptr_[level_ + NUM_LEVEL];
    fence();

    CHECK_CUDA_ERROR(cudaMalloc((void **)&bucket_cnt_[level_ + NUM_LEVEL], (1LL << (level_ + NUM_LEVEL)) * sizeof(uint32_t)));
    CHECK_CUDA_ERROR(cudaMemset(bucket_cnt_[level_ + NUM_LEVEL], 0x00, (1LL << (level_ + NUM_LEVEL)) * sizeof(uint32_t)));

    CHECK_CUDA_ERROR(cudaMalloc((void **)&bucket_pos_[level_ + NUM_LEVEL], (1LL << (level_ + NUM_LEVEL)) * sizeof(uint32_t)));
    CHECK_CUDA_ERROR(cudaMemset(bucket_pos_[level_ + NUM_LEVEL], 0xFF, (1LL << (level_ + NUM_LEVEL)) * sizeof(uint32_t)));

    CHECK_CUDA_ERROR(cudaMalloc((void **)&bucket_hit_[level_ + NUM_LEVEL], (1LL << (level_ + NUM_LEVEL)) * sizeof(uint32_t)));
    CHECK_CUDA_ERROR(cudaMemset(bucket_hit_[level_ + NUM_LEVEL], 0x00, (1LL << (level_ + NUM_LEVEL)) * sizeof(uint32_t)));

    CHECK_CUDA_ERROR(cudaMalloc((void **)&bucket_exp_[level_ + NUM_LEVEL], (1LL << (level_ + NUM_LEVEL)) * sizeof(uint32_t)));
    CHECK_CUDA_ERROR(cudaMemset(bucket_exp_[level_ + NUM_LEVEL], 0x00, (1LL << (level_ + NUM_LEVEL)) * sizeof(uint32_t)));

    CHECK_CUDA_ERROR(cudaMalloc((void **)&bucket_version_[level_ + NUM_LEVEL], (1LL << (level_ + NUM_LEVEL)) * sizeof(uint32_t)));
    CHECK_CUDA_ERROR(cudaMemset(bucket_version_[level_ + NUM_LEVEL], 0x00, (1LL << (level_ + NUM_LEVEL)) * sizeof(uint32_t)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_level_ptr_, level_ptr_, MAX_LEVEL * sizeof(uint64_t *), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_bucket_cnt_, bucket_cnt_, MAX_LEVEL * sizeof(uint32_t *), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_bucket_pos_, bucket_pos_, MAX_LEVEL * sizeof(uint32_t *), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_bucket_hit_, bucket_hit_, MAX_LEVEL * sizeof(uint32_t *), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_bucket_exp_, bucket_exp_, MAX_LEVEL * sizeof(uint32_t *), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_bucket_version_, bucket_version_, MAX_LEVEL * sizeof(uint32_t *), cudaMemcpyHostToDevice));

    if (h_exp != nullptr)
    {
        delete h_exp;
    }

    h_exp = new uint32_t[1 << (level_ + NUM_LEVEL)];

    printf("begin rehash!\n");

    cudaDeviceSynchronize();

    const uint32_t num_blocks = ((1 << level_) + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    level_rehash<<<num_blocks, BLOCKSIZE_>>>(level_, GPHash_ctx_);

    cudaDeviceSynchronize();

    printf("finish rehash!\n");

    level_++;
    GPHash_ctx_.level_ = level_;
    *((uint64_t *)pm_base_ + 2) = (uint64_t)level_;
    fence();

    *((uint64_t *)pm_base_ + 1) = (uint64_t)0;
    fence();

    cudaDeviceSynchronize();
}

uint32_t GPHash::capacity()
{
    uint32_t sum = 0;
    for (int i = level_; i < level_ + NUM_LEVEL; ++i)
    {
        sum += (1 << i) * NUM_SLOT;
    }
    return sum;
}

double GPHash::loadFactor()
{
    uint32_t inserted = 0;
    uint32_t sum = 0;
    for (int i = level_; i < level_ + NUM_LEVEL; ++i)
    {
        sum += (1 << i) * NUM_SLOT;
        for (int j = 0; j < (1 << i); ++j)
            for (int k = 0; k < NUM_SLOT; ++k)
            {
#ifndef BYTE_8

                uint64_t *fp_ptr = (uint64_t *)((uint64_t)(level_ptr_[i]) + (j * NUM_SLOT + k) * FP_SIZE);
                if (*fp_ptr != EMPTY_KEY_64)
                    ++inserted;

#else
                uint64_t key = *((uint64_t *)((uint64_t)(level_ptr_[i]) + (1 << i) * NUM_SLOT * FP_SIZE + (j * NUM_SLOT + k) * KEY_SIZE));
                if (key != EMPTY_KEY_64)
                    ++inserted;

#endif
            }
    }
    return 1.0 * inserted / sum;
}