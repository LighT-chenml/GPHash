#pragma once

#include "GPHash.cuh"

class HashTable
{
public:
    uint32_t max_batch_size_;
    uint32_t device_idx_;

    uint32_t *d_ops_;
    uint64_t *d_keys_;
    uint64_t *d_values_;

    GPHash *hash_table_;
    void *pm_base_;
    double cache_size_rate_;

    HashTable(char *pm_file, u_int64_t file_size, uint32_t max_batch_size, uint32_t &level, double cache_size_rate, uint32_t device_idx)
        : max_batch_size_(max_batch_size), cache_size_rate_(cache_size_rate), device_idx_(device_idx)
    {
        int32_t devCount = 0;
        CHECK_CUDA_ERROR(cudaGetDeviceCount(&devCount));
        assert(device_idx_ < devCount);
        CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));

        bool recovery = false;

        if (access(pm_file, F_OK) == 0)
        {
            recovery = true;
        }

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        size_t mapped_len;
        int is_pmem;

        if (!recovery)
        {
            pm_base_ = (void *)pmem_map_file(pm_file, file_size, PMEM_FILE_CREATE, 0666, &mapped_len, &is_pmem);
        }
        else
        {
            pm_base_ = (void *)pmem_map_file(pm_file, 0, PMEM_FILE_EXCL, 0, &mapped_len, &is_pmem);
        }

        if (pm_base_ == NULL)
        {
            printf("pmem_map_file fail\n");
            exit(1);
        }

        clock_gettime(CLOCK_MONOTONIC, &end);

        double pm_mapping_time = (end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec);
        printf("pm mapping time: %.3f\tmsec\n", pm_mapping_time / 1e6);

        clock_gettime(CLOCK_MONOTONIC, &start);

        CHECK_CUDA_ERROR(cudaHostRegister(pm_base_, mapped_len, cudaHostRegisterMapped | cudaHostRegisterPortable));

        clock_gettime(CLOCK_MONOTONIC, &end);

        double gpu_mapping_time = (end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec);
        printf("gpu mapping time: %.3f\tmsec\n", gpu_mapping_time / 1e6);

        hash_table_ = new GPHash(recovery, pm_base_, level, device_idx_);

        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_ops_, max_batch_size * sizeof(uint32_t)));
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_keys_, max_batch_size * KEY_SIZE));
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_values_, max_batch_size * max(KEY_SIZE, VALUE_SIZE)));

        cudaDeviceSynchronize();
    }

    double batchedOperations(uint32_t *h_ops, uint64_t *h_keys, uint32_t offset, uint32_t num_ops)
    {
        CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));

        cudaDeviceSynchronize();

        CHECK_CUDA_ERROR(cudaMemcpy(d_ops_, h_ops + offset, sizeof(uint32_t) * num_ops, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_keys_, h_keys + offset * KEY_SIZE / 8, KEY_SIZE * num_ops, cudaMemcpyHostToDevice));
        for (int i = 0; i < max(1, VALUE_SIZE / KEY_SIZE); ++i)
        {
            CHECK_CUDA_ERROR(cudaMemcpy(d_values_ + i * num_ops * KEY_SIZE / 8, h_keys + offset * KEY_SIZE / 8, KEY_SIZE * num_ops, cudaMemcpyHostToDevice));
        }

        float temp_time = 0.0f;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        hash_table_->batchedOperations(d_ops_, d_keys_, d_values_, num_ops);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&temp_time, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        CHECK_CUDA_ERROR(cudaMemcpy(h_ops + offset, d_ops_, sizeof(uint32_t) * num_ops, cudaMemcpyDeviceToHost));

        cudaDeviceSynchronize();

        return temp_time;
    }

    void load_cache()
    {
        CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));

        uint32_t cache_size = cache_size_rate_ * capacity() / NUM_SLOT;
        cache_size = (cache_size / 128) * 128; // cache_size needs to be a multiple of 128

        printf("begin load cache %u\n", cache_size);

        cudaDeviceSynchronize();

        float temp_time = 0.0f;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        hash_table_->loadCache(cache_size);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&temp_time, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaDeviceSynchronize();

        printf("finish load cache!\n");
        printf("load cache time: %.3f\tmsec\n", temp_time);
    }

    void resize()
    {
        CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));

        printf("begin resize level %u\n", hash_table_->level_);

        cudaDeviceSynchronize();

        float temp_time = 0.0f;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        hash_table_->resize();

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&temp_time, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaDeviceSynchronize();

        printf("finish resize!\n");
        printf("resize time: %.3f\tmsec\n", temp_time);
    }

    uint64_t hitCount()
    {
        CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));

        cudaDeviceSynchronize();

        uint64_t hit_count = hash_table_->hitCount();

        cudaDeviceSynchronize();

        return hit_count;
    }

    void clearExp()
    {
        CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));

        cudaDeviceSynchronize();

        hash_table_->clearExp();

        cudaDeviceSynchronize();
    }

    void clearHit()
    {
        CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));

        cudaDeviceSynchronize();

        hash_table_->clearHit();

        cudaDeviceSynchronize();
    }

    void invalidateCache()
    {
        CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));

        cudaDeviceSynchronize();

        hash_table_->invalidateCache();

        cudaDeviceSynchronize();
    }

    uint32_t capacity()
    {
        CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));

        cudaDeviceSynchronize();

        uint32_t capacity = hash_table_->capacity();

        cudaDeviceSynchronize();

        return capacity;
    }

    double loadFactor()
    {
        CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));

        cudaDeviceSynchronize();

        double load_factor = hash_table_->loadFactor();

        cudaDeviceSynchronize();

        return load_factor;
    }

    void setValueOffset(uint64_t value_offset)
    {
        cudaDeviceSynchronize();

        hash_table_->setValueOffset(value_offset);

        cudaDeviceSynchronize();
    }
};