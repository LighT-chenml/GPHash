#pragma once

#include <algorithm>
#include <vector>
#include <thread>

#define CHECK_CUDA_ERROR(call)                                                                \
	do                                                                                        \
	{                                                                                         \
		cudaError_t err = call;                                                               \
		if (err != cudaSuccess)                                                               \
		{                                                                                     \
			printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
			exit(EXIT_FAILURE);                                                               \
		}                                                                                     \
	} while (0)

#define CLOCK_RATE 1380000			  /* modify for different device */
__device__ void deviceSleep(double t) // sleep for t sec
{
	clock_t t0 = clock64();
	clock_t t1 = t0;
	while ((t1 - t0) < t * CLOCK_RATE)
		t1 = clock64();
}

#ifndef BYTE_8
static constexpr uint32_t KEY_SIZE = 32;
#else
static constexpr uint32_t KEY_SIZE = 8;
#endif

static constexpr uint32_t VALUE_SIZE = 32;
static constexpr uint32_t FP_SIZE = 8;

static constexpr uint32_t MAX_LEVEL = 28;
static constexpr uint32_t NUM_LEVEL = 4;
static constexpr uint32_t GROUP_SIZE = 2; // the number of hash functions
static constexpr uint32_t NUM_SLOT = 4;
static constexpr uint32_t NUM_ACCESS_BUCKET = NUM_LEVEL * GROUP_SIZE;
#ifdef INPLACE_KEY
static constexpr uint32_t SLOT_SIZE = FP_SIZE + KEY_SIZE + 8;
#else
static constexpr uint32_t SLOT_SIZE = 8;
#endif
static constexpr uint32_t BUCKET_SIZE = NUM_SLOT * SLOT_SIZE;
static constexpr uint32_t NUM_SLOT_PER_GROUP = GROUP_SIZE * NUM_SLOT;

static constexpr uint64_t NUM_VALUE = (1LL << 24) + (1 << 26);

// the values larger than mod can be used
static constexpr uint64_t EMPTY_KEY_64 = 0xFFFFFFFFFFFFFFFFLL;
static constexpr uint64_t INSERTING_64 = 0xFFFFFFFFFFFFFFFELL;

static constexpr uint32_t NOT_CACHED = 0xFFFFFFFF;

class GPHashContext
{
public:
	uint64_t hash_x_[4] = {29, 20000623, 401, 499};
	uint64_t hash_y_[4] = {71, 277, 467, 461};
	uint64_t hash_mod_[4] = {1000000007, 998244353, 1004535809, 469762049};

	uint64_t *values_;

	uint32_t level_; // bottom level

	uint32_t cache_size_;
	uint64_t *cache_ptr_;
	uint32_t *cache_ref_;
	uint32_t *cached_bucket_level_;
	uint32_t *cached_bucket_id_;

	uint64_t **level_ptr_;
	uint32_t **bucket_cnt_;
	uint32_t **bucket_pos_;
	uint32_t **bucket_hit_;
	uint32_t **bucket_exp_;
	uint32_t **bucket_version_;

#pragma hd_warning_disable
	__host__ __device__ GPHashContext()
		: level_(0)
	{
	}

#pragma hd_warning_disable
	__host__ __device__ GPHashContext(GPHashContext &t)
	{
		values_ = t.values_;
		level_ = t.level_;

		cache_size_ = t.cache_size_;
		cache_ptr_ = t.cache_ptr_;
		cache_ref_ = t.cache_ref_;
		cached_bucket_level_ = t.cached_bucket_level_;
		cached_bucket_id_ = t.cached_bucket_id_;

		level_ptr_ = t.level_ptr_;
		bucket_cnt_ = t.bucket_cnt_;
		bucket_pos_ = t.bucket_pos_;
		bucket_hit_ = t.bucket_hit_;
		bucket_exp_ = t.bucket_exp_;
		bucket_version_ = t.bucket_version_;
	}

	__host__ void initParameters(uint64_t *values, uint32_t level, uint64_t **level_ptr,
								 uint32_t cache_size, uint64_t *cache_ptr, uint32_t *cache_ref, uint32_t *cached_bucket_level, uint32_t *cached_bucket_id,
								 uint32_t **bucket_cnt, uint32_t **bucket_pos, uint32_t **bucket_hit, uint32_t **bucket_exp, uint32_t **bucket_version)
	{
		values_ = values;
		level_ = level;

		cache_size_ = cache_size;
		cache_ptr_ = cache_ptr;
		cache_ref_ = cache_ref;
		cached_bucket_level_ = cached_bucket_level;
		cached_bucket_id_ = cached_bucket_id;

		level_ptr_ = level_ptr;
		bucket_cnt_ = bucket_cnt;
		bucket_pos_ = bucket_pos;
		bucket_hit_ = bucket_hit;
		bucket_exp_ = bucket_exp;
		bucket_version_ = bucket_version;
	}

	__device__ __host__ __forceinline__ uint32_t computeBucket(uint64_t hash_value, uint32_t level)
	{
		return hash_value & ((1 << level) - 1);
	}

	__device__ __host__ __forceinline__ uint64_t calHash(uint64_t *key, uint32_t hash_id)
	{
		uint64_t hash_value = 0;
		for (int i = 0; i < KEY_SIZE / 8; ++i)
		{
			hash_value = (hash_value * hash_x_[hash_id] + *(key + i) % hash_mod_[hash_id] + hash_y_[hash_id]) % hash_mod_[hash_id];
		}
		return hash_value;
	}

	__device__ __host__ __forceinline__ uint32_t calHash8Byte(uint64_t key, uint32_t hash_id)
	{
		return (key % hash_mod_[hash_id] * hash_x_[hash_id] + hash_y_[hash_id]) % hash_mod_[hash_id];
	}

	__device__ __forceinline__ bool compareKey(uint64_t fp1, uint64_t *key1, uint64_t fp2, uint64_t *key2) // assume fp2 is true fp
	{
#ifndef INPLACE_KEY
		if (fp1 == EMPTY_KEY_64) return false;
		fp1 >>= 48;
		fp2 >>= 48;
#endif
		if (fp1 != fp2)
		{
			return false;
		}
		for (int i = 0; i < KEY_SIZE / 8; ++i)
		{
			if (*(key1 + i) != *(key2 + i))
				return false;
		}
		return true;
	}

	__device__ __host__ __forceinline__ void setKey(uint64_t *key1, uint64_t *key2)
	{
		for (int i = 0; i < KEY_SIZE / 8; ++i)
			*(key1 + i) = *(key2 + i);
	}

	__device__ __host__ __forceinline__ void setValue(uint64_t *value1, uint64_t *value2)
	{
		for (int i = 0; i < VALUE_SIZE / 8; ++i)
			*(value1 + i) = *(value2 + i);
	}

	__device__ __forceinline__ uint32_t getCachedBucketRef(uint32_t bucket)
	{
		return *(cache_ref_ + bucket);
	}

	__device__ __forceinline__ void addCachedBucketRef(uint32_t bucket)
	{
		atomicAdd(cache_ref_ + bucket, 1);
	}

	__device__ __forceinline__ void decCachedBucketRef(uint32_t bucket)
	{
		atomicDec(cache_ref_ + bucket, 1);
	}

	__device__ __forceinline__ uint64_t *getFPPtrFromCache(uint32_t bucket, uint32_t slot)
	{
		return (uint64_t *)((uint64_t)cache_ptr_ + (bucket * NUM_SLOT + slot) * FP_SIZE);
	}

	__device__ __forceinline__ uint64_t *getKeyPtrFromCache(uint32_t bucket, uint32_t slot)
	{
		return (uint64_t *)((uint64_t)cache_ptr_ + cache_size_ * NUM_SLOT * FP_SIZE + (bucket * NUM_SLOT + slot) * KEY_SIZE);
	}

	__device__ __forceinline__ uint64_t *getValueAddressPtrFromCache(uint32_t bucket, uint32_t slot)
	{
		return (uint64_t *)((uint64_t)cache_ptr_ + cache_size_ * NUM_SLOT * (FP_SIZE + KEY_SIZE) + (bucket * NUM_SLOT + slot) * 8);
	}

	__device__ __host__ __forceinline__ uint64_t *getFPPtr(uint32_t level, uint32_t bucket, uint32_t slot)
	{
#ifdef INPLACE_KEY
		return (uint64_t *)((uint64_t)(level_ptr_[level]) + (bucket * NUM_SLOT + slot) * FP_SIZE);
#else
		return getValueAddressPtr(level, bucket, slot);
#endif
	}

	__device__ __host__ __forceinline__ uint64_t *getKeyPtr(uint32_t level, uint32_t bucket, uint32_t slot)
	{
#ifdef INPLACE_KEY
		return (uint64_t *)((uint64_t)(level_ptr_[level]) + (1 << level) * NUM_SLOT * FP_SIZE + (bucket * NUM_SLOT + slot) * KEY_SIZE);
#else
		uint64_t value_address = *getValueAddressPtr(level, bucket, slot);
		value_address <<= 16;
		value_address >>= 16;
		if (value_address >= NUM_VALUE)
		{
			return nullptr;
		}
		else
			return values_ + (value_address * (VALUE_SIZE + KEY_SIZE) + VALUE_SIZE) / 8;
#endif
	}

	__device__ __host__ __forceinline__ uint64_t *getValueAddressPtr(uint32_t level, uint32_t bucket, uint32_t slot)
	{
#ifdef INPLACE_KEY
		return (uint64_t *)((uint64_t)(level_ptr_[level]) + (1 << level) * NUM_SLOT * (FP_SIZE + KEY_SIZE) + (bucket * NUM_SLOT + slot) * 8);
#else
		return (uint64_t *)((uint64_t)(level_ptr_[level]) + (bucket * NUM_SLOT + slot) * 8);
#endif
	}

	__device__ __forceinline__ void addBucketExp(uint32_t level, uint32_t bucket, uint32_t hit_bitmap)
	{
		uint32_t exp = 1; // LFU
		// uint32_t exp = clock(); // LRU

		atomicAdd(bucket_exp_[level] + bucket, exp); // LFU
		// atomicMax(bucket_exp_[level] + bucket, exp); // LRU
	}

	__device__ __forceinline__ void addBucketHit(uint32_t level, uint32_t bucket)
	{
		atomicAdd(bucket_hit_[level] + bucket, 1);
	}

	__device__ __forceinline__ void addBucketCnt(uint32_t level, uint32_t bucket)
	{
		atomicAdd(bucket_cnt_[level] + bucket, 1);
	}

	__device__ __forceinline__ void decBucketCnt(uint32_t level, uint32_t bucket)
	{
		atomicDec(bucket_cnt_[level] + bucket, 1);
	}

	__device__ __forceinline__ void addBucketVersion(uint32_t level, uint32_t bucket)
	{
		atomicAdd(bucket_version_[level] + bucket, 1);
	}

	__device__ __host__ __forceinline__ uint32_t getBucketCnt(uint32_t level, uint32_t bucket)
	{
		return *(bucket_cnt_[level] + bucket);
	}

	__device__ __host__ __forceinline__ uint32_t getBucketVersion(uint32_t level, uint32_t bucket)
	{
		return *(bucket_version_[level] + bucket);
	}

	__device__ __forceinline__ bool insert(bool &ongoing,
										   uint32_t lane_id,
										   uint64_t *my_key,
										   uint64_t my_value_address,
										   uint32_t level);
	__device__ __forceinline__ bool insert8Byte(bool &ongoing,
												uint32_t lane_id,
												uint64_t my_key,
												uint64_t my_value_address,
												uint32_t level);
	__device__ __forceinline__ bool insertPerThread(bool &ongoing,
													uint32_t lane_id,
													uint64_t *my_key,
													uint64_t my_value_address,
													uint32_t level);
	__device__ __forceinline__ bool update(bool &ongoing,
										   uint32_t lane_id,
										   uint64_t *my_key,
										   uint64_t my_value_address,
										   uint32_t level);
	__device__ __forceinline__ bool update8Byte(bool &ongoing,
												uint32_t lane_id,
												uint64_t my_key,
												uint64_t my_value_address,
												uint32_t level);
	__device__ __forceinline__ bool updatePerThread(bool &ongoing,
													uint32_t lane_id,
													uint64_t *my_key,
													uint64_t my_value_address,
													uint32_t level);
	__device__ __forceinline__ bool del(bool &ongoing,
										uint32_t lane_id,
										uint64_t *my_key,
										uint32_t level);
	__device__ __forceinline__ bool del8Byte(bool &ongoing,
											 uint32_t lane_id,
											 uint64_t my_key,
											 uint32_t level);
	__device__ __forceinline__ bool delPerThread(bool &ongoing,
												 uint32_t lane_id,
												 uint64_t *my_key,
												 uint32_t level);
	__device__ __forceinline__ bool search(bool &ongoing,
										   uint32_t lane_id,
										   uint64_t *my_key,
										   uint64_t *my_value,
										   uint32_t level);
	__device__ __forceinline__ bool search8Byte(bool &ongoing,
												uint32_t lane_id,
												uint64_t my_key,
												uint64_t *my_value,
												uint32_t level);
	__device__ __forceinline__ bool searchPerThread(bool &ongoing,
													uint32_t lane_id,
													uint64_t *my_key,
													uint64_t *my_value,
													uint32_t level);
	__device__ __forceinline__ void rehash(bool &ongoing,
										   uint32_t lane_id,
										   uint32_t level,
										   uint32_t bucket_id);
	__device__ __forceinline__ void rehash8Byte(bool &ongoing,
												uint32_t lane_id,
												uint32_t level,
												uint32_t bucket_id);
	__device__ __forceinline__ void checkBucket(bool &ongoing,
												uint32_t lane_id,
												uint32_t level,
												uint32_t bucket_id);
	__device__ __forceinline__ void fetchBucket(bool &ongoing,
												uint32_t cached_pos,
												uint32_t level_id,
												uint32_t bucket_id);
};

class GPHash
{
public:
	// fixed parameters:
	static constexpr uint32_t BLOCKSIZE_ = 128;

	uint32_t device_idx_;

	GPHashContext GPHash_ctx_;
	void *pm_base_;
	void *pm_ptr_;
	uint64_t value_offset_;
	uint64_t *values_;
	uint32_t level_;
	uint64_t *level_ptr_[MAX_LEVEL];

	uint32_t cache_size_ = 0;
	uint64_t *cache_ptr_ = nullptr;
	uint32_t *cache_ref_ = nullptr;
	uint32_t *cached_bucket_level_ = nullptr;
	uint32_t *cached_bucket_id_ = nullptr;

	uint32_t *bucket_cnt_[MAX_LEVEL];
	uint32_t *bucket_pos_[MAX_LEVEL];
	uint32_t *bucket_hit_[MAX_LEVEL];
	uint32_t *bucket_exp_[MAX_LEVEL];
	uint32_t *bucket_version_[MAX_LEVEL];

	uint64_t **d_level_ptr_;
	uint32_t **d_bucket_cnt_;
	uint32_t **d_bucket_pos_;
	uint32_t **d_bucket_hit_;
	uint32_t **d_bucket_exp_;
	uint32_t **d_bucket_version_;

	uint32_t *h_exp = nullptr;
	uint32_t *h_cached_bucket_level = nullptr;
	uint32_t *h_cached_bucket_id = nullptr;
	uint32_t *d_cached_bucket_level_ = nullptr;
	uint32_t *d_cached_bucket_id_ = nullptr;

	bool is_constructing_cache = false;

	GPHash(bool recovery, void *pm_base, uint32_t &level, uint32_t device_idx)
	{
		device_idx_ = device_idx;

		uint64_t is_resizing = 0; // if resizing, level is temp_level's level

		pm_base_ = pm_base;

		CHECK_CUDA_ERROR(cudaMalloc((void **)&d_level_ptr_, MAX_LEVEL * sizeof(uint64_t *)));
		CHECK_CUDA_ERROR(cudaMalloc((void **)&d_bucket_cnt_, MAX_LEVEL * sizeof(uint32_t *)));
		CHECK_CUDA_ERROR(cudaMalloc((void **)&d_bucket_pos_, MAX_LEVEL * sizeof(uint32_t *)));
		CHECK_CUDA_ERROR(cudaMalloc((void **)&d_bucket_hit_, MAX_LEVEL * sizeof(uint32_t *)));
		CHECK_CUDA_ERROR(cudaMalloc((void **)&d_bucket_exp_, MAX_LEVEL * sizeof(uint32_t *)));
		CHECK_CUDA_ERROR(cudaMalloc((void **)&d_bucket_version_, MAX_LEVEL * sizeof(uint32_t *)));

		if (recovery)
		{
			pm_ptr_ = (void *)(*((uint64_t *)pm_base_));
			is_resizing = *((uint64_t *)pm_base_ + 1);
			level_ = (uint32_t)(*((uint64_t *)pm_base_ + 2));
			value_offset_ = *((uint64_t *)pm_base_ + 3);
			for (int i = 0; i < MAX_LEVEL; ++i)
			{
				level_ptr_[i] = (uint64_t *)(*((uint64_t *)pm_base_ + 4 + i));
				bucket_cnt_[i] = nullptr;
				bucket_pos_[i] = nullptr;
				bucket_hit_[i] = nullptr;
				bucket_exp_[i] = nullptr;
				bucket_version_[i] = nullptr;
			}
			values_ = (uint64_t *)((uint64_t)pm_base_ + (4 + MAX_LEVEL) * 8);
		}
		else
		{
			pm_ptr_ = pm_base_;
			level_ = level;
			pm_ptr_ = (void *)((uint64_t)pm_ptr_ + (4 + MAX_LEVEL) * 8);
			value_offset_ = 0;
			for (int i = 0; i < MAX_LEVEL; ++i)
			{
				level_ptr_[i] = nullptr;
				bucket_cnt_[i] = nullptr;
				bucket_pos_[i] = nullptr;
				bucket_hit_[i] = nullptr;
				bucket_exp_[i] = nullptr;
				bucket_version_[i] = nullptr;
			}
			for (int i = level_; i < level_ + NUM_LEVEL; ++i)
			{
				memset(pm_ptr_, 0xFF, (1LL << i) * BUCKET_SIZE);
				level_ptr_[i] = (uint64_t *)pm_ptr_;
				pm_ptr_ = (void *)((uint64_t)pm_ptr_ + (1LL << i) * BUCKET_SIZE);
			}
#ifdef INPLACE_KEY
			memset(pm_ptr_, 0x00, NUM_VALUE * VALUE_SIZE);
			values_ = (uint64_t *)pm_ptr_;
			pm_ptr_ = (void *)((uint64_t)pm_ptr_ + NUM_VALUE * VALUE_SIZE);
#else
			memset(pm_ptr_, 0x00, NUM_VALUE * (VALUE_SIZE + KEY_SIZE));
			values_ = (uint64_t *)pm_ptr_;
			pm_ptr_ = (void *)((uint64_t)pm_ptr_ + NUM_VALUE * (VALUE_SIZE + KEY_SIZE));
#endif

			*((uint64_t *)pm_base_) = (uint64_t)pm_ptr_;
			*((uint64_t *)pm_base_ + 1) = (uint64_t)0;
			*((uint64_t *)pm_base_ + 2) = (uint64_t)level;
			*((uint64_t *)pm_base_ + 3) = (uint64_t)value_offset_;
			for (int i = 0; i < MAX_LEVEL; ++i)
			{
				*((uint64_t *)pm_base_ + 4 + i) = (uint64_t)level_ptr_[i];
			}
			fence();
		}

		for (int i = level_; i < level_ + NUM_LEVEL; ++i)
		{
			CHECK_CUDA_ERROR(cudaMalloc((void **)&bucket_cnt_[i], (1LL << i) * sizeof(uint32_t)));
			CHECK_CUDA_ERROR(cudaMemset(bucket_cnt_[i], 0x00, (1LL << i) * sizeof(uint32_t)));

			CHECK_CUDA_ERROR(cudaMalloc((void **)&bucket_pos_[i], (1LL << i) * sizeof(uint32_t)));
			CHECK_CUDA_ERROR(cudaMemset(bucket_pos_[i], 0xFF, (1LL << i) * sizeof(uint32_t)));

			CHECK_CUDA_ERROR(cudaMalloc((void **)&bucket_hit_[i], (1LL << i) * sizeof(uint32_t)));
			CHECK_CUDA_ERROR(cudaMemset(bucket_hit_[i], 0x00, (1LL << i) * sizeof(uint32_t)));

			CHECK_CUDA_ERROR(cudaMalloc((void **)&bucket_exp_[i], (1LL << i) * sizeof(uint32_t)));
			CHECK_CUDA_ERROR(cudaMemset(bucket_exp_[i], 0x00, (1LL << i) * sizeof(uint32_t)));

			CHECK_CUDA_ERROR(cudaMalloc((void **)&bucket_version_[i], (1LL << i) * sizeof(uint32_t)));
			CHECK_CUDA_ERROR(cudaMemset(bucket_version_[i], 0x00, (1LL << i) * sizeof(uint32_t)));
		}

		CHECK_CUDA_ERROR(cudaMemcpy(d_level_ptr_, level_ptr_, MAX_LEVEL * sizeof(uint64_t *), cudaMemcpyHostToDevice));
		CHECK_CUDA_ERROR(cudaMemcpy(d_bucket_cnt_, bucket_cnt_, MAX_LEVEL * sizeof(uint32_t *), cudaMemcpyHostToDevice));
		CHECK_CUDA_ERROR(cudaMemcpy(d_bucket_pos_, bucket_pos_, MAX_LEVEL * sizeof(uint32_t *), cudaMemcpyHostToDevice));
		CHECK_CUDA_ERROR(cudaMemcpy(d_bucket_hit_, bucket_hit_, MAX_LEVEL * sizeof(uint32_t *), cudaMemcpyHostToDevice));
		CHECK_CUDA_ERROR(cudaMemcpy(d_bucket_exp_, bucket_exp_, MAX_LEVEL * sizeof(uint32_t *), cudaMemcpyHostToDevice));
		CHECK_CUDA_ERROR(cudaMemcpy(d_bucket_version_, bucket_version_, MAX_LEVEL * sizeof(uint32_t *), cudaMemcpyHostToDevice));

		h_exp = new uint32_t[1 << (level_ + NUM_LEVEL - 1)];

		GPHash_ctx_.initParameters(values_, level_, d_level_ptr_,
								   0, cache_ptr_, cache_ref_, cached_bucket_level_, cached_bucket_id_,
								   d_bucket_cnt_, d_bucket_pos_, d_bucket_hit_, d_bucket_exp_, d_bucket_version_);

		printf("finish init parameters!\n");

		if (recovery)
		{
			struct timespec start, end;
			clock_gettime(CLOCK_MONOTONIC, &start);

			for (int i = level_; i < level_ + NUM_LEVEL + is_resizing; ++i)
				checkLevel(i);

			clock_gettime(CLOCK_MONOTONIC, &end);

			double check_time = (end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec);
			printf("check time: %.3f\tmsec\n", check_time / 1e6);
			printf("finish check!\n");
		}

		if (is_resizing)
		{
			printf("continue resize level %u\n", level_);
			resize();
			printf("finish resize!\n");
		}
	}

	void clearHit()
	{
		for (int i = 0; i < MAX_LEVEL; ++i)
			if (bucket_hit_[i] != nullptr)
			{
				CHECK_CUDA_ERROR(cudaMemset(bucket_hit_[i], 0x00, (1LL << i) * sizeof(uint32_t)));
			}
	}

	void clearExp()
	{
		for (int i = 0; i < MAX_LEVEL; ++i)
			if (bucket_exp_[i] != nullptr)
			{
				CHECK_CUDA_ERROR(cudaMemset(bucket_exp_[i], 0x00, (1LL << i) * sizeof(uint32_t)));
			}
	}

	void invalidateCache()
	{
		for (int i = level_; i < level_ + NUM_LEVEL; ++i)
			if (bucket_pos_[i] != nullptr)
			{
				CHECK_CUDA_ERROR(cudaMemset(bucket_pos_[i], 0xFF, (1LL << i) * sizeof(uint32_t)));
			}
	}

	void setValueOffset(uint64_t value_offset)
	{
		value_offset_ = value_offset;
		*((uint64_t *)pm_base_ + 3) = value_offset_;
		fence();
	}

	void batchedOperations(uint32_t *d_ops, uint64_t *d_keys, uint64_t *d_values, uint32_t num_ops);
	void checkLevel(uint32_t level);
	void resize();
	void loadCache(uint32_t cache_size);
	void constructCache(uint32_t cache_size);
	void fetchBuckets();
	uint32_t capacity();
	double loadFactor();
	uint64_t hitCount();
};