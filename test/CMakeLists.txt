# Enable the following compilation flags in order
# -DWARP_COOPERATE
# -DINPLACE_KEY
# -DENABLE_CACHE
# -DBYTE_8
# -DCOUNT_HIT
# -DASYNC_LOADING

# Select one caching policy
# -DRANDOM_CACHE
# -DLFU_CACHE
# -DLRU_CACHE

cuda_add_executable(ycsb_bench ycsb_bench.cu OPTIONS ${GENCODE} ${CUDA_VERBOSE_PTXAS} -DWARP_COOPERATE -DINPLACE_KEY -DENABLE_CACHE -DCOUNT_HIT -DASYNC_LOADING -DRANDOM_CACHE)
target_link_libraries(ycsb_bench -lpmemobj -lpmem -lpthread -lpci)