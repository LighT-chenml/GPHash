#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>

#include "change_ddio.h"
#include "../src/hash_table.cuh"

using namespace std;

uint32_t reload_interval = (1 << 22);
uint32_t batch_size = (1 << 12);
uint32_t ini_level = 12;
double cache_size_rate = 0.25;

int main(int argc, char *argv[])
{
    if (argc < 8)
    {
        printf("Usage:  %s  pm_file input_file num_load num_run log_batch_size ini_level cache_size_rate\n", argv[0]);
        exit(1);
    }

    char *pm_file = argv[1];

    size_t file_size = 16LL * 1024 * 1024 * 1024;

    uint32_t num_load = atoi(argv[3]);
    uint32_t num_run = atoi(argv[4]);
    batch_size = 1 << atoi(argv[5]);
    ini_level = atoi(argv[6]);
    cache_size_rate = atof(argv[7]);

    printf("batch_size = %u\n", batch_size);
    printf("KEY_SIZE = %u\n", KEY_SIZE);
    printf("ini_level = %u\n", ini_level);
    printf("cache_size_rate = %.3lf\n", cache_size_rate);

    uint32_t *load_opts = new uint32_t[num_load];
    uint64_t *load_keys = new uint64_t[num_load * (KEY_SIZE / 8)];

    uint32_t *run_opts = new uint32_t[num_run];
    uint64_t *run_keys = new uint64_t[num_run * (KEY_SIZE / 8)];

    uint32_t *warmup_opts = new uint32_t[num_run];

    string dataset(argv[2]);
    string load_file = dataset + ".load";
    string run_file = dataset + ".run";

    uint32_t insert_keys = 0;

    ifstream ifs;
    ifs.open(load_file);
    if (!ifs)
    {
        cerr << "No Load file." << endl;
        exit(1);
    }
    else
    {
        uint64_t t;
        string opt;
        for (int i = 0; i < num_load; i++)
        {
            ifs >> opt >> t;
            for (int j = 0; j < KEY_SIZE / 8; ++j)
            {
                *(load_keys + i * (KEY_SIZE / 8) + j) = t + j;
            }
            if (opt == "INSERT")
            {
                load_opts[i] = 1;
                ++insert_keys;
            }
            else
            {
                load_opts[i] = 0;
            }
        }
        ifs.close();
        cout << load_file << " is used." << endl;
    }

    ifs.open(run_file);
    if (!ifs)
    {
        cerr << "No Run file." << endl;
        exit(1);
    }
    else
    {
        uint64_t t;
        string opt;
        for (int i = 0; i < num_run; i++)
        {
            ifs >> opt >> t;
            for (int j = 0; j < KEY_SIZE / 8; ++j)
            {
                *(run_keys + i * (KEY_SIZE / 8) + j) = t + j;
            }
            if (opt == "READ")
            {
                run_opts[i] = 4;
            }
            else if (opt == "INSERT")
            {
                run_opts[i] = 1;
                ++insert_keys;
            }
            else if (opt == "UPDATE")
            {
                run_opts[i] = 2;
                ++insert_keys;
            }
            else if (opt == "DELETE")
            {
                run_opts[i] = 3;
                ++insert_keys;
            }
            else
            {
                run_opts[i] = 0;
            }
            warmup_opts[i] = run_opts[i] != 0 ? 4 : 0;
        }
        ifs.close();
        cout << run_file << " is used." << endl;
    }

#ifdef DDIO_OFF
    ddio_off();
    printf("turn off DDIO!\n");
#endif

    HashTable hash_table(pm_file, file_size, batch_size, ini_level, cache_size_rate, 0);

    cout << "Start Loading" << endl;

    double init_build_time = 0.0;
    double concurrent_time = 0.0;

    uint32_t num_keys = num_load;

    double batch_time = 0;
    double sum_batch_time = 0;
    uint32_t num_batch = 0;

    vector<double> load_factors;

    for (uint32_t offset = 0; offset < num_keys;)
    {
        batch_time += hash_table.batchedOperations(load_opts, load_keys, offset, min(batch_size, num_keys - offset));

        sum_batch_time += batch_time;
        ++num_batch;

        bool flag = false;
        for (uint32_t i = offset; i < offset + batch_size && i < num_keys; ++i)
        {
            if (load_opts[i] == 1)
            {
                flag = true;
                break;
            }
        }
        if (!flag)
        {
            offset += batch_size;
            init_build_time += batch_time;
            batch_time = 0;
        }
        else
        {
            double load_factor = hash_table.loadFactor();
            load_factors.push_back(load_factor);
            hash_table.resize();
        }
    }

    init_build_time *= 1000000;
    printf("%.3lf\tMops/sec\n", (num_keys / (init_build_time / 1000.0)));
    printf("%.3lf\tmsec\n", sum_batch_time / num_batch);
    cout << "load factor: " << hash_table.loadFactor() << endl;

    double avg_load_factor = 0;
    double max_load_factor = 0;
    for (auto load_factor: load_factors)
    {
        // cout << load_factor << endl;
        avg_load_factor += load_factor;
        max_load_factor = max(max_load_factor, load_factor);
    }
    cout << "avg load factor: " << avg_load_factor / load_factors.size() << endl;
    cout << "max load factor: " << max_load_factor << endl;

    cout << "________________________________________________Finish Loading!________________________________________________" << endl;

#ifdef ENABLE_CACHE
    hash_table.clearExp();
    hash_table.clearHit();
    hash_table.load_cache();

    cout << "begin warm up" << endl;

    double warmup_time = 0.0;

    num_keys = num_run;

    batch_time = 0;
    sum_batch_time = 0;
    num_batch = 0;

    uint32_t last_load_cache = 0;

    for (uint32_t offset = 0; offset < num_keys;)
    {
        batch_time += hash_table.batchedOperations(warmup_opts, run_keys, offset, min(batch_size, num_keys - offset));

        sum_batch_time += batch_time;
        ++num_batch;

        offset += batch_size;
        warmup_time += batch_time;
        batch_time = 0;

#ifdef ENABLE_CACHE
        if ((num_batch - last_load_cache) * batch_size >= reload_interval)
        {
            hash_table.load_cache();
            last_load_cache = num_batch;
        }
#endif
    }

    warmup_time *= 1000000;
    printf("%.3lf\tMops/sec\n", (num_keys / (warmup_time / 1000.0)));
    printf("%.3lf\tmsec\n", sum_batch_time / num_batch);
#ifdef ENABLE_CACHE
    cout << "warmup hit rate: " << 1.0 * hash_table.hitCount() / num_keys / 32 << endl;
#endif

    cout << "________________________________________________Finish Warmup!________________________________________________" << endl;
#endif

#ifdef ENABLE_CACHE
    hash_table.setValueOffset(num_load);
    hash_table.clearHit();
    hash_table.load_cache();

    last_load_cache = 0;
#endif

    cout << "Start Running" << endl;

    num_keys = num_run;

    batch_time = 0;
    sum_batch_time = 0;
    num_batch = 0;

    uint32_t unfound = 0;

    for (uint32_t offset = 0; offset < num_keys;)
    {
        batch_time += hash_table.batchedOperations(run_opts, run_keys, offset, min(batch_size, num_keys - offset));

        sum_batch_time += batch_time;
        ++num_batch;

        bool flag = false;
        for (int i = offset; i < offset + batch_size && i < num_keys; ++i)
        {
            if (run_opts[i] == 1)
            {
                flag = true;
            }
            if (run_opts[i] == 4)
            {
                ++unfound;
                run_opts[i] = 0;
            }
        }
        if (!flag)
        {
            offset += batch_size;
            concurrent_time += batch_time;
            batch_time = 0;

#ifdef ENABLE_CACHE
            if ((num_batch - last_load_cache) * batch_size >= reload_interval)
            {
                hash_table.load_cache();
                last_load_cache = num_batch;
            }
#endif
        }
        else
        {
            cout << "load factor: " << hash_table.loadFactor() << endl;
#ifdef ENABLE_CACHE
            hash_table.invalidateCache();
#endif
            hash_table.resize();
#ifdef ENABLE_CACHE
            hash_table.load_cache();
            last_load_cache = num_batch;
#endif
        }
    }

    concurrent_time *= 1000000;
    printf("%.3lf\tMops/sec\n", (num_keys / (concurrent_time / 1000.0)));
    printf("%.3lf\tmsec\n", sum_batch_time / num_batch);
    cout << "unfound: " << unfound << endl;
#ifdef ENABLE_CACHE
    cout << "hit rate: " << 1.0 * hash_table.hitCount() / num_keys / 32 << endl;
#endif

    cout << "________________________________________________Finish Test!________________________________________________" << endl;

#ifdef DDIO_OFF
    ddio_on();
    printf("turn on DDIO!\n");
#endif

    return 0;
}