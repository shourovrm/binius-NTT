// clang-format off
#include "md5.cuh"

#include <cstdint>
#include <cstdio>
#include <random>
#include <iostream>
#include <chrono>
#include <iomanip>

#include "ulvt/utils/common.cuh"
#include "ulvt/finite_fields/binary_tower.cuh"
#include "ulvt/ntt/nttconf.cuh"
#include "ulvt/ntt/additive_ntt.cuh"
#include "ulvt/ntt/modified_antt.cuh"
// clang-format on

using namespace std::chrono;

// Hash table for validation (from test_ntt.cu, log_rate=0)
const uint8_t additive_ntt_hashes[31][16] = {
{0},
{0x6c, 0x67, 0x4a, 0x56, 0x27, 0x5d, 0xfd, 0x6b, 0xaf, 0x96, 0x51, 0x63, 0xd6, 0xd4, 0x75, 0x7a},
{0x37, 0x3b, 0x75, 0x3b, 0x3e, 0x05, 0x3d, 0x12, 0x8c, 0xb5, 0x3a, 0xc2, 0x3f, 0x40, 0x3a, 0x1c},
{0x09, 0x33, 0xfa, 0x26, 0x68, 0x93, 0x78, 0x68, 0x4a, 0x4f, 0x6a, 0x46, 0x54, 0xde, 0xed, 0x44},
{0x3f, 0x8d, 0x24, 0x4d, 0xc6, 0x83, 0xe5, 0x85, 0x34, 0xc8, 0xa1, 0xbe, 0xf2, 0x28, 0x41, 0x27},
{0x2f, 0x72, 0x47, 0x0c, 0xe9, 0x05, 0xc9, 0x26, 0x13, 0x80, 0xba, 0xc9, 0x23, 0x2d, 0xb7, 0xae},
{0xa2, 0x2e, 0x4b, 0x3a, 0xe7, 0x3b, 0x2a, 0x7c, 0x44, 0x43, 0x28, 0x8e, 0x7f, 0x8f, 0xdf, 0xca},
{0x81, 0x17, 0x9f, 0x7e, 0x33, 0xb4, 0x52, 0x2b, 0x20, 0xba, 0xcb, 0xa9, 0xc0, 0x7d, 0xb9, 0xcd},
{0xfb, 0x4c, 0x30, 0x04, 0x90, 0x6e, 0xf7, 0xd5, 0x9d, 0x5c, 0x5a, 0x5a, 0x04, 0x85, 0xe2, 0x90},
{0xd0, 0x4b, 0xcc, 0xe5, 0xc7, 0xd1, 0xa8, 0x59, 0x95, 0xa9, 0xe9, 0xa6, 0x54, 0xb5, 0x83, 0x23},
{0x19, 0x1e, 0x2b, 0xc2, 0xee, 0x65, 0x53, 0x00, 0xc2, 0x7f, 0x7c, 0x24, 0x49, 0x52, 0xc0, 0xb7},
{0xe5, 0x4f, 0x05, 0x5f, 0x6b, 0xbf, 0x6c, 0x63, 0x1d, 0x8b, 0x18, 0x6f, 0x38, 0xce, 0x2d, 0x14},
{0x61, 0xfc, 0xc4, 0x3e, 0xe5, 0x2b, 0xbd, 0xb6, 0xe2, 0x7a, 0xe5, 0x85, 0x82, 0x81, 0xc9, 0xbe},
{0xbd, 0x00, 0x57, 0x75, 0x80, 0xa8, 0x55, 0xdb, 0x62, 0x50, 0x9d, 0x1b, 0x0b, 0x46, 0xab, 0x6d},
{0xd4, 0x73, 0x00, 0x90, 0xc2, 0x57, 0x3b, 0xd3, 0x9e, 0xe2, 0x69, 0x14, 0x1d, 0xc3, 0x6d, 0x44},
{0x9d, 0xdb, 0x71, 0x32, 0xcc, 0x22, 0x20, 0x81, 0xde, 0xe2, 0x6d, 0xb8, 0xb0, 0x37, 0x6d, 0x0a},
{0x66, 0xf0, 0x41, 0x10, 0xea, 0x24, 0xae, 0x09, 0xe7, 0x42, 0xbc, 0xd3, 0x61, 0x52, 0xe7, 0x80},
{0xa3, 0x21, 0x3a, 0xc4, 0x17, 0x29, 0x0e, 0x97, 0x78, 0xd1, 0xc8, 0x5e, 0x8a, 0x27, 0x4f, 0x82},
{0x4e, 0x8e, 0x21, 0x28, 0x9f, 0x55, 0xcf, 0x52, 0x84, 0x30, 0xf6, 0x8f, 0xc3, 0xc1, 0xa8, 0xe8},
{0xd3, 0x71, 0x0a, 0x4a, 0x4f, 0xca, 0x93, 0xee, 0xc2, 0xb8, 0x45, 0x94, 0x39, 0x58, 0x92, 0x2e},
{0xe6, 0xba, 0x17, 0xd4, 0x3f, 0x88, 0x62, 0x51, 0x0e, 0xdd, 0x3f, 0xec, 0x16, 0x48, 0xc7, 0xef},
{0xe0, 0x41, 0x56, 0xc7, 0xac, 0xa2, 0xd7, 0x51, 0x06, 0xa7, 0x6c, 0xae, 0x88, 0x19, 0xf5, 0x68},
{0x45, 0xbe, 0xfb, 0x3c, 0x29, 0x3f, 0x30, 0xa0, 0xbb, 0xf4, 0x04, 0x0b, 0x28, 0x69, 0xb0, 0xd8},
{0xf7, 0xc0, 0xd8, 0x9c, 0xca, 0xe0, 0x01, 0xfd, 0xd6, 0x8d, 0xa6, 0x87, 0x95, 0x4b, 0x00, 0x70},
{0x3d, 0xdc, 0xc9, 0xb4, 0x28, 0x59, 0xc8, 0xc9, 0xf9, 0xbe, 0x5c, 0x6c, 0xb5, 0xbd, 0x9e, 0xa9},
{0x2b, 0xdc, 0xa0, 0x1c, 0x18, 0xc8, 0xd6, 0x42, 0x05, 0xeb, 0x7a, 0x0c, 0xa8, 0x5e, 0x64, 0x9d},
{0xce, 0xc9, 0x31, 0xe2, 0x0b, 0x31, 0x18, 0x4b, 0x27, 0x0a, 0xe0, 0x36, 0x51, 0x18, 0x6c, 0xf8},
{0xfb, 0x8c, 0x00, 0x5b, 0x98, 0x9e, 0x3e, 0x02, 0xe1, 0xb0, 0xf1, 0xe1, 0x75, 0x91, 0x08, 0x82},
{0xda, 0x75, 0x62, 0xb4, 0x6e, 0x0b, 0x01, 0x18, 0x4f, 0x8b, 0xaf, 0x2a, 0xea, 0x57, 0x82, 0x6d},
{0xbe, 0x01, 0xe5, 0x10, 0xe4, 0xcf, 0x06, 0xb7, 0xa2, 0x64, 0x37, 0xe8, 0xc2, 0xb2, 0x8d, 0xc6},
{0xd4, 0x49, 0xc7, 0x4e, 0x93, 0x0c, 0x90, 0xa7, 0x7d, 0xc6, 0x3a, 0xd2, 0xae, 0xd2, 0xb7, 0xac},
};

// Timing result structure
struct TimingResult {
    int log_h;
    double original_ms;
    double modified_ms;
    double speedup;
    bool orig_correct;
    bool mod_correct;
};

// Run both NTT versions and return timing + correctness
TimingResult benchmark_both_antt(int log_h, int log_rate) {
    std::mt19937 gen(0xdeadbeef + log_h + log_rate);
    auto inp_size = 1 << log_h;
    auto out_size = 1 << (log_h + log_rate);
    
    // Prepare input data
    NTTData<uint32_t> ntt_inp(DataOrder::IN_ORDER, inp_size);
    for (size_t i = 0; i < inp_size; i++) {
        uint32_t r = gen();
        ntt_inp.data[i] = r;
    }
    
    TimingResult result;
    result.log_h = log_h;
    
    // ============ ORIGINAL NTT ============
    {
        AdditiveNTTConf<uint32_t, FanPaarTowerField<5>> nttconf(log_h, log_rate);
        AdditiveNTT<uint32_t, FanPaarTowerField<5>> add_ntt(nttconf);
        NTTData<uint32_t> ntt_out(out_size);
        
        // Warmup
        add_ntt.apply(ntt_inp, ntt_out);
        cudaDeviceSynchronize();
        
        // Benchmark
        auto start = high_resolution_clock::now();
        add_ntt.apply(ntt_inp, ntt_out);
        cudaDeviceSynchronize();
        auto end = high_resolution_clock::now();
        
        result.original_ms = duration_cast<milliseconds>(end - start).count();
        
        // Verify correctness
        MD5Context md5;
        md5Init(&md5);
        for (size_t i = 0; i < ntt_out.size; i++) {
            uint32_t d = ntt_out.data[i];
            md5Update(&md5, (uint8_t*)&d, 4);
        }
        md5Finalize(&md5);
        result.orig_correct = (memcmp(md5.digest, additive_ntt_hashes[log_h], 16) == 0);
    }
    
    // ============ MODIFIED NTT ============
    {
        ModifiedAdditiveNTTConf<uint32_t, FanPaarTowerField<5>> nttconf(log_h, log_rate);
        ModifiedAdditiveNTT<uint32_t, FanPaarTowerField<5>> mod_ntt(nttconf);
        NTTData<uint32_t> ntt_out(out_size);
        
        // Warmup
        mod_ntt.apply(ntt_inp, ntt_out);
        cudaDeviceSynchronize();
        
        // Benchmark
        auto start = high_resolution_clock::now();
        mod_ntt.apply(ntt_inp, ntt_out);
        cudaDeviceSynchronize();
        auto end = high_resolution_clock::now();
        
        result.modified_ms = duration_cast<milliseconds>(end - start).count();
        
        // Verify correctness
        MD5Context md5;
        md5Init(&md5);
        for (size_t i = 0; i < ntt_out.size; i++) {
            uint32_t d = ntt_out.data[i];
            md5Update(&md5, (uint8_t*)&d, 4);
        }
        md5Finalize(&md5);
        result.mod_correct = (memcmp(md5.digest, additive_ntt_hashes[log_h], 16) == 0);
    }
    
    result.speedup = result.original_ms / result.modified_ms;
    return result;
}

void print_header() {
    std::cout << "\n";
    std::cout << "====================================================================================\n";
    std::cout << "                  ADDITIVE NTT BENCHMARK: Original vs Modified (r=0)\n";
    std::cout << "====================================================================================\n";
    std::cout << std::left 
              << std::setw(8) << "log_h"
              << std::setw(15) << "Original (ms)"
              << std::setw(15) << "Modified (ms)"
              << std::setw(12) << "Speedup"
              << std::setw(12) << "Orig.Valid"
              << std::setw(12) << "Mod.Valid"
              << "\n";
    std::cout << "------------------------------------------------------------------------------------\n";
}

void print_result(const TimingResult& res) {
    std::cout << std::left 
              << std::setw(8) << res.log_h
              << std::fixed << std::setprecision(2)
              << std::setw(15) << res.original_ms
              << std::setw(15) << res.modified_ms
              << std::setprecision(3)
              << std::setw(12) << res.speedup << "x"
              << std::setw(12) << (res.orig_correct ? "✓ PASS" : "✗ FAIL")
              << std::setw(12) << (res.mod_correct ? "✓ PASS" : "✗ FAIL")
              << "\n";
}

void print_summary(const std::vector<TimingResult>& results) {
    double total_orig = 0, total_mod = 0;
    int orig_pass = 0, mod_pass = 0;
    
    for (const auto& res : results) {
        total_orig += res.original_ms;
        total_mod += res.modified_ms;
        if (res.orig_correct) orig_pass++;
        if (res.mod_correct) mod_pass++;
    }
    
    double avg_speedup = total_orig / total_mod;
    
    std::cout << "====================================================================================\n";
    std::cout << "SUMMARY:\n";
    std::cout << "  Original tests passed: " << orig_pass << "/" << results.size() << "\n";
    std::cout << "  Modified tests passed: " << mod_pass << "/" << results.size() << "\n";
    std::cout << "  Total original time: " << std::fixed << std::setprecision(2) << total_orig << " ms\n";
    std::cout << "  Total modified time: " << std::fixed << std::setprecision(2) << total_mod << " ms\n";
    std::cout << "  Average speedup: " << std::fixed << std::setprecision(3) << avg_speedup << "x\n";
    std::cout << "  Performance improvement: " << std::fixed << std::setprecision(1) 
              << ((avg_speedup - 1.0) * 100.0) << "%\n";
    
    if (avg_speedup > 1.0) {
        std::cout << "  ✓ OPTIMIZATION SUCCESSFUL!\n";
    } else if (avg_speedup < 1.0) {
        std::cout << "  ✗ WARNING: Modified version is SLOWER\n";
    } else {
        std::cout << "  - No significant change\n";
    }
    std::cout << "====================================================================================\n\n";
}

int main() {
    printf("\nStarting Additive NTT Benchmark (log_rate=0)...\n");
    printf("GPU capability check: %s\n\n", check_gpu_capabilities() ? "PASS" : "FAIL");
    
    if (!check_gpu_capabilities()) {
        printf("ERROR: GPU does not meet requirements\n");
        return 1;
    }
    
    // Test range: log_h from 1 to 28 (same as test_ntt.cu)
    constexpr int LOG_H_START = 1;
    constexpr int LOG_H_END = 28;
    constexpr int LOG_RATE = 0;
    
    std::vector<TimingResult> results;
    print_header();
    
    for (int log_h = LOG_H_START; log_h <= LOG_H_END; log_h++) {
        TimingResult res = benchmark_both_antt(log_h, LOG_RATE);
        results.push_back(res);
        print_result(res);
        
        if (!res.orig_correct) {
            printf("  ERROR: Original NTT output is incorrect!\n");
        }
        if (!res.mod_correct) {
            printf("  ERROR: Modified NTT output is incorrect!\n");
        }
    }
    
    print_summary(results);
    
    return 0;
}
