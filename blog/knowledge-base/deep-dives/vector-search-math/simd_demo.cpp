/*
 * SIMD Floating-Point Non-Associativity Demo
 * 
 * To compile and run on an x86/amd64 machine:
 *     g++ -O3 -msse3 simd_demo.cpp -o simd_demo
 *     ./simd_demo
 * 
 * Note: Ensure you use a capital 'O' in -O3 (optimization level 3), not a zero.
 */

#include <cmath>
#include <iomanip>
#include <immintrin.h> // SSE intrinsics
#include <iostream>

// This function intentionally accepts exactly the five-value teaching payload
// used in main. It is not a general-purpose array reduction.
float grouped_simd_sum(const float (&values)[5]) {
    // Load the four trailing 1.0 values into a 128-bit SSE register.
    __m128 hsum = _mm_loadu_ps(&values[1]);

    // _mm_hadd_ps performs pairwise horizontal additions. The two calls group
    // the four small values before adding their total to the large value.
    hsum = _mm_hadd_ps(hsum, hsum);
    hsum = _mm_hadd_ps(hsum, hsum);

    const float tail_sum = _mm_cvtss_f32(hsum);
    return values[0] + tail_sum;
}

int main() {
    // A normal float32 value has 24 bits of significand precision, including
    // its implicit leading bit. Around 2^24, adjacent representable values are
    // two units apart, so adding 1.0 lands on a rounding tie.
    alignas(16) const float values[5] = {
        16777216.0f, 1.0f, 1.0f, 1.0f, 1.0f
    };
    
    // Left-to-right accumulation rounds after every addition.
    float seq_sum = 0.0f;
    for (float value : values) {
        seq_sum += value;
    }
    
    // The grouped reduction first obtains exactly 4.0 from the small values.
    const float grouped_sum = grouped_simd_sum(values);
    
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Left-to-right sum: " << seq_sum << '\n';
    std::cout << "Grouped SIMD sum:  " << grouped_sum << '\n';
    std::cout << "Difference:        " << std::fabs(seq_sum - grouped_sum) << '\n';
    std::cout << "Both are float32 results from different operation orders.\n";
    return 0;
}
