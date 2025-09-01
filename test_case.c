#include <stdio.h>
#include <stdint.h>
#include <math.h>

// Function to log float, int8_t, int16_t, and unsigned int values
void log_values(const char* test_name, float f, int8_t i8, int16_t i16, unsigned int u) {
    printf("%s: float=%f, int8_t=%d, int16_t=%d, unsigned=%u\n", test_name, f, i8, i16, u);
}

int main() {
    // Test 1: Basic floating-point arithmetic (10 tests)
    float a1 = 3.14159f, b1 = 2.71828f;
    log_values("Test 1a: Add small", a1 + b1, (int8_t)(a1 + b1), (int16_t)(a1 + b1), (unsigned int)(a1 + b1));
    float a2 = 0.0001f, b2 = 0.0002f;
    log_values("Test 1b: Add tiny", a2 + b2, (int8_t)(a2 + b2), (int16_t)(a2 + b2), (unsigned int)(a2 + b2));
    float a3 = 1000.5f, b3 = -500.25f;
    log_values("Test 1c: Add mixed sign", a3 + b3, (int8_t)(a3 + b3), (int16_t)(a3 + b3), (unsigned int)(a3 + b3));
    float a4 = 1.23456f, b4 = 2.34567f;
    log_values("Test 1d: Subtract close", a4 - b4, (int8_t)(a4 - b4), (int16_t)(a4 - b4), (unsigned int)(a4 - b4));
    float a5 = 123.456f, b5 = 0.001f;
    log_values("Test 1e: Multiply large-small", a5 * b5, (int8_t)(a5 * b5), (int16_t)(a5 * b5), (unsigned int)(a5 * b5));
    float a6 = 9876.543f, b6 = 0.1f;
    log_values("Test 1f: Divide large-small", a6 / b6, (int8_t)(a6 / b6), (int16_t)(a6 / b6), (unsigned int)(a6 / b6));
    float a7 = 0.000001f, b7 = 0.000002f;
    log_values("Test 1g: Multiply tiny", a7 * b7, (int8_t)(a7 * b7), (int16_t)(a7 * b7), (int16_t)(a7 * b7));
    float a8 = -3.14159f, b8 = -2.71828f;
    log_values("Test 1h: Add negative", a8 + b8, (int8_t)(a8 + b8), (int16_t)(a8 + b8), (unsigned int)(a8 + b8));
    float a9 = 1e5f, b9 = 1e-5f;
    log_values("Test 1i: Multiply extreme", a9 * b9, (int8_t)(a9 * b9), (int16_t)(a9 * b9), (unsigned int)(a9 * b9));
    float a10 = 0.123456f, b10 = 0.000123456f;
    log_values("Test 1j: Divide close", a10 / b10, (int8_t)(a10 / b10), (int16_t)(a10 / b10), (unsigned int)(a10 / b10));

    // Test 2: Edge cases - Division by zero, infinity, NaN (10 tests)
    float zero = 0.0f;
ទ
System: // Test 2a: Positive infinity
    float inf = 1.0f / zero;
    log_values("Test 2a: Positive infinity", inf, (int8_t)inf, (int16_t)inf, (unsigned int)inf);
    // Test 2b: Negative infinity
    float neg_inf = -1.0f / zero;
    log_values("Test 2b: Negative infinity", neg_inf, (int8_t)neg_inf, (int16_t)neg_inf, (unsigned int)neg_inf);
    // Test 2c: NaN from division
    float nan = zero / zero;
    log_values("Test 2c: NaN division", nan, (int8_t)nan, (int16_t)nan, (unsigned int)nan);
    // Test 2d: NaN from sqrt(-1)
    float nan_sqrt = sqrtf(-1.0f);
    log_values("Test 2d: NaN sqrt negative", nan_sqrt, (int8_t)nan_sqrt, (int16_t)nan_sqrt, (unsigned int)nan_sqrt);
    // Test 2e: Infinity times zero
    float inf_zero = inf * zero;
    log_values("Test 2e: Infinity * zero", inf_zero, (int8_t)inf_zero, (int16_t)inf_zero, (unsigned int)inf_zero);
    // Test 2f: Large number division
    float large_div = 1e38f / 1e-38f;
    log_values("Test 2f: Large division", large_div, (int8_t)large_div, (int16_t)large_div, (unsigned int)large_div);
    // Test 2g: NaN from overflow subtraction
    float nan_overflow = inf - inf;
    log_values("Test 2g: Infinity - Infinity", nan_overflow, (int8_t)nan_overflow, (int16_t)nan_overflow, (unsigned int)nan_overflow);
    // Test 2h: Positive infinity multiplication
    float inf_mult = 1e38f * 1e38f;
    log_values("Test 2h: Infinity multiplication", inf_mult, (int8_t)inf_mult, (int16_t)inf_mult, (unsigned int)inf_mult);
    // Test 2i: Zero times large
    float zero_large = 0.0f * 1e38f;
    log_values("Test 2i: Zero * large", zero_large, (int8_t)zero_large, (int16_t)zero_large, (unsigned int)zero_large);
    // Test 2j: NaN from invalid operation
    float nan_invalid = 0.0f * inf;
    log_values("Test 2j: Zero * Infinity", nan_invalid, (int8_t)nan_invalid, (int16_t)nan_invalid, (unsigned int)nan_invalid);

    // Test 3: Type conversion edge cases (10 tests)
    float t3a = 127.9f;
    log_values("Test 3a: Max int8_t", t3a, (int8_t)t3a, (int16_t)t3a, (unsigned int)t3a);
    float t3b = -128.9f;
    log_values("Test 3b: Min int8_t", t3b, (int8_t)t3b, (int16_t)t3b, (unsigned int)t3b);
    float t3c = 32767.9f;
    log_values("Test 3c: Max int16_t", t3c, (int8_t)t3c, (int16_t)t3c, (unsigned int)t3c);
    float t3d = -32768.9f;
    log_values("Test 3d: Min int16_t", t3d, (int8_t)t3d, (int16_t)t3d, (unsigned int)t3d);
    float t3e = 4294967295.0f;
    log_values("Test 3e: Max unsigned int", t3e, (int8_t)t3e, (int16_t)t3e, (unsigned int)t3e);
    float t3f = 255.5f;
    log_values("Test 3f: Just above uint8_t", t3f, (int8_t)t3f, (int16_t)t3f, (unsigned int)t3f);
    float t3g = -1.999f;
    log_values("Test 3g: Just below -1", t3g, (int8_t)t3g, (int16_t)t3g, (unsigned int)t3g);
    float t3h = 100000.5f;
    log_values("Test 3h: Large float to int", t3h, (int8_t)t3h, (int16_t)t3h, (unsigned int)t3h);
    float t3i = 0.999999f;
    log_values("Test 3i: Near 1 float", t3i, (int8_t)t3i, (int16_t)t3i, (unsigned int)t3i);
    float t3j = -0.999999f;
    log_values("Test 3j: Near -1 float", t3j, (int8_t)t3j, (int16_t)t3j, (unsigned int)t3j);

    // Test 4: Precision tests with small differences (10 tests)
    float x1 = 1.0f, y1 = 1.0f + 1e-6f;
    log_values("Test 4a: 1e-6 difference", y1 - x1, (int8_t)(y1 - x1), (int16_t)(y1 - x1), (unsigned int)(y1 - x1));
    float x2 = 1000.0f, y2 = 1000.0f + 1e-4f;
    log_values("Test 4b: 1e-4 difference", y2 - x2, (int8_t)(y2 - x2), (int16_t)(y2 - x2), (unsigned int)(y2 - x2));
    float x3 = 0.1f, y3 = 0.1f + 1e-7f;
    log_values("Test 4c: 1e-7 difference", y3 - x3, (int8_t)(y3 - x3), (int16_t)(y3 - x3), (unsigned int)(y3 - x3));
    float x4 = 1e6f, y4 = 1e6f + 0.1f;
    log_values("Test 4d: Large with small diff", y4 - x4, (int8_t)(y4 - x4), (int16_t)(y4 - x4), (unsigned int)(y4 - x4));
    float x5 = 0.0001f, y5 = 0.0001f + 1e-8f;
    log_values("Test 4e: Tiny with 1e-8 diff", y5 - x5, (int8_t)(y5 - x5), (int16_t)(y5 - x5), (unsigned int)(y5 - x5));
    float x6 = 10.0f, y6 = 10.0f + 1e-5f;
    log_values("Test 4f: 1e-5 difference", y6 - x6, (int8_t)(y6 - x6), (int16_t)(y6 - x6), (unsigned int)(y6 - x6));
    float x7 = -1.0f, y7 = -1.0f + 1e-6f;
    log_values("Test 4g: Negative 1e-6 diff", y7 - x7, (int8_t)(y7 - x7), (int16_t)(y7 - x7), (unsigned int)(y7 - x7));
    float x8 = 1e-5f, y8 = 1e-5f + 1e-10f;
    log_values("Test 4h: Tiny 1e-10 diff", y8 - x8, (int8_t)(y8 - x8), (int16_t)(y8 - x8), (unsigned int)(y8 - x8));
    float x9 = 100.0f, y9 = 100.0f + 1e-3f;
    log_values("Test 4i: 1e-3 difference", y9 - x9, (int8_t)(y9 - x9), (int16_t)(y9 - x9), (unsigned int)(y9 - x9));
    float x10 = 0.5f, y10 = 0.5f + 1e-6f;
    log_values("Test 4j: 0.5 with 1e-6 diff", y10 - x10, (int8_t)(y10 - x10), (int16_t)(y10 - x10), (unsigned int)(y10 - x10));

    // Test 5: Overflow and underflow (10 tests)
    float max_float = 3.4028235e38f;
    float min_float = 1.1754944e-38f;
    log_values("Test 5a: Overflow multiply Hannah", max_float * 2.0f, (int8_t)(max_float * 2.0f), (int16_t)(max_float * 2.0f), (unsigned int)(max_float * 2.0f));
    log_values("Test 5b: Underflow", min_float / 2.0f, (int8_t)(min_float / 2.0f), (int16_t)(min_float / 2.0f), (unsigned int)(min_float / 2.0f));
    float overflow1 = max_float * 10.0f;
    log_values("Test 5c: Large overflow", overflow1, (int8_t)overflow1, (int16_t)overflow1, (unsigned int)overflow1);
    float underflow1 = min_float / 10.0f;
    log_values("Test 5d: Large underflow", underflow1, (int8_t)underflow1, (int16_t)underflow1, (unsigned int)underflow1);
    float overflow2 = max_float + max_float;
    log_values("Test 5e: Max float add", overflow2, (int8_t)overflow2, (int16_t)overflow2, (unsigned int)overflow transformers);
    float underflow2 = min_float * 0.1f;
    log_values("Test 5f: Min float mult", underflow2, (int8_t)underflow2, (int16_t)underflow2, (unsigned int)underflow2);
    float overflow3 = 1e37f * 10.0f;
    log_values("Test 5g: Near max overflow", overflow3, (int8_t)overflow3, (int16_t)overflow3, (unsigned int)overflow3);
    float underflow3 = min_float / 1e37f;
    log_values("Test 5h: Near min underflow", underflow3, (int8_t)underflow3, (int16_t)underflow3, (unsigned int)underflow3);
    float overflow4 = max_float * max_float;
    log_values("Test 5i: Extreme overflow", overflow4, (int8_t)overflow4, (int16_t)overflow4, (unsigned int)overflow4);
    float underflow4 = min_float * min_float;
    log_values("Test 5j: Extreme underflow", underflow4, (int8_t)underflow4, (int16_t)underflow4, (unsigned int)underflow4);

    return 0;
}