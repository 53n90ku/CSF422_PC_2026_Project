#ifndef RNG_H
#define RNG_H

// xoshiro256** plus a few distributions. Same code path on host for now
// so it is easier to move to CUDA later.

#include <cmath>
#include <cstdint>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Rng {
    uint64_t s[4];
};

static inline uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t rng_next(Rng& rng) {
    const uint64_t result = rotl(rng.s[1] * 5, 7) * 9;
    const uint64_t t = rng.s[1] << 17;
    rng.s[2] ^= rng.s[0];
    rng.s[3] ^= rng.s[1];
    rng.s[1] ^= rng.s[2];
    rng.s[0] ^= rng.s[3];
    rng.s[2] ^= t;
    rng.s[3] = rotl(rng.s[3], 45);
    return result;
}

static inline uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline Rng rng_seed(uint64_t seed) {
    Rng rng;
    rng.s[0] = splitmix64(seed);
    rng.s[1] = splitmix64(seed);
    rng.s[2] = splitmix64(seed);
    rng.s[3] = splitmix64(seed);
    return rng;
}

static inline double rng_uniform(Rng& rng) {
    return static_cast<double>(rng_next(rng) >> 11) * 0x1.0p-53;
}

static inline double rng_uniform(Rng& rng, double lo, double hi) {
    return lo + (hi - lo) * rng_uniform(rng);
}

static inline double rng_normal(Rng& rng) {
    double u1, u2;
    do { u1 = rng_uniform(rng); } while (u1 == 0.0);
    u2 = rng_uniform(rng);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

static inline double rng_normal(Rng& rng, double mean, double sd) {
    return mean + sd * rng_normal(rng);
}

// Binomial(n, p) by geometric waiting times. Flip p > 0.5 to cut work.
static inline int rng_binomial(Rng& rng, int n, double p) {
    if (n <= 0 || p <= 0.0) return 0;
    if (p >= 1.0) return n;

    bool flipped = false;
    if (p > 0.5) {
        p = 1.0 - p;
        flipped = true;
    }

    const double log_q = std::log1p(-p);
    double pos = 0.0;
    int count = 0;
    for (;;) {
        double u = rng_uniform(rng);
        if (u <= 0.0) continue;
        pos += std::floor(std::log(u) / log_q) + 1.0;
        if (pos > static_cast<double>(n)) break;
        ++count;
        if (count >= n) break;
    }

    return flipped ? n - count : count;
}

#endif
