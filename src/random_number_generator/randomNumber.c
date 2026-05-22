
#include <stdint.h>
#include <time.h>

uint32_t generateSeedRNG()
{
    uint32_t t = (uint32_t)time(NULL);
    uint32_t c = (uint32_t)clock();
    uint32_t rng_state = t ^ (c << 16);

    if (rng_state == 0)
        rng_state = 1;

    return rng_state;
}

uint32_t xorshift32(void)
{
    static uint32_t x = 0;

    if (x == 0)
    {
        x = generateSeedRNG();
    }

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;

    return x;
}

float randomNumber(float min, float max)
{
    uint32_t r = xorshift32();
    uint32_t mantissa = r & 0x007FFFFFu;
    uint32_t bits = 0x3F800000u | mantissa;

    union
    {
        uint32_t i;
        float f;
    } u;

    u.i = bits;
    float normalized = u.f - 1.0f;

    return min + (max - min) * normalized;
}
