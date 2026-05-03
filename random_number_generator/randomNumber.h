#ifndef RANDOMNUMBER_H
#define RANDOMNUMBER_H

#include <stdint.h>

float randomNumber(float min, float max);
uint32_t xorshift32(void);
uint32_t generateSeedRNG();

#endif