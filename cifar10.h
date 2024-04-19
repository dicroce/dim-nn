
#ifndef __cifar10_h
#define __cifar10_h

#include <map>
#include <vector>
#include <cstdint>
#include "mm.h"

std::pair<uint8_t, uint8_t*> get_image(r_memory_map& mm, uint32_t index);
std::vector<uint8_t> rgb32x32_to_gray8(const uint8_t* src);

#endif
