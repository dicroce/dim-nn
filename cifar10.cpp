
#include "cifar10.h"

using namespace std;

pair<uint8_t, uint8_t*> get_image(r_memory_map& mm, uint32_t index)
{
    // Images are 3073 bytes, so you can index one by doing (i * 3073).
    // First byte of image is label.
    // Next 3072 bytes are the image. 1024 bytes of red, 1024 bytes of green, 1024 bytes of blue.
    // Images are all 32x32

    uint8_t* img = ((uint8_t*)mm.map() + (index * 3073));

    return make_pair(*img, img+1);
}

vector<uint8_t> rgb32x32_to_gray8(const uint8_t* src)
{
    vector<uint8_t> output(32*32);

    const uint8_t* s_r = src;
    const uint8_t* s_g = src + 1024;
    const uint8_t* s_b = src + 2048;

    for(int i = 0; i < 1024; ++i)
    {
        uint32_t sum = *s_r;
        ++s_r;
        sum += *s_g;
        ++s_g;
        sum += *s_b;
        ++s_b;
        output[i] = sum / 3;
    }

    return output;
}
