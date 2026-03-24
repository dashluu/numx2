#include "../numx/memory/cpu_allocator.h"
#include "../numx/memory/sf_buffer_cache.h"
#include <gtest/gtest.h>

using namespace nx::memory;

TEST(TestMemory, TestMemoryLeak) {
    // CPUAllocator allocator(0);
    // SFBufferCache cache(&allocator, 1024);
    // Buffer * buff = cache.alloc(128);
    // cache.free(buff);
}
