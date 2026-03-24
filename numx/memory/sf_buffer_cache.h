#pragma once

#include "buffer_memory.h"
#include <bit>
#include <vector>

namespace nx::memory {
    inline std::size_t align_up_power_2(std::size_t nbytes) { return std::bit_ceil(nbytes); }

    class SFBufferPool;
    class SFBufferCache;

    struct SFBuffer : public Buffer {
    private:
        std::size_t m_buff_idx;

        friend class SFBufferPool;

    public:
        SFBuffer(BufferAllocator *allocator, std::size_t nbytes) : Buffer(allocator, nbytes) {}
    };

    using SFBufferPtr = SFBuffer *;

    class SFBufferPool {
    private:
        std::vector<SFBufferPtr> m_free_buffs;
        std::vector<SFBufferPtr> m_used_buffs;

        void push(SFBufferPtr buff);
        SFBufferPtr acquire();
        void release(SFBufferPtr buff);
        friend class SFBufferCache;

    public:
        SFBufferPool() = default;
        SFBufferPool(const SFBufferPool &) = delete;
        SFBufferPool(SFBufferPool &&) noexcept = default;
        ~SFBufferPool();
        SFBufferPool &operator=(const SFBufferPool &) = delete;
        SFBufferPool &operator=(SFBufferPool &&) noexcept = default;
        bool empty() const { return m_free_buffs.empty(); }
        std::size_t size() const { return m_free_buffs.size(); }
    };

    class SFBufferCache : public BufferMemory {
    private:
        std::vector<SFBufferPool> m_pools;

        static std::size_t get_pool_idx(std::size_t nbytes) { return std::bit_width(nbytes - 1); }

        SFBufferPool &get_pool(std::size_t nbytes) {
            std::size_t pool_idx = get_pool_idx(nbytes);
            return m_pools.at(pool_idx);
        }

        friend class SFBuffer;

    public:
        SFBufferCache(BufferAllocatorPtr allocator, std::size_t max_nbytes);
        Buffer *alloc(std::size_t nbytes) override;
        void free(Buffer *buff) override;
    };
} // namespace nx::memory
