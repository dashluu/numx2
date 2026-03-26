#include "sf_buffer_cache.h"

namespace nx::memory {
    void SFBufferPool::push(SFBufferPtr buff) {
        buff->m_buff_idx = m_free_buffs.size();
        m_free_buffs.emplace_back(buff);
    }

    SFBufferPtr SFBufferPool::acquire() {
        SFBufferPtr buff = m_free_buffs.back();
        m_free_buffs.pop_back();
        buff->m_buff_idx = m_used_buffs.size();
        m_used_buffs.emplace_back(buff);
        return buff;
    }

    void SFBufferPool::release(SFBufferPtr buff) {
        std::size_t buff_idx = buff->m_buff_idx;
        // This only works if pool and buffer are on the same device
        assert(buff_idx < m_used_buffs.size() && m_used_buffs[buff_idx] == buff);
        SFBufferPtr last_buff = m_used_buffs[m_used_buffs.size() - 1];
        last_buff->m_buff_idx = buff_idx;
        m_used_buffs[buff_idx] = last_buff;
        m_used_buffs[m_used_buffs.size() - 1] = buff;
        m_used_buffs.pop_back();
        push(buff);
    }

    SFBufferPool::~SFBufferPool() {
        for (auto &buff : m_free_buffs) {
            delete buff;
        }

        for (auto &buff : m_used_buffs) {
            delete buff;
        }
    }

    Buffer *SFBufferCache::alloc(std::size_t nbytes) {
        if (nbytes == 0) {
            throw std::invalid_argument("nbytes must be > 0.");
        }

        std::size_t aligned_nbytes = align_up_power_2(nbytes);
        std::size_t npools = get_pool_idx(aligned_nbytes) + 1;

        if (npools > m_pools.size()) {
            m_pools.resize(npools);
        }

        auto &pool = get_pool(aligned_nbytes);
        SFBufferPtr buff;

        if (pool.empty()) {
            buff = new SFBuffer(m_allocator.get(), nbytes);
            pool.push(buff);
        }

        buff = pool.acquire();
        return buff;
    }

    void SFBufferCache::free(Buffer *buff) {
        if (!buff) {
            return;
        }

        // Check if the buffer is on the correct device
        assert(buff->allocator()->id() == m_allocator->id());
        SFBufferPtr sf_buff = static_cast<SFBuffer *>(buff);
        std::size_t aligned_nbytes = align_up_power_2(sf_buff->nbytes());
        auto &pool = get_pool(aligned_nbytes);
        pool.release(sf_buff);
    }
} // namespace nx::memory
