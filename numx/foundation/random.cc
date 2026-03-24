#include "random.h"

namespace nx::foundation {
    void RandomKeyGenerator::update_counter() {
        std::uint32_t ctr_x = m_counter + 1;
        std::uint32_t ctr_y = ctr_x + 1;
        m_counter = static_cast<uint64_t>(ctr_x) << 32 | ctr_y;
    }

    std::uint64_t RandomKeyGenerator::next() {
        std::uint64_t key1 = threefry2x32(m_key, m_counter);
        update_counter();
        std::uint64_t key2 = threefry2x32(m_key, m_counter);
        update_counter();
        m_key = key1;
        return key2;
    }

    std::uint64_t current_time_seed() {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    }

    std::uint64_t threefry2x32(std::uint64_t key, std::uint64_t counter) {
        std::uint32_t key_x = key >> 32;
        std::uint32_t key_y = key;
        std::uint32_t ks[] = {key_x, key_y, 0x1BD11BDA ^ key_x ^ key_y};
        std::uint32_t X_x = counter >> 32;
        std::uint32_t X_y = counter;
        X_x += ks[0];
        X_y += ks[1];
        short j = 1;

        // 20 rounds
        for (short i = 0; i < 20; ++i) {
            X_x += X_y;
            X_y = rotl32(X_y, s_rot2x32[i % 8]);
            X_y ^= X_x;

            if (i % 4 == 3) {
                X_x += ks[j % 3];
                X_y += ks[(j + 1) % 3];
                X_y += j;
                j++;
            }
        }

        return static_cast<uint64_t>(X_x) << 32 | X_y;
    }
} // namespace nx::foundation
