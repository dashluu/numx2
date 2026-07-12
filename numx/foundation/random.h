#pragma once

#include "utils.h"
#include <chrono>
#include <memory>

namespace nx::foundation {
    struct RandomKeyGenerator {
    private:
        std::uint64_t m_key;
        std::uint64_t m_counter = 1;

        void update_counter();

    public:
        explicit RandomKeyGenerator(std::uint64_t seed) : m_key(seed) {}
        RandomKeyGenerator(const RandomKeyGenerator &) = delete;
        RandomKeyGenerator(RandomKeyGenerator &&) noexcept = delete;
        ~RandomKeyGenerator() = default;
        RandomKeyGenerator &operator=(const RandomKeyGenerator &) = delete;
        RandomKeyGenerator &operator=(RandomKeyGenerator &&) noexcept = delete;
        std::uint64_t next();
    };

    using RandomKeyGeneratorPtr = std::unique_ptr<RandomKeyGenerator>;

    inline std::unique_ptr<RandomKeyGenerator> make_random_key_generator(std::uint64_t seed) {
        return std::make_unique<RandomKeyGenerator>(seed);
    }

    inline static std::optional<std::uint64_t> s_seed;
    std::uint64_t current_time_seed();

    inline std::uint64_t seed() {
        if (!s_seed) {
            s_seed = current_time_seed();
        }

        return *s_seed;
    }

    inline static constexpr std::uint32_t s_rot2x32[] = {13, 15, 26, 6, 17, 29, 16, 24};
    inline std::uint32_t rotl32(std::uint32_t x, std::uint32_t N) { return (x << (N & 31)) | (x >> ((32 - N) & 31)); }
    std::uint64_t threefry2x32(std::uint64_t key, std::uint64_t counter);
} // namespace nx::foundation
