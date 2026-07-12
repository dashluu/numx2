#pragma once

#include "../../foundation/utils.h"
#include "Metal/Metal.hpp"

namespace nx::runtime::metal {
    struct MTLKernel {
    private:
        // NS::SharedPtr works similarly to std::shared_ptr
        NS::SharedPtr<MTL::Function> m_function;
        NS::SharedPtr<MTL::ComputePipelineState> m_state;
        std::string m_name;

    public:
        MTLKernel(std::string name, NS::SharedPtr<MTL::Device> device, NS::SharedPtr<MTL::Library> lib);
        MTLKernel(const MTLKernel &) = delete;
        MTLKernel(MTLKernel &&) noexcept = delete;
        ~MTLKernel() = default;
        MTLKernel &operator=(const MTLKernel &) = delete;
        MTLKernel &operator=(MTLKernel &&) noexcept = delete;
        NS::SharedPtr<MTL::Function> function() const { return m_function; }
        NS::SharedPtr<MTL::ComputePipelineState> state() const { return m_state; }
    };

    using MTLKernelPtr = std::unique_ptr<MTLKernel>;

    template <class... Args>
    MTLKernelPtr make_mtl_kernel(Args &&...args) { return std::make_unique<MTLKernel>(std::forward<Args>(args)...); }
} // namespace nx::runtime::metal
