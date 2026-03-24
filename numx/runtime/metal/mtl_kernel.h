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
        explicit MTLKernel(std::string name) : m_name(std::move(name)) {}
        MTLKernel(const MTLKernel &) = delete;
        MTLKernel(MTLKernel &&) noexcept = delete;
        ~MTLKernel() = default;
        MTLKernel &operator=(const MTLKernel &) = delete;
        MTLKernel &operator=(MTLKernel &&) noexcept = delete;

        void init(NS::SharedPtr<MTL::Device> device, NS::SharedPtr<MTL::Library> lib) {
            NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
            auto ns_name = NS::String::string(m_name.c_str(), NS::UTF8StringEncoding);
            m_function = NS::TransferPtr<MTL::Function>(lib->newFunction(ns_name));
            NS::Error *error = nullptr;
            m_state = NS::TransferPtr<MTL::ComputePipelineState>(device->newComputePipelineState(m_function.get(), &error));

            if (error) {
                pool->release();
                const std::string description = error->localizedDescription()->utf8String();
                throw std::runtime_error(description);
            }

            pool->release();
        }

        NS::SharedPtr<MTL::Function> function() const { return m_function; }
        NS::SharedPtr<MTL::ComputePipelineState> state() const { return m_state; }
    };

    using MTLKernelPtr = std::unique_ptr<MTLKernel>;

    template <class... Args>
    MTLKernelPtr make_mtl_kernel(Args &&...args) { return std::make_unique<MTLKernel>(std::forward<Args>(args)...); }
} // namespace nx::runtime::metal
