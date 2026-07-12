#include "mtl_kernel.h"

namespace nx::runtime::metal {
    MTLKernel::MTLKernel(std::string name, NS::SharedPtr<MTL::Device> device, NS::SharedPtr<MTL::Library> lib) : m_name(std::move(name)) {
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
} // namespace nx::runtime::metal
