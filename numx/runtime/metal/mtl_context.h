#pragma once

#include "../../foundation/random.h"
#include "../../graph/functional.h"
#include "../runtime_context.h"
#include "mtl_kernel.h"
#include <memory>

namespace nx::runtime::metal {
    using foundation::DTypeKind;
    using foundation::usize;

    class MTLContext : public RuntimeContext {
    private:
        NS::SharedPtr<MTL::Device> m_mtl_device;
        NS::SharedPtr<MTL::Library> m_lib;
        NS::SharedPtr<MTL::CommandQueue> m_cmd_queue;
        std::unordered_map<std::string, MTLKernelPtr> m_kernel_by_name;

        void init_kernel(const std::string &name);
        void init_kernels(const std::vector<std::string_view> &names, DTypeKind dtype_kind);
        void init_strided_kernels(const std::vector<std::string_view> &names, DTypeKind dtype_kind);
        void init_kernels(const std::string &name, DTypeKind dtype_kind);
        void init_strided_kernels(const std::string &name, DTypeKind dtype_kind);
        void init_initializer_kernels();
        void init_unary_kernels();
        void init_binary_kernels();
        void init_reduce_kernels();
        void init_gemm_kernels();
        void init_copy_kernels();

    public:
        MTLContext(MTL::Device *mtl_device, const std::string &lib_path, DevicePtr device, BufferMemoryPtr memory, RandomKeyGeneratorPtr key_gen);
        void init_kernels();
        bool register_kernel(const std::string &name, MTLKernelPtr kernel);
        NS::SharedPtr<MTL::Device> mtl_device() const { return m_mtl_device; }
        NS::SharedPtr<MTL::CommandQueue> cmd_queue() const { return m_cmd_queue; }
        MTLKernel *kernel(const std::string &name) const { return m_kernel_by_name.contains(name) ? m_kernel_by_name.at(name).get() : nullptr; }
    };

    using MTLContextPtr = std::unique_ptr<MTLContext>;
} // namespace nx::runtime::metal
