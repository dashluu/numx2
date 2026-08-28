#include "mtl_context.h"

namespace nx::runtime::metal {
    void MTLContext::init_kernel(const std::string &name) {
        MTLKernelPtr kernel = make_mtl_kernel(name, m_mtl_device, m_lib);
        m_kernel_by_name[name] = std::move(kernel);
    }

    void MTLContext::init_kernels(const std::vector<std::string_view> &names, DTypeKind dtype_kind) {
        for (auto &name : names) {
            init_kernels(std::string(name), dtype_kind);
        }
    }

    void MTLContext::init_strided_kernels(const std::vector<std::string_view> &names, DTypeKind dtype_kind) {
        for (auto &name : names) {
            init_strided_kernels(std::string(name), dtype_kind);
        }
    }

    void MTLContext::init_kernels(const std::string &name, DTypeKind dtype_kind) {
        for (auto &dtype : foundation::all_dtypes) {
            if (dtype->has_kind(dtype_kind)) {
                init_kernel(std::format("{}_{}", name, dtype->str()));
            }
        }
    }

    void MTLContext::init_strided_kernels(const std::string &name, DTypeKind dtype_kind) {
        for (auto &dtype : foundation::all_dtypes) {
            if (dtype->has_kind(dtype_kind)) {
                init_kernel(std::format("strided_{}_{}", name, dtype->str()));
            }
        }
    }

    void MTLContext::init_initializer_kernels() {
        init_kernels(std::string(graph::FullOp::s_opname), DTypeKind::All);
        init_kernels(std::string(graph::ArangeOp::s_opname), DTypeKind::Numeric);
        init_kernels(std::string(graph::UniformOp::s_opname), DTypeKind::Float);
    }

    void MTLContext::init_unary_kernels() {
        std::vector<std::string_view> unary_name_views = {graph::NegOp::s_opname, graph::SqOp::s_opname};
        std::vector<std::string_view> unary_int_name_views = {graph::BitwiseNot::s_opname};
        std::vector<std::string_view> unary_bool_name_views = {graph::LogicNot::s_opname};

        std::vector<std::string_view> unary_float_name_views = {
            graph::SqrtOp::s_opname,
            graph::ExpOp::s_opname,
            graph::LogOp::s_opname,
            graph::RecipOp::s_opname,
            graph::SinOp::s_opname,
            graph::CosOp::s_opname};

        init_kernels(unary_name_views, DTypeKind::Numeric);
        init_strided_kernels(unary_name_views, DTypeKind::Numeric);
        init_kernels(unary_int_name_views, DTypeKind::Int);
        init_strided_kernels(unary_int_name_views, DTypeKind::Int);
        init_kernels(unary_bool_name_views, DTypeKind::Bool);
        init_strided_kernels(unary_bool_name_views, DTypeKind::Bool);
        init_kernels(unary_float_name_views, DTypeKind::Float);
        init_strided_kernels(unary_float_name_views, DTypeKind::Float);
    }

    void MTLContext::init_binary_kernels() {
        std::vector<std::string_view> binary_float_name_views = {
            graph::PowOp::s_opname};

        std::vector<std::string_view> binary_name_views = {
            graph::AddOp::s_opname,
            graph::SubOp::s_opname,
            graph::MulOp::s_opname,
            graph::DivOp::s_opname,
            graph::LessOp::s_opname,
            graph::GreaterOp::s_opname,
            graph::MinimumOp::s_opname,
            graph::MaximumOp::s_opname};

        std::vector<std::string_view> binary_int_name_views = {
            graph::BitwiseAndOp::s_opname,
            graph::BitwiseOrOp::s_opname,
            graph::BitwiseXorOp::s_opname};

        std::vector<std::string_view> binary_bool_name_views = {
            graph::LogicAndOp::s_opname,
            graph::LogicOrOp::s_opname};

        std::vector<std::string_view> equal_name_views = {
            graph::EqOp::s_opname,
            graph::NeqOp::s_opname};

        init_kernels(binary_float_name_views, DTypeKind::Float);
        init_strided_kernels(binary_float_name_views, DTypeKind::Float);
        init_kernels(binary_name_views, DTypeKind::Numeric);
        init_strided_kernels(binary_name_views, DTypeKind::Numeric);
        init_kernels(binary_int_name_views, DTypeKind::Int);
        init_strided_kernels(binary_int_name_views, DTypeKind::Int);
        init_kernels(binary_bool_name_views, DTypeKind::Bool);
        init_strided_kernels(binary_bool_name_views, DTypeKind::Bool);
        init_kernels(equal_name_views, DTypeKind::All);
        init_strided_kernels(equal_name_views, DTypeKind::All);
    }

    void MTLContext::init_reduce_kernels() {
        std::vector<std::string_view> reduce_name_views = {
            graph::SumOp::s_opname,
            graph::MaxOp::s_opname,
            graph::MinOp::s_opname,
            graph::ArgmaxOp::s_opname,
            graph::ArgminOp::s_opname};

        for (auto &name_view : reduce_name_views) {
            auto name = std::string(name_view);
            init_kernels(std::format("{}_all", name), DTypeKind::Numeric);
            init_kernels(std::format("strided_{}_all", name), DTypeKind::Numeric);

            for (uint8_t i = 1; i <= 32; i <<= 1) {
                for (uint8_t j = 1; i * j <= 32; j <<= 1) {
                    init_kernels(std::format("{}_col_{}x{}", name, i, j), DTypeKind::Numeric);
                    init_kernels(std::format("strided_{}_col_{}x{}", name, i, j), DTypeKind::Numeric);
                }
            }
        }
    }

    void MTLContext::init_gemm_kernels() {
        init_kernels("simd_gevv", DTypeKind::Numeric);
        init_kernels("strided_simd_gevv", DTypeKind::Numeric);
        init_kernels("naive_gemm2d", DTypeKind::Numeric);
        init_kernels("tiled_gemm2d", DTypeKind::Float);
        init_kernels("tensor_gemm2d", DTypeKind::Float);
        init_kernels("naive_gemm3d", DTypeKind::Numeric);
        init_kernels("tiled_gemm3d", DTypeKind::Float);
        init_kernels("tensor_gemm3d", DTypeKind::Float);
        init_kernels("strided_naive_gemm2d", DTypeKind::Numeric);
        init_kernels("strided_tiled_gemm2d", DTypeKind::Float);
        init_kernels("strided_naive_gemm3d", DTypeKind::Numeric);
        init_kernels("strided_tiled_gemm3d", DTypeKind::Float);
    }

    void MTLContext::init_copy_kernels() {
        for (auto &dtype1 : foundation::all_dtypes) {
            for (auto &dtype2 : foundation::all_dtypes) {
                init_kernel(std::format("copy_{}_{}", dtype1->str(), dtype2->str()));
                init_kernel(std::format("strided_copy_{}_{}", dtype1->str(), dtype2->str()));
            }
        }
    }

    MTLContext::MTLContext(MTL::Device *mtl_device, const std::string &lib_path, DevicePtr device, BufferMemoryPtr memory, RandomKeyGeneratorPtr key_gen) : RuntimeContext(std::move(device), std::move(memory), std::move(key_gen)) {
        m_mtl_device = NS::TransferPtr<MTL::Device>(mtl_device);
        NS::String *path = NS::String::string(lib_path.c_str(), NS::ASCIIStringEncoding);
        auto url = NS::URL::fileURLWithPath(path);
        NS::Error *error = nullptr;
        m_lib = NS::TransferPtr<MTL::Library>(m_mtl_device->newLibrary(url, &error));

        if (error) {
            const std::string description = error->localizedDescription()->utf8String();
            throw std::runtime_error(description);
        }

        m_cmd_queue = NS::TransferPtr<MTL4::CommandQueue>(m_mtl_device->newMTL4CommandQueue());
        m_cmd_buff = NS::TransferPtr<MTL4::CommandBuffer>(m_mtl_device->newCommandBuffer());
        m_cmd_allocator = NS::TransferPtr<MTL4::CommandAllocator>(m_mtl_device->newCommandAllocator());
    }

    void MTLContext::init_kernels() {
        // Initializes kernels here
        init_initializer_kernels();
        init_unary_kernels();
        init_binary_kernels();
        init_reduce_kernels();
        init_gemm_kernels();
        init_copy_kernels();
    }

    bool MTLContext::register_kernel(const std::string &name, MTLKernelPtr kernel) {
        if (m_kernel_by_name.contains(name)) {
            return false;
        }

        m_kernel_by_name.insert(std::make_pair(name, std::move(kernel)));
        return true;
    }
} // namespace nx::runtime::metal
