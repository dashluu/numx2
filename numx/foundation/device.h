#pragma once

#include "utils.h"
#include <memory>

namespace nx::foundation {
    enum struct DeviceKind {
        CPU,
        MPS
    };

    struct Device {
    private:
        usize m_id;
        DeviceKind m_kind;

    public:
        Device(usize id, DeviceKind kind) : m_id(id), m_kind(kind) {}
        Device(const Device &) = delete;
        Device(Device &&) noexcept = delete;
        ~Device() = default;
        Device &operator=(const Device &) = delete;
        Device &operator=(Device &&) noexcept = delete;
        DeviceKind kind() const { return m_kind; }
        usize id() const { return m_id; }
        bool operator==(const Device &device) const { return m_kind == device.m_kind && m_id == device.m_id; }
        friend std::ostream &operator<<(std::ostream &os, const Device &device) { return os << device.str(); }

        std::string str() const {
            std::string result;

            switch (m_kind) {
            case DeviceKind::CPU:
                result = "cpu";
                break;
            default:
                result = "mps";
                break;
            }

            result += std::to_string(m_id);
            return result;
        }
    };

    using DevicePtr = std::unique_ptr<Device>;
    const std::string default_device_name = "mps:0";
    inline DevicePtr make_device(usize id, DeviceKind kind) { return std::make_unique<Device>(id, kind); }
} // namespace nx::foundation
