#pragma once

#include "utils.h"

struct Add {
    template <class T>
    T operator()(T lhs, T rhs) { return lhs + rhs; }
};

struct Sub {
    template <class T>
    T operator()(T lhs, T rhs) { return lhs - rhs; }
};

struct Mul {
    template <class T>
    T operator()(T lhs, T rhs) { return lhs * rhs; }
};

struct Div {
    template <class T>
    T operator()(T lhs, T rhs) { return lhs / rhs; }
};

struct Pow {
    template <class T>
    T operator()(T lhs, T rhs) { return metal::pow(lhs, rhs); }
};

struct LogicAnd {
    bool operator()(bool lhs, bool rhs) { return lhs && rhs; }
};

struct LogicOr {
    bool operator()(bool lhs, bool rhs) { return lhs || rhs; }
};

struct BitwiseAnd {
    template <class T>
    bool operator()(T lhs, T rhs) { return lhs & rhs; }
};

struct BitwiseOr {
    template <class T>
    bool operator()(T lhs, T rhs) { return lhs | rhs; }
};

struct BitwiseXor {
    template <class T>
    bool operator()(T lhs, T rhs) { return lhs ^ rhs; }
};

struct Eq {
    template <class T>
    bool operator()(T lhs, T rhs) { return lhs == rhs; }
};

struct Neq {
    template <class T>
    bool operator()(T lhs, T rhs) { return lhs != rhs; }
};

struct Less {
    template <class T>
    bool operator()(T lhs, T rhs) { return lhs < rhs; }
};

struct Greater {
    template <class T>
    bool operator()(T lhs, T rhs) { return lhs > rhs; }
};

struct Leq {
    template <class T>
    bool operator()(T lhs, T rhs) { return lhs <= rhs; }
};

struct Geq {
    template <class T>
    bool operator()(T lhs, T rhs) { return lhs >= rhs; }
};

struct Minimum {
    template <class T>
    T operator()(T lhs, T rhs) { return lhs < rhs ? lhs : rhs; }
};

struct Maximum {
    template <class T>
    T operator()(T lhs, T rhs) { return lhs > rhs ? lhs : rhs; }
};
