#pragma once

#include "functional.h"

namespace nx::graph {
    class Graph : public std::enable_shared_from_this<Graph> {
    protected:
        OpPtr m_output;
        std::vector<Op *> m_fw_tape;
        std::vector<Op *> m_bw_tape;

        void recur_fw_sort(Op *op);
        void recur_bw_sort(Op *op);

    public:
        explicit Graph(OpPtr output) : m_output(output) {}
        Graph(const Graph &) = delete;
        Graph(Graph &&) noexcept = delete;
        virtual ~Graph() = default;
        Graph &operator=(const Graph &) = delete;
        Graph &operator=(Graph &&) noexcept = delete;
        OpPtr output() const { return m_output; }
        Op *output_raw() const { return m_output.get(); }
        std::size_t fw_tape_size() const { return m_fw_tape.size(); }
        std::size_t bw_tape_size() const { return m_bw_tape.size(); }
        void fw_sort();
        void bw_sort();
        void clear_grad();
        // std::string str() const;
        // friend std::ostream &operator<<(std::ostream &os, const Graph &graph) { return os << graph.str(); }
        std::vector<Op *>::const_iterator fw_begin() const { return m_fw_tape.cbegin(); }
        std::vector<Op *>::const_iterator fw_end() const { return m_fw_tape.cend(); }
        std::vector<Op *>::const_iterator bw_begin() const { return m_bw_tape.cbegin(); }
        std::vector<Op *>::const_iterator bw_end() const { return m_bw_tape.cend(); }
    };

    using GraphPtr = std::shared_ptr<Graph>;

    template <class... Args>
    GraphPtr make_graph(Args &&...args) { return std::make_shared<Graph>(std::forward<Args>(args)...); }
} // namespace nx::graph
