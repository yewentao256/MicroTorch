#pragma once

#include "tensor.hpp"

namespace tinytorch {

typedef void (*ApplyFunc)(std::vector<Tensor>&, std::vector<Tensor>&);

class OpNode {
 public:
  std::string name_;
  std::vector<Tensor> ins_, outs_;


  explicit OpNode(const std::string& name) : name_(name) {}
  explicit OpNode(std::string&& name) : name_(name) {}

  OpNode(const OpNode&) = delete;        // no copy constructor
  OpNode(OpNode&&) = delete;             // no move constructor
  OpNode& operator=(OpNode&) = delete;   // no copy assignment
  OpNode& operator=(OpNode&&) = delete;  // no move assignment

  OpNode& ins(const std::vector<Tensor>& ins) {
    for (auto& in : ins) {
      // ins_.push_back(Tensor(in.impl()));
      ins_.push_back(in);
    }
    return *this;
  }

  OpNode& outs(const std::vector<Tensor>& outs) {
    for (auto& out : outs) {
      // outs_.push_back(Tensor(out.impl()));
      outs_.push_back(out);
    }
    return *this;
  }

  void apply(ApplyFunc func) { (*func)(ins_, outs_); }
};

}  // namespace tinytorch