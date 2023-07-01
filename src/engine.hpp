#pragma once

#include <unordered_map>

#include "tensor.hpp"

namespace tinytorch {

typedef void (*ApplyFunc)(std::vector<Tensor>&, std::vector<Tensor>&);

class OpRegistry {
 public:
  static OpRegistry& Instance() {
    static OpRegistry instance;
    return instance;
  }

  bool RegisterOp(const std::string& name, ApplyFunc func) {
    if (ops_.count(name) > 0) {
      return false;
    }
    ops_[name] = func;
    return true;
  }

  ApplyFunc GetOp(const std::string& name) {
    if (ops_.count(name) > 0) {
      return ops_[name];
    }
    return nullptr;
  }

 private:
  std::unordered_map<std::string, ApplyFunc> ops_;
};

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

  void apply() {
    ApplyFunc func = OpRegistry::Instance().GetOp(name_);
    TORCH_CHECK(func != nullptr, "function <", name_,
                "> has not been registered!");
    (*func)(ins_, outs_);
  }
};

}  // namespace tinytorch