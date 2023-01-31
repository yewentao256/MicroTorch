#pragma once

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <vector>

namespace tinytorch {

using data_t = float;
using index_t = long unsigned int;

struct Edge;

struct TensorImpl {
 private:
  // variables

  index_t offset_ = 0;
  index_t ndim_;
  std::vector<index_t> shape_;  // TODO: 之后支持多维
  std::vector<index_t> stride_;

 public:
  std::vector<data_t> data_;
  std::vector<data_t> grad_;  // for .backward
  std::shared_ptr<Edge> edge_;
  std::string arch_ = "host";

  // constructors
  TensorImpl(size_t size, index_t ndim = 1) : data_(size), ndim_(ndim) {
    shape_ = {size};
    // init_stride();
  }
  TensorImpl(std::vector<data_t> data, index_t ndim = 1)
      : data_(data), ndim_(ndim) {
    shape_ = {data.size()};
    // init_stride();
  }

  // funcs
  void init_stride() {
    index_t stride = 1;
    for (index_t i = ndim_ - 1; i >= 0; i--) {
      stride_[i] = stride;
      stride *= shape_[i];
    }
  }
};

struct Tensor {
 private:
  std::shared_ptr<TensorImpl> impl_;

 public:
  // constructor
  Tensor(size_t size = 0) : impl_(std::make_shared<TensorImpl>(size)) {}
  Tensor(std::vector<data_t> data)
      : impl_(std::make_shared<TensorImpl>(data)) {}

  // operator override
  data_t& operator[](int idx) { return impl_->data_[idx]; }

  // tensor functions
  size_t size() { return impl_->data_.size(); }
  void resize(size_t size) {
    assert(impl_);
    impl_->data_.resize(size, 0);
    impl_->grad_.resize(size, 0);
  }
  void clearGrad() { impl_->grad_.clear(); }

  Tensor grad() { return Tensor(impl_->grad_); }

  void addInplace(Tensor t) {
    resize(t.size());
    for (size_t i = 0; i < size(); i++) {
      impl_->data_[i] += t[i];
    }
  }
  void addGradInplace(Tensor g) {
    resize(g.size());
    for (size_t i = 0; i < size(); i++) {
      impl_->grad_[i] += g[i];
    }
  }
  std::shared_ptr<Edge> getEdge() { return impl_->edge_; };
  void setEdge(std::shared_ptr<Edge> edge) { impl_->edge_ = edge; };

  data_t* data() { return impl_->data_.data(); };

  Tensor cuda() {
    impl_->arch_ = "cuda";
    return *this;
  };
  std::string arch() { return impl_->arch_; };
};

}  // namespace tinytorch