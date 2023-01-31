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

struct TensorImpl2 {
 private:
  data_t* data_;
  index_t offset_;
  index_t ndim_;
  index_t* shape_;
  index_t* stride_;

 public:
  TensorImpl2(index_t ndim, index_t* shape) : ndim_(ndim), shape_(shape) {
    init_stride();
  };
  TensorImpl2(data_t* data, index_t ndim, index_t* shape)
      : data_(data), ndim_(ndim), shape_(shape) {
    init_stride();
  };

  std::vector<float> grad;  // for .backward
  std::shared_ptr<Edge> edge;

  void init_stride() {
    int stride = 1;
    for (index_t i = ndim_ - 1; i >= 0; i--) {
      stride_[i] = stride;
      stride *= shape_[i];
    }
  }

  // get value
  data_t operator[](std::vector<index_t> inds) {
    assert(ndim_ == inds.size());
    index_t offset = offset_;
    for(index_t i = 0; i < ndim_; ++i)
        offset += inds[i] * stride_[i];
    return data_[offset];
  }
};

struct Tensor2 {
 private:
  std::shared_ptr<TensorImpl2> impl_;

 public:
  // constructor
  Tensor2(index_t ndim, index_t* shape)
      : impl_(std::make_shared<TensorImpl2>(ndim, shape)) {}
  Tensor2(data_t* data, index_t ndim, index_t* shape)
      : impl_(std::make_shared<TensorImpl2>(data, ndim, shape)) {}

  // operator override
  data_t operator[](std::vector<index_t> idxs) { return (*impl_)[idxs]; }

  /* // Tensor2 functions
  size_t size() { return impl_->data.size(); }
  void resize(size_t size) {
    assert(impl_);
    impl_->data.resize(size, 0);
    impl_->grad.resize(size, 0);
  }
  void clearGrad() { impl_->grad.clear(); }

  Tensor2 grad() { return Tensor2(impl_->grad); }

  void addInplace(Tensor2 t) {
    resize(t.size());
    for (size_t i = 0; i < size(); i++) {
      impl_->data[i] += t[i];
    }
  }
  void addGradInplace(Tensor2 g) {
    resize(g.size());
    for (size_t i = 0; i < size(); i++) {
      impl_->grad[i] += g[i];
    }
  }
  std::shared_ptr<Edge> getEdge() { return impl_->edge; };
  void setEdge(std::shared_ptr<Edge> edge) { impl_->edge = edge; };

  float* data() { return impl_->data.data(); };

  Tensor2 cuda() {
    impl_->arch = "cuda";
    return *this;
  };
  std::string arch() { return impl_->arch; }; */
};

}  // namespace tinytorch