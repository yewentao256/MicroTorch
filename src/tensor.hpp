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

struct Edge;

struct TensorImpl {
 public:
  std::vector<data_t> grad_;  // for .backward
  std::shared_ptr<Edge> edge_;
  std::string arch_ = "host";

  // constructors
  TensorImpl(size_t size, std::vector<size_t> shape)
      : data_(size), ndim_(shape.size()), shape_(shape), stride_(ndim_) {
    init_stride();
  }
  TensorImpl(std::vector<data_t> data, std::vector<size_t> shape)
      : data_(data), ndim_(shape.size()), shape_(shape), stride_(ndim_) {
    init_stride();
  }

  // properties
  size_t offset() { return offset_; }
  size_t ndim() { return ndim_; }
  const std::vector<size_t>& stride() { return stride_; }
  std::vector<data_t>& data() {return data_;}

  // funcs
  data_t& operator[](std::vector<size_t> idxs) {
    assert(ndim_ == idxs.size());
    size_t offset = offset_;
    for (size_t i = 0; i < ndim_; i++) {
      offset += idxs[i] * stride_[i];
    }
    return data_[offset];
  }

  const std::shared_ptr<TensorImpl> transpose(size_t dim_0, size_t dim_1) {
    // assert(dim_0 < ndim_ && dim_1 < ndim_);
    // // TODO: 交换shape和stride
    // TensorImpl new_tensor(this->data());  // 小心！注意观察是否两个data是同一份（按照pytorch，两个是同一份data）
    // return std::make_shared<TensorImpl>(size, shape)
  } 

 private:
  std::vector<data_t> data_;
  size_t offset_ = 0;
  size_t ndim_;
  std::vector<size_t> shape_;  // TODO: 之后支持多维
  std::vector<size_t> stride_;

  // funcs
  void init_stride() {
    size_t stride = 1;
    for (int i = ndim_ - 1; i >= 0; i--) {
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
  Tensor(size_t size = 0) {
    std::vector<size_t> shape = {size};
    impl_ = std::make_shared<TensorImpl>(size, shape);
  }
  Tensor(std::vector<data_t> data) {
    std::vector<size_t> shape = {data.size()};
    impl_ = std::make_shared<TensorImpl>(data, shape);
  }

  // operator override
  data_t& operator[](size_t idx) { return impl_->operator[]({idx}); }

  // properties
  size_t offset() { return impl_->offset(); }
  size_t ndim() { return impl_->ndim(); }
  const std::vector<size_t>& stride() { return impl_->stride(); }
  std::vector<data_t>& data() { return impl_->data(); }

  // tensor functions
  const Tensor transpose(size_t dim_0, size_t dim_1) {
    // return Tensor(impl_->transpose());
  }

  size_t size() { return impl_->data().size(); }
  void resize(size_t size) {
    assert(impl_);
    impl_->data().resize(size, 0);
    impl_->grad_.resize(size, 0);
  }
  void clearGrad() { impl_->grad_.clear(); }

  Tensor grad() { return Tensor(impl_->grad_); }

  void addInplace(Tensor t) {
    resize(t.size());
    for (size_t i = 0; i < size(); i++) {
      impl_->data()[i] += t[i];
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

  data_t* data_ptr() { return impl_->data().data(); };

  Tensor cuda() {
    impl_->arch_ = "cuda";
    return *this;
  };
  std::string arch() { return impl_->arch_; };
};

}  // namespace tinytorch