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
 private:

  data_t* data_;
  size_t offset_ = 0;
  size_t ndim_;
  std::vector<size_t> shape_;  // TODO: 之后支持多维
  std::vector<size_t> stride_;
  bool need_delete_ = false;

  // funcs
  void init_stride() {
    size_t stride = 1;
    for (int i = ndim_ - 1; i >= 0; i--) {
      stride_[i] = stride;
      stride *= shape_[i];
    }
  }

 public:
  std::vector<data_t> grad_;  // for backward
  std::shared_ptr<Edge> edge_;
  std::string arch_;

  // constructors
  TensorImpl(std::vector<size_t> shape, std::string arch)
      : data_(new data_t[shape[0]]),
        ndim_(shape.size()),
        shape_(shape),
        stride_(ndim_),
        arch_(arch) {
    for (int i = 0; i < shape[0]; i++) {
      data_[i] = 0;  // TODO: remove this deep copy
    }

    need_delete_ = true;
    init_stride();
  }
  TensorImpl(data_t* data, std::vector<size_t> shape, std::string arch)
      : data_(new data_t[shape[0]]),
        ndim_(shape.size()),
        shape_(shape),
        stride_(ndim_),
        arch_(arch) {
    init_stride();
    for (int i = 0; i < shape[0]; i++) {
      data_[i] = data[i];
    }
  }
  TensorImpl(data_t* data, std::vector<size_t> shape,
             std::vector<size_t> stride)
      : data_(new data_t[shape[0]]),
        ndim_(shape.size()),
        shape_(shape),
        stride_(stride) {
    for (int i = 0; i < shape[0]; i++) {
      data_[i] = data[i];
    }
  }

  ~TensorImpl() {
    if (need_delete_) {
      // TODO: new way to construct tensor then remove this
      delete[] data_;
    }
  }

  // properties
  size_t offset() { return offset_; }
  size_t ndim() { return ndim_; }
  const std::vector<size_t>& stride() { return stride_; }
  const std::vector<size_t>& size() { return shape_; }
  data_t* data() { return data_; }

  // operator override
  data_t& operator[](std::vector<size_t> idxs) {
    assert(ndim_ == idxs.size());
    size_t offset = offset_;
    for (size_t i = 0; i < ndim_; i++) {
      offset += idxs[i] * stride_[i];
    }
    return data_[offset];
  }
  data_t operator[](std::vector<size_t> idxs) const {
    assert(ndim_ == idxs.size());
    size_t offset = offset_;
    for (size_t i = 0; i < ndim_; i++) {
      offset += idxs[i] * stride_[i];
    }
    return data_[offset];
  }

  // funcs
  const std::shared_ptr<TensorImpl> transpose(size_t dim0, size_t dim1) {
    assert(dim0 < ndim_ && dim1 < ndim_);
    std::vector<size_t> shape(shape_);  // deep copy
    shape[dim0] = shape_[dim1];
    shape[dim1] = shape_[dim0];

    std::vector<size_t> stride(stride_);  // deep copy
    stride[dim0] = stride_[dim1];
    stride[dim1] = stride_[dim0];

    return std::make_shared<TensorImpl>(data_, shape, stride);
  }

  const std::shared_ptr<TensorImpl> permute(std::vector<size_t> dims) {
    assert(dims.size() == ndim_);
    std::vector<size_t> shape(ndim_);
    std::vector<size_t> stride(ndim_);
    for (size_t i = 0; i < ndim_; i++) {
      shape[i] = shape_[dims[i]];
      stride[i] = stride_[dims[i]];
    }
    return std::make_shared<TensorImpl>(data_, shape, stride);
  }

  bool is_contiguous() const {
    size_t stride = 1;
    for (int i = ndim_ - 1; i >= 0; i--) {
      if (stride_[i] != stride) {
        return false;
      }
      stride *= shape_[i];
    }
    return true;
  }
};

struct Tensor {
 private:
  std::shared_ptr<TensorImpl> impl_;

 public:
  // constructors
  Tensor(size_t size = 0, std::string arch = "host") {
    std::vector<size_t> shape = {size};
    impl_ = std::make_shared<TensorImpl>(shape, arch);
  }
  // Tensor(std::vector<data_t> data, std::string arch = "host") {
  //   std::vector<size_t> shape = {data.size()};
  //   impl_ = std::make_shared<TensorImpl>(data.data(), shape, arch);
  // }
  Tensor(std::shared_ptr<TensorImpl> impl) : impl_(impl) {}

  // operator override
  data_t& operator[](size_t idx) { return impl_->operator[]({idx}); }
  data_t& operator[](std::vector<size_t> idxs) {
    return impl_->operator[](idxs);
  }
  data_t operator[](size_t idx) const { return impl_->operator[]({idx}); }
  data_t operator[](std::vector<size_t> idxs) const {
    return impl_->operator[](idxs);
  }

  // properties
  size_t offset() { return impl_->offset(); }
  size_t ndim() { return impl_->ndim(); }
  const std::vector<size_t>& stride() { return impl_->stride(); }
  data_t* data_ptr() { return impl_->data(); };
  const std::shared_ptr<TensorImpl>& impl() const { return impl_; }

  // tensor functions
  const Tensor transpose(size_t dim0, size_t dim1) {
    return Tensor(impl_->transpose(dim0, dim1));
  }
  const Tensor permute(std::vector<size_t> dims) {
    return Tensor(impl_->permute(dims));
  }

  bool is_contiguous() const { return impl_->is_contiguous(); }

  size_t size() { return impl_->size()[0]; }  // TODO: multi dimension
  void resize(size_t size) {
    assert(impl_);
    // impl_->data().resize(size, 0);
    impl_->grad_.resize(size, 0);
  }
  void clearGrad() { impl_->grad_.clear(); }

  std::vector<data_t> grad() { return impl_->grad_; }

  void addInplace(Tensor t) {
    assert(t.size() == size());
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

  Tensor cuda() {
    impl_->arch_ = "cuda";
    return *this;
  };
  std::string arch() { return impl_->arch_; };
};

}  // namespace tinytorch