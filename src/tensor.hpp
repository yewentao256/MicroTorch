#pragma once

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

#include "storage.hpp"

namespace tinytorch {

struct Edge;

struct TensorImpl {
 private:
  Storage storage_;
  size_t offset_ = 0;          // TODO: remove this(storage has it)
  std::vector<size_t> shape_;  // TODO: 之后支持多维
  std::vector<size_t> stride_;

  // funcs
  void init_stride() {
    size_t stride = 1;
    for (int i = ndim() - 1; i >= 0; i--) {
      stride_[i] = stride;
      stride *= shape_[i];
    }
  }

 public:
  std::vector<data_t> grad_;  // for backward
  std::shared_ptr<Edge> edge_;

  // constructors
  TensorImpl(const Storage& storage, std::vector<size_t>& shape)
      : storage_(storage), shape_(shape), stride_(shape_.size()) {
    init_stride();
  }
  TensorImpl(const Storage& storage, std::vector<size_t>& shape,
             std::vector<size_t> stride)
      : storage_(storage), shape_(shape), stride_(stride) {}
  TensorImpl(std::vector<size_t> shape, std::string arch = "cpu")
      : TensorImpl(Storage(std::accumulate(shape.begin(), shape.end(), 0),
                           Device(arch)),
                   shape) {}
  TensorImpl(data_t* data, std::vector<size_t> shape, std::string arch = "cpu")
      : TensorImpl(Storage(data, std::accumulate(shape.begin(), shape.end(), 0),
                           Device(arch)),
                   shape) {}
  TensorImpl(data_t* data, std::vector<size_t> shape,
             std::vector<size_t> stride, std::string arch = "cpu")
      : TensorImpl(Storage(data, std::accumulate(shape.begin(), shape.end(), 0),
                           Device(arch)),
                   shape, stride) {}

  ~TensorImpl() {}

  // properties
  size_t offset() { return offset_; }
  size_t ndim() const { return shape_.size(); }
  const std::vector<size_t>& stride() { return stride_; }
  const std::vector<size_t>& size() { return shape_; }
  data_t* data() { return storage_.data(); }
  std::string arch() { return storage_.device().str(); }
  size_t nbytes() const { return storage_.size(); }

  // operator override
  data_t& operator[](std::vector<size_t> idxs) {
    assert(ndim() == idxs.size());
    size_t offset = offset_;
    for (size_t i = 0; i < ndim(); i++) {
      offset += idxs[i] * stride_[i];
    }
    return storage_[offset];
  }
  data_t operator[](std::vector<size_t> idxs) const {
    assert(ndim() == idxs.size());
    size_t offset = offset_;
    for (size_t i = 0; i < ndim(); i++) {
      offset += idxs[i] * stride_[i];
    }
    return storage_[offset];
  }

  // funcs
  const std::shared_ptr<TensorImpl> transpose(size_t dim0, size_t dim1) {
    assert(dim0 < ndim() && dim1 < ndim());
    std::vector<size_t> shape(shape_);  // deep copy
    shape[dim0] = shape_[dim1];
    shape[dim1] = shape_[dim0];

    std::vector<size_t> stride(stride_);  // deep copy
    stride[dim0] = stride_[dim1];
    stride[dim1] = stride_[dim0];

    return std::make_shared<TensorImpl>(storage_, shape, stride);
  }

  const std::shared_ptr<TensorImpl> permute(std::vector<size_t> dims) {
    assert(dims.size() == ndim());
    std::vector<size_t> shape(ndim());
    std::vector<size_t> stride(ndim());
    for (size_t i = 0; i < ndim(); i++) {
      shape[i] = shape_[dims[i]];
      stride[i] = stride_[dims[i]];
    }
    return std::make_shared<TensorImpl>(storage_, shape, stride);
  }

  bool is_contiguous() const {
    size_t stride = 1;
    for (int i = ndim() - 1; i >= 0; i--) {
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
  Tensor(size_t size = 0, std::string arch = "cpu") {
    std::vector<size_t> shape = {size};
    impl_ = std::make_shared<TensorImpl>(shape, arch);
  }
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
    // TODO: data copy
    return Tensor(impl_->nbytes(), "cuda");
  };

  std::string arch() { return impl_->arch(); };
};

}  // namespace tinytorch