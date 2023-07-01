#pragma once

#include <algorithm>
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
struct Tensor;

struct TensorImpl {
 private:
  size_t offset_ = 0;
  std::vector<size_t> shape_;  // TODO: 之后支持多维
  std::vector<size_t> stride_;
  size_t numel_;
  Storage storage_;  // TODO: 这里是否应该用指针？

  bool requires_grad_;
  // TODO: 使用autogradmeta来支持version和next_fn，function等？好像已经是下面edge_的功能了
  std::unique_ptr<Tensor> grad_ = nullptr;

 public:
  std::shared_ptr<Edge> edge_;

  // constructors
  TensorImpl(std::vector<size_t>& shape, Device device, bool requires_grad)
      : shape_(shape),
        stride_(shape_.size()),
        numel_(std::accumulate(shape.begin(), shape.end(), 0)),
        storage_(Storage(numel_ * sizeof(data_t), device)),
        requires_grad_(requires_grad) {
    size_t stride = 1;
    for (int i = ndim() - 1; i >= 0; i--) {
      stride_[i] = stride;
      stride *= shape_[i];
    }
    if (requires_grad_) {
      grad_ = std::make_unique<Tensor>(numel_, device, false);
    }
  }
  TensorImpl(const Storage& storage, std::vector<size_t>& shape,
             std::vector<size_t> stride, Device device, bool requires_grad)
      : shape_(shape),
        stride_(stride),
        numel_(std::accumulate(shape.begin(), shape.end(), 0)),
        storage_(storage),
        requires_grad_(requires_grad) {
    if (requires_grad_) {
      grad_ = std::make_unique<Tensor>(numel_, device, false);
    }
  }

  ~TensorImpl() {}  // TODO:这里析构什么都不做？check一下

  // properties
  size_t offset() { return offset_; }
  size_t ndim() const { return shape_.size(); }
  const std::vector<size_t>& stride() { return stride_; }
  const std::vector<size_t>& size() { return shape_; }
  data_t* data() { return storage_.data(); }
  Device device() const { return storage_.device(); }
  size_t nbytes() const { return storage_.nbytes(); }
  size_t numel() const { return numel_; }
  bool requires_grad() const { return requires_grad_; }
  Tensor& grad() {
    TORCH_CHECK(requires_grad_, "tensor should be in requires_grad mode");
    return *grad_;
  }

  // operator override
  data_t& operator[](std::vector<size_t> idxs) {
    // this is for updating tensor value? TODO: test this
    // TODO(low priority): supprot this.
    TORCH_CHECK(device().is_cpu(),
                "we do not support setting value for cuda tensor currently.");
    TORCH_CHECK(ndim() == idxs.size(),
                "idxs size should equal to tensor's ndim");
    size_t offset = offset_;
    for (size_t i = 0; i < ndim(); i++) {
      offset += idxs[i] * stride_[i];
    }
    return storage_[offset];
  }
  // data_t operator[](std::vector<size_t> idxs) const {
  //   // this is for getting value
  //   TORCH_CHECK(ndim() == idxs.size(),
  //               "idxs size should equal to tensor's ndim");
  //   TORCH_CHECK(device() != "cuda",
  //               "we do not support getting value from cuda tensor
  //               currently.");
  //   size_t offset = offset_;
  //   for (size_t i = 0; i < ndim(); i++) {
  //     offset += idxs[i] * stride_[i];
  //   }
  //   return storage_[offset];
  // }

  // funcs
  // const std::shared_ptr<TensorImpl> transpose(size_t dim0, size_t dim1) {
  //   // view op
  //   TORCH_CHECK(dim0 < ndim() && dim1 < ndim(),
  //               "transpose dim should not bigger than tensor's dim");
  //   std::vector<size_t> shape(shape_);  // deep copy
  //   shape[dim0] = shape_[dim1];
  //   shape[dim1] = shape_[dim0];

  //   std::vector<size_t> stride(stride_);  // deep copy
  //   stride[dim0] = stride_[dim1];
  //   stride[dim1] = stride_[dim0];
  // TODO: 找一个更合适的方式，似乎走impl重定义不是很好
  //   return std::make_shared<TensorImpl>(storage_, shape, stride);
  // }

  // const std::shared_ptr<TensorImpl> permute(std::vector<size_t> dims) {
  //   // view op
  //   TORCH_CHECK(dims.size() == ndim(),
  //               "permute dims size should equal to tensor's dim");
  //   std::vector<size_t> shape(ndim());
  //   std::vector<size_t> stride(ndim());
  //   for (size_t i = 0; i < ndim(); i++) {
  //     shape[i] = shape_[dims[i]];
  //     stride[i] = stride_[dims[i]];
  //   }
  //     // TODO: 找一个更合适的方式，似乎走impl重定义不是很好
  //   return std::make_shared<TensorImpl>(storage_, shape, stride);
  // }

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
  Tensor(size_t size = 0, Device device = Device("cpu"),
         bool requires_grad = true) {
    // TODO: 这里应该传进来shape而不是size，用于支持多d创建
    // TODO:
    // 现在所有tensor默认都会初始化一个impl_，这是没有必要的，当一些undefine的初始化时如vector<Tensor>创建的应该是未定义的tensor，然后引入defiend方法
    std::vector<size_t> shape = {size};
    impl_ = std::make_shared<TensorImpl>(shape, Device(device), requires_grad);
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
  size_t numel() { return impl_->numel(); }
  bool defined() { return impl_ && this->numel() > 0; }

  // tensor functions
  // const Tensor transpose(size_t dim0, size_t dim1) {
  //   return Tensor(impl_->transpose(dim0, dim1));
  // }
  // const Tensor permute(std::vector<size_t> dims) {
  //   return Tensor(impl_->permute(dims));
  // }

  bool is_contiguous() const { return impl_->is_contiguous(); }

  size_t size() { return impl_->size()[0]; }  // TODO: multi dimension

  Tensor& grad() { return impl_->grad(); }
  Tensor& zero_() {
    // TODO: make this a op, supporting cuda and cpu
    for (size_t i = 0; i < numel(); i++) {
      (*this)[i] = 0;
    }
    return *this;
  }

  // Pytorch tensorbody里可以调用add_的原因：gen
  // data生成了大量inline代码，里面有add_方法，随后经过dispatch调度到实际的add_op上
  Tensor operator+(const Tensor& other);
  Tensor& operator+=(const Tensor& other);
  Tensor operator-(const Tensor& other);
  Tensor& operator-=(const Tensor& other);

  std::shared_ptr<Edge> get_edge() { return impl_->edge_; };
  void set_edge(std::shared_ptr<Edge> edge) { impl_->edge_ = edge; };

  Tensor cuda() {
#ifdef USE_CUDA
    if (this->device() == "cuda") {
      return *this;
    }
    Tensor t = Tensor(impl_->numel(), "cuda");
    // TODO: move to copy.cu
    cudaMemcpy(t.data_ptr(), this->data_ptr(), impl_->nbytes(),
               cudaMemcpyHostToDevice);
    return t;
#else
    TORCH_CHECK(false, "TinyTorch not compiled with CUDA enabled");
#endif
  };

  Tensor cpu() {
#ifdef USE_CUDA
    if (this->device() == "cuda") {
      Tensor t = Tensor(impl_->numel(), "cpu");
      cudaMemcpy(t.data_ptr(), this->data_ptr(), impl_->nbytes(),
                 cudaMemcpyDeviceToHost);
      return t;
    }
#endif
    return *this;
  };

  Device device() { return impl_->device(); };
};

}  // namespace tinytorch