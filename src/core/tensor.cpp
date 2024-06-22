/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#include "tensor.hpp"

#include "graph.hpp"
#include "ops.hpp"

namespace microtorch {

TensorImpl::TensorImpl(const IntArrayRef& shape, Device device,
                       bool requires_grad, const data_t* data, int64_t offset)
    : shape_(shape),
      stride_(calculate_init_stride(shape)),
      storage_(Storage(shape_.numel() * sizeof(data_t), device, data)),
      requires_grad_(requires_grad),
      offset_(offset) {}

TensorImpl::TensorImpl(const Storage& storage, const IntArrayRef& shape,
                       IntMArrayRef stride, Device device,
                       bool requires_grad, int64_t offset)
    : shape_(shape),
      stride_(stride),
      storage_(storage),
      requires_grad_(requires_grad),
      offset_(offset) {}

data_t& TensorImpl::operator[](const IntArrayRef& idxs) {
  TORCH_CHECK(
      ndim() == idxs.size(),
      "Indices size must equal tensor's dimensions. Expect indices to be ndim ",
      ndim(), ", but got ", idxs.size());
  int64_t offset = offset_;
  for (int64_t i = 0; i < ndim(); i++) {
    int64_t index = idxs[i];
    if (index < 0) {
      index += shape_[i];
    }
    TORCH_CHECK(0 <= index && index < shape_[i], "Index ", idxs[i],
                " is out of bounds for dimension ", i, " with size ",
                shape_[i]);
    offset += index * stride_[i];
  }
  return storage_[offset];
}

data_t TensorImpl::operator[](const IntArrayRef& idxs) const {
  TORCH_CHECK(ndim() == idxs.size(),
              "Indices size must equal tensor's dimensions.");
  int64_t offset = offset_;
  for (int64_t i = 0; i < ndim(); i++) {
    int64_t index = idxs[i];
    if (index < 0) {
      index += shape_[i];
    }
    TORCH_CHECK(0 <= index && index < shape_[i], "Index ", idxs[i],
                " is out of bounds for dimension ", i, " with size ",
                shape_[i]);
    offset += index * stride_[i];
  }
  return storage_[offset];
}

bool TensorImpl::is_contiguous() const {
  int64_t stride = 1;
  for (int i = ndim() - 1; i >= 0; i--) {
    if (stride_[i] != stride) {
      return false;
    }
    stride *= shape_[i];
  }
  return true;
}

data_t TensorImpl::item() const {
  TORCH_CHECK(numel() == 1,
              "item() can be called only on scalar (1-element) tensors.");
  return storage_[0];
}

Tensor::Tensor(const IntArrayRef& shape, Device device, bool requires_grad) {
  impl_ = std::make_shared<TensorImpl>(shape, Device(device), requires_grad);
  if (requires_grad) {
    impl_->set_edge(
        std::make_shared<Edge>(std::make_shared<AccumulateGrad>(*this), 0));
  }
}

Tensor::Tensor(std::vector<data_t> data, Device device, bool requires_grad) {
  IntArrayRef shape = {static_cast<int64_t>(data.size())};
  impl_ = std::make_shared<TensorImpl>(shape, Device(device), requires_grad,
                                       data.data());
  if (requires_grad) {
    impl_->set_edge(
        std::make_shared<Edge>(std::make_shared<AccumulateGrad>(*this), 0));
  }
}

Tensor Tensor::cuda() {
#ifdef USE_CUDA
  if (this->device().is_cuda()) {
    return *this;
  }
  Tensor t = Tensor(this->shape(), Device("cuda"), this->requires_grad());
  cudaMemcpy(t.data_ptr(), this->data_ptr(), impl_->nbytes(),
             cudaMemcpyHostToDevice);
  return t;
#else
  TORCH_CHECK(false, "MicroTorch not compiled with CUDA enabled");
#endif
}

Tensor Tensor::cpu() {
#ifdef USE_CUDA
  if (this->device().is_cuda()) {
    Tensor t = Tensor(this->shape(), Device("cpu"), this->requires_grad());
    cudaMemcpy(t.data_ptr(), this->data_ptr(), impl_->nbytes(),
               cudaMemcpyDeviceToHost);
    return t;
  }
#endif
  return *this;
}

Tensor& Tensor::operator=(const Tensor& other) {
  if (&other == this) {
    return *this;  // handle self-assignment
  }
  impl_ = other.impl_;
  return *this;
}

Tensor Tensor::operator==(const Tensor& other) const {
  TORCH_CHECK(other.device() == this->device(), "device should be the same");
  TORCH_CHECK(other.shape() == this->shape(), "shape should be the same");
  Tensor out = zeros(this->shape(), this->device());
  DISPATCH_OP(eq_impl, this->device(), *this, other, out);
  return out;
}

Tensor& Tensor::zero_() { return this->fill_(0); }

Tensor Tensor::square() { return microtorch::square(*this); }

bool Tensor::equal(const Tensor other) const {
  if (this->numel() != other.numel() || this->shape() != other.shape() ||
      this->device() != other.device()) {
    return false;
  }
  return this->numel() ==
         static_cast<int64_t>(microtorch::sum(*this == other).item());
}

Tensor& Tensor::fill_(const data_t value) {
  fill_scalar(*this, value);
  return *this;
}

std::string Tensor::str() const { return print_with_size(*this); }
void Tensor::backward() { ::microtorch::backward(*this); }

Tensor Tensor::as_strided(const IntArrayRef& shape, IntMArrayRef stride,
                          int64_t offset) const {
  auto new_impl = std::make_shared<TensorImpl>(impl_->storage(), shape, stride,
                                               impl_->device(),
                                               impl_->requires_grad(), offset);
  return Tensor(new_impl);
}

}  // namespace microtorch
