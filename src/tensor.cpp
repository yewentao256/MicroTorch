// tinytorch.cpp
#include "tensor.hpp"

#include "backward.hpp"
#include "ops.hpp"

namespace tinytorch {

TensorImpl::TensorImpl(std::vector<size_t>& shape, Device device,
                       bool requires_grad, const data_t* data)
    : shape_(shape),
      stride_(shape_.size()),
      numel_(std::accumulate(shape.begin(), shape.end(), 1,
                             std::multiplies<unsigned long>())),
      storage_(Storage(numel_ * sizeof(data_t), device, data)),
      requires_grad_(requires_grad) {
  size_t stride = 1;
  for (int i = ndim() - 1; i >= 0; i--) {
    stride_[i] = stride;
    stride *= shape_[i];
  }
}

TensorImpl::TensorImpl(const Storage& storage, std::vector<size_t>& shape,
                       std::vector<size_t> stride, Device device,
                       bool requires_grad)
    : shape_(shape),
      stride_(stride),
      numel_(std::accumulate(shape.begin(), shape.end(), 1,
                             std::multiplies<unsigned long>())),
      storage_(storage),
      requires_grad_(requires_grad) {}

data_t& TensorImpl::operator[](std::vector<size_t> idxs) {
  // this is for updating tensor value
  TORCH_CHECK(ndim() == idxs.size(), "idxs size should equal to tensor's ndim");
  size_t offset = offset_;
  for (size_t i = 0; i < ndim(); i++) {
    offset += idxs[i] * stride_[i];
  }
  return storage_[offset];
}

data_t TensorImpl::operator[](std::vector<size_t> idxs) const {
  // this is for getting value
  TORCH_CHECK(ndim() == idxs.size(), "idxs size should equal to tensor's ndim");
  size_t offset = offset_;
  for (size_t i = 0; i < ndim(); i++) {
    offset += idxs[i] * stride_[i];
  }
  return storage_[offset];
}

bool TensorImpl::is_contiguous() const {
  size_t stride = 1;
  for (int i = ndim() - 1; i >= 0; i--) {
    if (stride_[i] != stride) {
      return false;
    }
    stride *= shape_[i];
  }
  return true;
}

Tensor::Tensor(std::vector<size_t> shape, Device device, bool requires_grad) {
  impl_ = std::make_shared<TensorImpl>(shape, Device(device), requires_grad);
  if (requires_grad) {
    impl_->set_edge(
        std::make_shared<Edge>(std::make_shared<AccumulateGrad>(*this), 0));
  }
}

Tensor::Tensor(std::vector<data_t> data, Device device, bool requires_grad) {
  std::vector<size_t> shape = {data.size()};
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
  TORCH_CHECK(false, "TinyTorch not compiled with CUDA enabled");
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

Tensor Tensor::operator+(const Tensor& other) {
  Tensor out = zeros(this->shape(), this->device());
  add_out(*this, other, out);
  return out;
}

Tensor& Tensor::operator+=(const Tensor& other) {
  add_out(*this, other, *this);
  return *this;
}

Tensor Tensor::operator-(const Tensor& other) {
  Tensor out = zeros(this->shape(), this->device());
  sub_out(*this, other, out);
  return out;
}

Tensor& Tensor::operator-=(const Tensor& other) {
  sub_out(*this, other, *this);
  return *this;
}

Tensor Tensor::operator*(const Tensor& other) {
  Tensor out = zeros(this->shape(), this->device());
  mul_out(*this, other, out);
  return out;
}
Tensor Tensor::operator*(const data_t& other) {
  Tensor out = zeros(this->shape(), this->device());
  // TODO: to realize a mul kernel with scalar is better
  Tensor t = zeros(this->shape(), this->device());
  t.fill_(other);
  mul_out(*this, t, out);
  return out;
}

Tensor& Tensor::operator*=(const Tensor& other) {
  mul_out(*this, other, *this);
  return *this;
}
Tensor& Tensor::operator*=(const data_t& other) {
  Tensor t = zeros(this->shape(), this->device());
  t.fill_(other);
  mul_out(*this, t, *this);
  return *this;
}

Tensor& Tensor::operator=(const Tensor& other) {
  if (&other == this) {
    return *this;  // handle self-assignment
  }
  impl_ = other.impl_;
  return *this;
}

Tensor& Tensor::zero_() { return fill_(0); }

Tensor& Tensor::fill_(data_t value) {
  fill_scalar(*this, value);
  return *this;
}

std::string Tensor::str() { return print_with_size(*this); }
void Tensor::backward() { ::tinytorch::backward(*this); }

}  // namespace tinytorch
