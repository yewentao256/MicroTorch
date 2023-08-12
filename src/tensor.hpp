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
  std::vector<size_t> shape_;
  std::vector<size_t> stride_;
  size_t numel_;
  Storage storage_;

  bool requires_grad_;
  std::unique_ptr<Tensor> grad_ = nullptr;

 public:
  std::shared_ptr<Edge> edge_;

  // constructors
  explicit TensorImpl(std::vector<size_t>& shape, Device device,
                      bool requires_grad, const data_t* data = nullptr);
  explicit TensorImpl(const Storage& storage, std::vector<size_t>& shape,
                      std::vector<size_t> stride, Device device,
                      bool requires_grad);

  ~TensorImpl() = default;

  // properties
  size_t offset() { return offset_; }
  size_t ndim() const { return shape_.size(); }
  const std::vector<size_t>& stride() { return stride_; }
  const std::vector<size_t>& shape() { return shape_; }
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
  data_t& operator[](std::vector<size_t> idxs);
  data_t operator[](std::vector<size_t> idxs) const;

  bool is_contiguous() const;
};

struct Tensor {
 private:
  std::shared_ptr<TensorImpl> impl_ = nullptr;

 public:
  // constructors
  Tensor() {}
  explicit Tensor(std::vector<size_t> shape, Device device = Device("cpu"),
                  bool requires_grad = true) {
    impl_ = std::make_shared<TensorImpl>(shape, Device(device), requires_grad);
  }
  explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(impl) {}
  explicit Tensor(std::vector<data_t> data, Device device = Device("cpu"),
                  bool requires_grad = true) {
    std::vector<size_t> shape = {data.size()};
    impl_ = std::make_shared<TensorImpl>(shape, Device(device), requires_grad,
                                         data.data());
  }
  Tensor(const Tensor& other) : impl_(other.impl()) {}
  Tensor(Tensor&& other) noexcept : impl_(std::move(other.impl_)) {}

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

  bool is_contiguous() const { return impl_->is_contiguous(); }

  const std::vector<size_t>& shape() { return impl_->shape(); }

  Tensor& grad() { return impl_->grad(); }
  bool requires_grad() const { return impl_->requires_grad(); }
  Tensor& zero_();
  Tensor& fill_(data_t value);

  // overwrite ops
  Tensor operator+(const Tensor& other);
  Tensor& operator+=(const Tensor& other);
  Tensor operator-(const Tensor& other);
  Tensor& operator-=(const Tensor& other);
  Tensor operator*(const Tensor& other);
  Tensor& operator*=(const Tensor& other);
  Tensor& operator=(const Tensor& other);

  std::shared_ptr<Edge> get_edge() { return impl_->edge_; };
  void set_edge(std::shared_ptr<Edge> edge) { impl_->edge_ = edge; };

  Tensor cuda();
  bool is_cuda() const { return impl_->device().is_cuda(); }
  Tensor cpu();

  Device device() { return impl_->device(); };
};

}  // namespace tinytorch