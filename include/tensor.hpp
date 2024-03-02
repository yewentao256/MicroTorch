/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#pragma once

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>

#include "array.hpp"
#include "storage.hpp"

namespace microtorch {

struct Edge;
struct Tensor;

struct TensorImpl {
 private:
  IntArrayRef shape_;
  IntArrayRef stride_;
  Storage storage_;

  bool requires_grad_;
  int64_t offset_ = 0;
  std::shared_ptr<Edge> edge_ = nullptr;

 public:
  std::unique_ptr<Tensor> grad_ = nullptr;

  // constructors
  explicit TensorImpl(const IntArrayRef& shape, Device device,
                      bool requires_grad, const data_t* data = nullptr,
                      int64_t offset = 0);
  explicit TensorImpl(const Storage& storage, const IntArrayRef& shape,
                      const IntArrayRef& stride, Device device,
                      bool requires_grad, int64_t offset = 0);

  ~TensorImpl() = default;

  // properties
  int64_t offset() const { return offset_; }
  int64_t ndim() const { return shape_.size(); }
  const IntArrayRef& stride() { return stride_; }
  const IntArrayRef& shape() { return shape_; }
  data_t* data() { return storage_.data(); }
  Device device() const { return storage_.device(); }
  int64_t nbytes() const { return storage_.nbytes(); }
  int64_t numel() const { return shape_.numel(); }
  bool requires_grad() const { return requires_grad_; }
  void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }
  std::shared_ptr<Edge> edge() const { return edge_; }
  void set_edge(std::shared_ptr<Edge> edge) { edge_ = edge; }

  // operator override
  data_t& operator[](const IntArrayRef& idxs);
  data_t operator[](const IntArrayRef& idxs) const;

  bool is_contiguous() const;
  data_t item() const;

  const Storage& storage() const { return storage_; }
};

struct Tensor {
 private:
  std::shared_ptr<TensorImpl> impl_ = nullptr;

 public:
  // constructors
  Tensor() {}
  explicit Tensor(const IntArrayRef& shape, Device device = Device("cpu"),
                  bool requires_grad = false);
  explicit Tensor(std::vector<data_t> data, Device device = Device("cpu"),
                  bool requires_grad = false);
  explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(impl) {}
  Tensor(const Tensor& other) : impl_(other.impl()) {}
  Tensor(Tensor&& other) noexcept : impl_(std::move(other.impl_)) {}

  // operator override
  data_t& operator[](int64_t idx) { return impl_->operator[]({idx}); }
  data_t& operator[](const IntArrayRef& idxs) {
    return impl_->operator[](idxs);
  }
  data_t operator[](int64_t idx) const {
    return static_cast<const TensorImpl*>(impl_.get())->operator[]({idx});
  }

  data_t operator[](const IntArrayRef& idxs) const {
    return static_cast<const TensorImpl*>(impl_.get())->operator[](idxs);
  }

  // properties
  int64_t offset() const { return impl_->offset(); }
  int64_t ndim() const { return impl_->ndim(); }
  const IntArrayRef& stride() const { return impl_->stride(); }
  data_t* data_ptr() const { return impl_->data(); };
  const std::shared_ptr<TensorImpl>& impl() const { return impl_; }
  int64_t numel() const { return impl_->numel(); }
  bool defined() const { return impl_ && this->numel() > 0; }

  bool is_contiguous() const { return impl_->is_contiguous(); }

  const IntArrayRef& shape() const { return impl_->shape(); }

  Tensor grad() {
    if (impl_->grad_) {
      return *(impl_->grad_);
    }
    return Tensor();
  }
  void set_grad(Tensor grad) { impl_->grad_ = std::make_unique<Tensor>(grad); }
  bool requires_grad() const { return impl_->requires_grad(); }
  void set_requires_grad(bool requires_grad) {
    impl_->set_requires_grad(requires_grad);
  }
  Tensor& zero_();
  Tensor& fill_(const data_t value);
  Tensor square();
  bool equal(const Tensor other) const;

  // overwrite ops
  Tensor operator-() const;
  Tensor operator+(const Tensor& other) const;
  Tensor& operator+=(const Tensor& other);
  Tensor operator-(const Tensor& other) const;
  Tensor& operator-=(const Tensor& other);
  Tensor operator*(const Tensor& other) const;
  Tensor operator*(const data_t other) const;
  Tensor& operator*=(const Tensor& other);
  Tensor& operator*=(const data_t other);
  Tensor operator/(const Tensor& other) const;
  Tensor& operator/=(const Tensor& other);
  Tensor& operator=(const Tensor& other);
  Tensor operator==(const Tensor& other) const;

  std::shared_ptr<Edge> edge() const { return impl_->edge(); };
  void set_edge(std::shared_ptr<Edge> edge) { impl_->set_edge(edge); };

  Tensor cuda();
  bool is_cuda() const { return impl_->device().is_cuda(); }
  Tensor cpu();

  std::string str() const;
  Tensor clone() const;
  void backward();

  Device device() const { return impl_->device(); };
  data_t item() const { return impl_->item(); }
  int64_t element_size() const { return sizeof(data_t); }

  // Note: you need to be careful when you this function to view a tensor.
  Tensor as_strided(const IntArrayRef& shape, const IntArrayRef& stride,
                    int64_t offset = 0) const;
};

}  // namespace microtorch