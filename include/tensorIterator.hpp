/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#pragma once
#include <algorithm>

#include "array.hpp"
#include "exception.hpp"
#include "funcRef.hpp"
#include "irange.hpp"
#include "tensor.hpp"

namespace microtorch {

constexpr int64_t GRAIN_SIZE = 32768;

struct OperandInfo {
  OperandInfo() = default;
  explicit OperandInfo(Tensor&& t) { tensor(std::move(t)); }
  explicit OperandInfo(const Tensor& t) { tensor_ = t; }

  OperandInfo(const OperandInfo&) = default;
  OperandInfo& operator=(const OperandInfo&) = default;
  OperandInfo(OperandInfo&&) noexcept = default;
  OperandInfo& operator=(OperandInfo&&) noexcept = default;
  ~OperandInfo() = default;

  // The data pointer. This may be different from tensor->data_ptr() if the
  // iterator is split.
  void* data = nullptr;

  // Stride after broadcasting. The stride is in bytes, not number of elements.
  IntArrayRef stride_bytes;
  bool is_output = false;
  bool will_resize = false;
  bool is_read_write = false;

  // The tensor operand. Note that the strides, data pointer, and
  // other attributes may differ due to dimension reordering and
  // coalescing.
  // TODO: check this tensor, make sure it doesn't affect the original one
  const Tensor& tensor() const { return tensor_; }
  void tensor(Tensor&& tensor) { tensor_ = std::move(tensor); }
  Device optional_device() const {
    return tensor_.defined() ? tensor_.device() : Device("cpu");
  }
  bool optional_requires_grad() const {
    return tensor_.defined() ? tensor_.requires_grad() : false;
  }

 private:
  Tensor tensor_;
};

struct DimCounter {
  DimCounter(IntArrayRef shape, Range range);

  void increment(const std::array<int64_t, 2>& step);
  bool is_done() const { return offset >= range.end; }
  std::array<int64_t, 2> max_2d_step() const;

  IntArrayRef shape;
  Range range;
  IntArrayRef values;
  int64_t offset;
};

enum class FastSetupType : uint8_t {
  NONE,
  CONTIGUOUS,
};

struct TensorIterator {
  // The inner-loop function operates on the fastest moving dimension.
  //
  // Arguments:
  //  data: data pointers for each operand (length `ntensors`)
  //  strides: stride for each operand (length `ntensors`)
  //  size: size of inner loop
  using loop2d_t = FuncRef<void(char** data, const int64_t* strides,
                                int64_t size0, int64_t size1)>;

  void build();
  int ndim() const { return static_cast<int>(shape_.size()); }
  IntArrayRef shape() const { return shape_; }
  int64_t numel() const;
  int ntensors() const { return static_cast<int>(operands_.size()); }
  int noutputs() const { return num_outputs_; }
  int ninputs() const { return ntensors() - noutputs(); }

  // Reducible to 1-dimensional and all operands are contiguous
  bool is_contiguous() const;

  // Accessors for each operand
  IntArrayRef strides(int arg) const { return operands_[arg].stride_bytes; }
  void* data_ptr(int arg) const { return operands_[arg].data; }

  int64_t element_size(int arg) const {
    return static_cast<int64_t>(sizeof(data_t));
  }

  const Tensor& tensor(int arg) const { return operands_[arg].tensor(); }

  const Tensor& output(int arg = 0) const {
    TORCH_CHECK(arg < num_outputs_, "arg < num_outputs_");
    return tensor(arg);
  }

  const Tensor& input(int arg = 0) const {
    TORCH_CHECK(arg >= 0 && arg < ntensors() - num_outputs_,
                "arg >= 0 && arg < ntensors() - num_outputs_");
    return tensor(num_outputs_ + arg);
  }

  TensorIterator& add_output(Tensor& output) {
    TORCH_CHECK(num_inputs_ == 0,
                "You have to add outputs first before adding any input.");
    operands_.push_back(OperandInfo(std::move(output)));
    num_outputs_++;
    return *this;
  }
  TensorIterator& add_input(const Tensor& input) {
    operands_.push_back(OperandInfo(input));
    num_inputs_++;
    return *this;
  }
  // Do not allow borrow from temporaries.
  TensorIterator& add_output(Tensor&& output) = delete;
  TensorIterator& add_input(Tensor&& input) = delete;

 private:
  template <typename loop1d_t>
  auto loop_2d_from_1d(const loop1d_t& loop) {
    return [loop, ntensor = ntensors()](char** base, const int64_t* strides,
                                        int64_t size0, int64_t size1) {
      PtrArrayRef data(base, base + ntensor);
      const int64_t* outer_strides = &strides[ntensor];
      for (const auto i : irange(size1)) {
        if (i > 0) {
          for (const auto arg : irange(ntensor)) {
            data[arg] += outer_strides[arg];
          }
        }
        loop(data.vec().data(), strides, size0);
      }
    };
  }

 public:
  template <typename loop1d_t,
            std::enable_if_t<
                std::is_convertible<loop1d_t,
                                    FuncRef<void(char**, const int64_t* strides,
                                                 int64_t size)>>::value,
                int> = 0>
  void for_each(loop1d_t loop, int64_t grain_size = GRAIN_SIZE) {
    for_each(loop_2d_from_1d(loop), grain_size);
  }

  void for_each(loop2d_t loop, int64_t grain_size = GRAIN_SIZE);

  template <typename loop1d_t,
            std::enable_if_t<
                std::is_convertible<loop1d_t,
                                    FuncRef<void(char**, const int64_t* strides,
                                                 int64_t size)>>::value,
                int> = 0>
  void serial_for_each(loop1d_t loop, Range range) {
    serial_for_each(loop_2d_from_1d(loop), range);
  }

  void serial_for_each(loop2d_t loop, Range range) const;

  // Create a strides array for a Tensor with shape of this iterator. The
  // parameter `element_size` specifies the size of Tensor's data type in
  // bytes (e.g. `4` for `float`)
  IntArrayRef compatible_stride(int element_size) const;

  // Inverts the re-ordering done by reorder_dimensions. This can only be
  // called *before* coalesce_dimensions() is called.
  IntArrayRef invert_perm(IntArrayRef input) const;

  // Helper functions for CPU iteration
  IntArrayRef get_dim_strides(int dim) const;
  IntArrayRef get_strides() const;
  IntArrayRef get_inner_strides() const { return get_dim_strides(0); }
  PtrArrayRef get_base_ptrs() const;

  const Tensor& maybe_get_output(int64_t output_idx) {
    return output(output_idx);
  };

  bool has_contiguous_first_dim() const {
    if (ndim() == 0) {
      return true;
    }

    int num_tensors = ntensors();
    for (const auto i : irange(num_tensors)) {
      if (strides(i)[0] != element_size(i)) {
        return false;
      }
    }
    return true;
  }

  void set_output_raw_strided(int64_t output_idx, IntArrayRef sizes,
                              IntArrayRef strides, Device device,
                              bool requires_grad);

 protected:
  void mark_outputs();
  void mark_resize_outputs();
  void compute_shape();
  void compute_strides();
  void reorder_dimensions();
  void permute_dimensions(IntArrayRef perm);
  void compute_device();
  void allocate_or_resize_outputs();
  bool fast_set_up();
  FastSetupType compute_fast_setup_type();
  void coalesce_dimensions();

 protected:
  // Records the "computation" shape of the output tensor.
  IntArrayRef shape_;

  // Temporarily records the permutation computed by reorder_dimensions.
  // This permutation maps the computation output dimension (dim) to
  // the original true output dimension (perm_[dim]).  It is used by
  // invert_perm to undo the permutation.  After coalesce_dimensions is
  // called, the permutation is no longer valid (as, in general, there
  // is no permutation that will make computation dimensions to
  // output dimensions); methods that manipulate perm_ are obligated
  // to test that !has_coalesced_dimensions
  IntArrayRef perm_;

  // Has coalesce_dimensions() (or any moral equivalent, e.g., fast_build())
  // been called?  This is SOLELY used to check validity of perm_.
  bool has_coalesced_dimensions_ = false;

  // The operands of the TensorIterator: both the inputs and outputs.  The
  // outputs MUST come first in the operands_ list.
  ArrayRef<OperandInfo> operands_;
  int num_outputs_ = 0;
  int num_inputs_ = 0;

  // Whether or not all operands have the same shape and are 1d+.
  bool all_ops_same_shape_ = false;
  Device common_device_ = Device("cpu");
  bool is_reduction_ = false;
};

}  // namespace microtorch
