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

  void* data = nullptr;

  // Stride after broadcasting. The stride is in bytes, not number of elements.
  IntArrayRef stride_bytes;
  bool is_out = false;
  bool should_resize = false;
  bool is_in_out = false;

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

// DimCounter ensures that every element is processed.
struct DimCounter {
  DimCounter(IntArrayRef shape, Range range);

  void increment(int64_t step0, int64_t step1);
  bool is_done() const { return offset_ >= range_.end; }
  std::array<int64_t, 2> get_max_2d_steps() const;

  // the shape of current tensor
  IntArrayRef shape_;
  // `range` is the range of elements to be processed, like {0, numel()}.
  Range range_;
  // The offset on each dimension, will be gradually updates to [0, 0, 0]
  IntArrayRef dim_offsets_;
  // The offset of the current element being processed.
  int64_t offset_;
};

struct TensorIterator {
  void build();
  int64_t ndim() const { return shape_.size(); }
  IntArrayRef shape() const { return shape_; }
  int64_t numel() const { return shape_.numel(); }
  int64_t ntensors() const { return operands_.size(); }
  int64_t noutputs() const { return num_outputs_; }
  int64_t ninputs() const { return ntensors() - noutputs(); }

  // Reducible to 1-dimensional and all operands are contiguous
  bool is_contiguous() const;

  // Accessors for each operand
  IntArrayRef strides(int64_t arg) const { return operands_[arg].stride_bytes; }
  void* data_ptr(int64_t arg) const { return operands_[arg].data; }

  const Tensor& tensor(int64_t arg) const { return operands_[arg].tensor(); }

  const Tensor& output(int64_t arg = 0) const {
    TORCH_CHECK(arg < num_outputs_, "arg < num_outputs_");
    return tensor(arg);
  }

  const Tensor& input(int64_t arg = 0) const {
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
        loop(data.data(), strides, size0);
      }
    };
  }

 public:
  template <typename loop1d_t,
            std::enable_if_t<
                std::is_convertible_v<
                    loop1d_t, FuncRef<void(char**, const int64_t* strides,
                                           int64_t size)>>,
                int> = 0>
  void for_each(loop1d_t loop, int64_t grain_size = GRAIN_SIZE) {
    for_each(loop_2d_from_1d(loop), grain_size);
  }

  // The inner-loop function operates on the fastest moving dimension.
  // Arguments:
  //  data: data pointers for each operand (length `ntensors`)
  //  strides: stride for each operand (length `ntensors`)
  //  size: size of inner loop
  using loop2d_t = FuncRef<void(char** data, const int64_t* strides,
                                int64_t size0, int64_t size1)>;
  void for_each(loop2d_t loop, int64_t grain_size = GRAIN_SIZE);

  template <typename loop1d_t,
            std::enable_if_t<
                std::is_convertible_v<
                    loop1d_t, FuncRef<void(char**, const int64_t* strides,
                                           int64_t size)>>,
                int> = 0>
  void serial_for_each(loop1d_t loop, Range range) {
    serial_for_each(loop_2d_from_1d(loop), range);
  }

  void serial_for_each(loop2d_t loop, Range range) const;
  void parallel_reduce(loop2d_t loop);

  // Create a strides array for a Tensor with shape of this iterator. The
  // parameter `element_size` specifies the size of Tensor's data type in
  // bytes (e.g. `4` for `float`)
  IntArrayRef compatible_stride(int64_t element_size) const;

  // Inverts the re-ordering done by reorder_dimensions. This can only be
  // called *before* coalesce_dimensions() is called.
  IntArrayRef invert_perm(IntArrayRef input) const;

  // Helper functions for CPU iteration
  IntArrayRef get_dim_strides(int64_t dim) const;
  IntArrayRef get_inner_strides() const { return get_dim_strides(0); }

  const Tensor& maybe_get_output(int64_t output_idx) {
    return output(output_idx);
  };

  bool has_contiguous_first_dim() const {
    if (ndim() == 0) {
      return true;
    }
    for (const auto i : irange(ntensors())) {
      if (strides(i)[0] != operands_[i].tensor().element_size()) {
        return false;
      }
    }
    return true;
  }

  void configure_output(OperandInfo& op, IntArrayRef sizes, IntArrayRef strides,
                        bool requires_grad);

  // set properties
  TensorIterator& resize_outs(bool resize_outs) {
    resize_outs_ = resize_outs;
    return *this;
  }
  TensorIterator& is_reduction(bool is_reduction) {
    is_reduction_ = is_reduction;
    return *this;
  }

 protected:
  void mark_outs();
  void mark_resize_outs();
  void compute_common_shape();
  void compute_strides();
  void reorder_dimensions();
  void permute_dimensions(IntArrayRef perm);
  void check_device();
  void allocate_or_resize_outputs();
  void fast_set_up();
  bool can_do_fast_setup();
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
  int64_t num_outputs_ = 0;
  int64_t num_inputs_ = 0;

  // Whether or not all operands have the same shape and are 1d+.
  bool all_ops_same_shape_ = true;
  Device common_device_ = Device("cpu");
  bool is_reduction_ = false;
  bool resize_outs_ = true;
};

}  // namespace microtorch
