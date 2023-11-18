#include "tensorIterator.hpp"

#include "ops.hpp"

namespace microtorch {

namespace internal {

// `get_strides` stores the strides of all tensors in order (from low dimension
// to high dimension) for easier value calculation.
// For example, `output.stride_bytes` = [4, 256], `input.stride_bytes` = [80, 4]
// the result is [4, 80, 256, 4]
inline void get_strides(int64_t* strides, ArrayRef<OperandInfo> operands,
                        int64_t ndim) {
  for (const auto dim : irange(ndim)) {
    for (const auto arg : irange(operands.size())) {
      *strides++ = operands[arg].stride_bytes[dim];
    }
  }
  // Always at least 2d strides to support 2d for_each loops
  if (ndim < 2) {
    std::fill_n(strides, (2 - ndim) * operands.size(), 0);
  }
}

inline void get_data_ptrs(char** ptrs, ArrayRef<char*> base,
                          IntArrayRef strides, IntArrayRef counter) {
  const int64_t ntensors = base.size();
  std::copy(base.begin(), base.end(), ptrs);
  // Calculating the starting pointer of each tensor under the current range
  // algorithm: sum of the product of the offset and stride byte on all dimensions.
  for (const auto dim : irange(counter.size())) {
    int64_t value = counter[dim];
    for (const auto arg : irange(ntensors)) {
      ptrs[arg] += value * strides[dim * ntensors + arg];
    }
  }
}

inline void serial_for_each_(IntArrayRef shape, IntArrayRef strides,
                            PtrArrayRef tensor_ptrs,
                            typename TensorIterator::loop2d_t loop,
                            Range range) {
  char** base_ptrs = tensor_ptrs.vec().data();
  auto ntensors = tensor_ptrs.size();
  const auto ndim = shape.size();
  if (ndim <= 1) {
    if (range.begin == 0) {
      loop(base_ptrs, strides.vec().data(), range.size(), 1);
    } else {
      PtrArrayRef ptrs(ntensors);
      std::vector<char*> ptrs_vector(base_ptrs, base_ptrs + ntensors);
      get_data_ptrs(ptrs.vec().data(), ptrs_vector, strides, {range.begin});
      loop(ptrs.vec().data(), strides.vec().data(), range.size(), 1);
    }
  } else {
    // `ptrs` here stores the addresses that need to be processed in the current batch.
    PtrArrayRef ptrs(ntensors);
    auto counter = DimCounter(shape, range);
    // DimCounter ensures that every element is processed.
    // `is_done` judges whether the offset is greater than range.end.
    while (!counter.is_done()) {
      std::vector<char*> ptrs_vector(base_ptrs, base_ptrs + ntensors);
      get_data_ptrs(ptrs.vec().data(), ptrs_vector, strides, counter.values);
      auto step = counter.max_2d_step();
      loop(ptrs.vec().data(), strides.vec().data(), step[0], step[1]);
      counter.increment(step);
    }
  }
}

}  // namespace internal

void TensorIterator::parallel_reduce(loop2d_t loop) {
  TORCH_CHECK(ntensors() == 2,
              "parallel_reduce only supports one input and one output");
  serial_for_each(loop, {0, numel()});
  // TODO: we may support parallel in the future
  // parallel_dim_reduction(*this, loop);
}

IntArrayRef TensorIterator::compatible_stride(int64_t element_size) const {
  auto stride = IntArrayRef();
  int64_t next_stride = element_size;
  for (const auto dim : irange(ndim())) {
    stride.push_back(next_stride);
    next_stride *= shape_[dim];
  }
  return stride;
}

IntArrayRef TensorIterator::invert_perm(IntArrayRef input) const {
  // Invert the permutation caused by reorder_dimensions. This is not valid
  // after coalesce_dimensions is called.
  TORCH_CHECK(!has_coalesced_dimensions_, "has_coalesced_dimensions_");
  TORCH_CHECK(input.size() == perm_.size(), "input.size() == perm_.size()");
  auto res = IntArrayRef(input.size());  // no initialization needed, every
                                         // value in res should be written to.
  for (const auto dim : irange(ndim())) {
    res[perm_[dim]] = input[dim];
  }
  return res;
}

// checks that all tensors are on the same device
void TensorIterator::check_device() {
  // consider the device of first input as the common device
  common_device_ = operands_[num_outputs_].optional_device();
  for (auto& op : operands_) {
    if (!op.tensor().defined()) {
      TORCH_CHECK(op.is_out, "Found undefined input tensor!");
      continue;
    }
    // Checks all tensors are on the same device
    TORCH_CHECK(op.tensor().device() == common_device_,
                "Expected all tensors to be on the same device, but "
                "found at least two devices.");
  }
}

IntArrayRef TensorIterator::get_dim_strides(int64_t dim) const {
  auto dims = ndim();
  auto inner_strides = IntArrayRef();
  for (auto& op : operands_) {
    inner_strides.push_back(dims == 0 ? 0 : op.stride_bytes[dim]);
  }
  return inner_strides;
}

void TensorIterator::permute_dimensions(IntArrayRef perm) {
  TORCH_INTERNAL_ASSERT(perm.size() == ndim());

  auto reorder = [perm](IntArrayRef data) {
    auto res = IntArrayRef(data.size());
    for (const auto i : irange(perm.size())) {
      res[i] = data[perm[i]];
    }
    return res;
  };

  // Update shape and strides
  shape_ = reorder(shape_);
  for (auto& op : operands_) {
    if (!op.stride_bytes.empty()) {
      op.stride_bytes = reorder(op.stride_bytes);
    }
  }
}

void TensorIterator::for_each(TensorIterator::loop2d_t loop,
                              int64_t grain_size) {
  // TODO: we may support parallel in the future
  return serial_for_each(loop, {0, numel()});
}

void TensorIterator::serial_for_each(TensorIterator::loop2d_t loop,
                                     Range range) const {
  if (range.size() == 0) {
    return;
  }

  PtrArrayRef ptrs(ntensors());
  IntArrayRef strides(ntensors() * (ndim() > 2 ? ndim() : 2));

  // converts data ptrs to char* type, and stores them in the pointer list.
  std::transform(
      operands_.begin(), operands_.end(), ptrs.vec().data(),
      [](const OperandInfo& op) { return static_cast<char*>(op.data); });
  internal::get_strides(strides.vec().data(), operands_, ndim());
  internal::serial_for_each_(shape_, strides, ptrs, loop, range);
}

bool TensorIterator::is_contiguous() const {
  if (numel() == 1) {
    return true;
  }
  if (ndim() != 1) {
    return false;
  }
  return has_contiguous_first_dim();
}

void TensorIterator::mark_outs() {
  for (const auto i : irange(num_outputs_)) {
    operands_[i].is_out = true;
    const auto& out = tensor(i);
    if (!out.defined()) continue;

    // check if output is also an input
    for (const auto arg : irange(num_outputs_, ntensors())) {
      const auto& input = tensor(arg);
      if (out.equal(input)) {
        operands_[i].is_in_out = true;
      }
    }
  }
}

void TensorIterator::mark_resize_outs() {
  // Check that the shape of the outputs matches the inferred shape.
  for (const auto i : irange(num_outputs_)) {
    const auto& output = tensor(i);
    if (output.defined() && output.shape() != shape_) {
      if (resize_outs_ && !operands_[i].is_in_out) {
        operands_[i].should_resize = true;
        continue;
      }
      // for reduction, output size does not match shape_, as output is reduced
      TORCH_CHECK(is_reduction_, "output with shape ", output.shape(),
                  " doesn't match the broadcast shape ", shape_);
    }
  }
}

// coumpute broadcast shape
// for example: a = [2, 1, 3], b = [2, 1], the result shape would be [2, 2, 3]
IntArrayRef compute_broadcast_shape(IntArrayRef a, IntArrayRef b) {
  size_t ndim_a = a.size();
  size_t ndim_b = b.size();
  size_t ndim = ndim_a > ndim_b ? ndim_a : ndim_b;
  // size of result is the bigger ndim
  IntArrayRef result(ndim);

  // Use ptrdiff_t to ensure signed comparison.
  for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; --i) {
    // starting from the last index of a and b, then moving forward
    ptrdiff_t dim_a = ndim_a - ndim + i;
    ptrdiff_t dim_b = ndim_b - ndim + i;
    // if the index is smaller than 0, consider it as 1
    auto size_a = (dim_a >= 0) ? a[dim_a] : 1;
    auto size_b = (dim_b >= 0) ? b[dim_b] : 1;

    TORCH_CHECK(size_a == size_b || size_a == 1 || size_b == 1,
                "The size of tensor a (", size_a,
                ") must match the size of tensor b (", size_b,
                ") at non-singleton dimension ", i);

    // 1 is mapped to the other size (even for 0).
    result[i] = size_a == 1 ? size_b : size_a;
  }

  return result;
}

void TensorIterator::compute_common_shape() {
  bool has_scalars = false;
  bool has_tensors = false;
  for (auto& op : operands_) {
    if (!op.tensor().defined()) continue;
    // Output shapes don't participate in shape computation.
    // If the output tensor is also an input, we'll pick it up later.
    if (resize_outs_ && op.is_out) continue;
    auto shape = op.tensor().shape();
    if (shape.empty()) {
      has_scalars = true;
    } else {
      has_tensors = true;
    }
    if (has_scalars && has_tensors) {
      all_ops_same_shape_ = false;
    }
    if (shape_.empty()) {
      shape_ = shape;
    } else if (!(shape == shape_)) {
      all_ops_same_shape_ = false;
      shape_ = compute_broadcast_shape(shape_, shape);
    }
  }
}

bool TensorIterator::can_do_fast_setup() {
  if (is_reduction_ || !all_ops_same_shape_) {
    return false;
  }
  // non-contiguous tensor can not be used for fast setup
  for (const auto& op : operands_) {
    if (op.tensor().defined() && !op.should_resize) {
      if (!op.tensor().is_contiguous()) {
        return false;
      }
    }
  }
  return true;
}

// This function tries to do a fast setup to avoid needless reordering of
// dimensions and tracking output strides.
void TensorIterator::fast_set_up() {
  for (const auto i : irange(num_outputs_)) {
    configure_output(operands_[i], shape_, {},
                     operands_[i].optional_requires_grad());
  }
  // coalescing dimensions consists of collapsing dimensions to 1
  if (ndim() > 1) {
    has_coalesced_dimensions_ = true;
  }
  if (ndim() >= 1) {
    shape_[0] = numel();
    shape_.resize(1);
  }
  for (auto& op : operands_) {
    op.stride_bytes.resize(ndim());
    if (ndim() > 0) {
      op.stride_bytes[0] = op.tensor().element_size();
    }
  }
}

// compute the op's strides (in bytes level, not the original one)
// eg: a float tensor with shape[2, 3], strides[3, 1] now becomes [12, 4]
// eg2: a float tensor with shape[1, 3], strides[0, 1] now becomes [0, 4]
void TensorIterator::compute_strides() {
  for (auto& op : operands_) {
    if (op.tensor().defined() && !op.should_resize) {
      auto original_shape = op.tensor().shape();
      auto original_stride = op.tensor().stride();
      auto element_size_in_bytes = op.tensor().element_size();
      auto offset = ndim() - original_shape.size();
      if (offset > 0) {
        op.stride_bytes.resize(ndim(), 0);
      } else {
        op.stride_bytes.resize(ndim());
      }
      for (const auto i : irange(original_shape.size())) {
        if (original_shape[i] == 1 && shape_[offset + i] != 1) {
          // cases when broadcasting, consider the strides as 0
          op.stride_bytes[offset + i] = 0;
        } else {
          op.stride_bytes[offset + i] =
              original_stride[i] * element_size_in_bytes;
        }
      }
    }
  }
}

// Sort the dimensions based on strides in ascending order with reduced dims
// at the front. NOTE: this inverts the order of C-contiguous tensors.
// strides[0] is the fastest moving dimension instead of strides[ndim - 1].
void TensorIterator::reorder_dimensions() {
  perm_.resize(ndim());
  if (ndim() == 1) {
    perm_[0] = 0;
    return;
  }

  // initialize perm with n-1, n-2, ..., 1, 0
  std::iota(perm_.rbegin(), perm_.rend(), 0);

  // returns 1 if the dim0 should come after dim1, -1 if dim0 should come
  // before dim1, and 0 if the comparison is ambiguous.
  auto should_swap = [&](size_t dim0, size_t dim1) {
    for (const auto arg : irange(ntensors())) {
      // ignore undefined or incorrectly sized tensors
      if (operands_[arg].stride_bytes.empty() || operands_[arg].should_resize) {
        continue;
      }
      int64_t stride0 = operands_[arg].stride_bytes[dim0];
      int64_t stride1 = operands_[arg].stride_bytes[dim1];
      if (is_reduction_ && operands_[arg].is_out) {
        // move reduced dimensions to the front strides
        if ((stride0 == 0) != (stride1 == 0)) {
          return stride1 == 0 ? 1 : -1;
        }
      }
      // move on to the next input if one of the dimensions is broadcasted
      if (stride0 == 0 || stride1 == 0) {
        continue;
        // it is important to return here only with strict comparisons, for
        // equal strides we try to break the tie later by comparing
        // corresponding dimensions or if that does not work, moving on to the
        // next tensor
      } else if (stride0 < stride1) {
        return -1;
      } else if (stride0 > stride1) {
        return 1;
      } else {  // equal strides, use dimensions themselves as the tie-breaker.
        // at this point, with zero strides out of the way, we are guaranteed
        // that operand dimensions are equal to shape_
        auto t_dim0 = shape_[dim0];
        auto t_dim1 = shape_[dim1];
        // return only if dimensions should be swapped, otherwise move on to the
        // next tensor
        if (t_dim0 > t_dim1) {
          return 1;
        }
      }
    }
    return 0;
  };

  // insertion sort with support for ambiguous comparisons
  for (const auto i : irange(1, ndim())) {
    int64_t dim1 = i;
    for (int64_t dim0 = i - 1; dim0 >= 0; dim0--) {
      int64_t comparison = should_swap(perm_[dim0], perm_[dim1]);
      if (comparison > 0) {
        std::swap(perm_[dim0], perm_[dim1]);
        dim1 = dim0;
      } else if (comparison < 0) {
        break;
      }
    }
  }

  // perform re-ordering of shape and strides
  permute_dimensions(perm_);
}

void TensorIterator::allocate_or_resize_outputs() {
  for (const auto i : irange(num_outputs_)) {
    auto& op = operands_[i];
    if (!op.tensor().defined() || op.should_resize) {
      int64_t element_size = op.tensor().element_size();
      op.stride_bytes = compatible_stride(element_size);
      // check if permutation is just an inverted order
      bool inverted = true;
      for (const auto j : irange(ndim())) {
        if (perm_[j] != ndim() - j - 1) {
          inverted = false;
          break;
        }
      }
      auto tensor_shape = invert_perm(shape_);
      if (inverted) {
        // can just return contiguous output
        // it is faster because it avoids allocating 0 size tensor and
        // resizing and restriding it
        configure_output(op, tensor_shape, {}, op.optional_requires_grad());
      } else {
        auto tensor_stride = invert_perm(op.stride_bytes);
        for (const auto dim : irange(ndim())) {
          tensor_stride[dim] /= element_size;
        }
        configure_output(op, tensor_shape, tensor_stride,
                         op.optional_requires_grad());
      }
    } else if (op.tensor().defined()) {
      configure_output(op, op.tensor().shape(), {},
                       op.tensor().requires_grad());
    }
  }
}

// Try coalescing the adjacent dims.
// For example:
// `shape_` = [64, 4, 5, 1], `output.stride_bytes` = [4, 256, 1024, 5120],
// `input.stride_bytes` = [80, 4, 16, 5120]
// Changes to `shape_` = [64, 20],
// `output.stride_bytes` = [4, 256], `input.stride_bytes` = [80, 4]
void TensorIterator::coalesce_dimensions() {
  if (ndim() <= 1) {
    return;
  }

  // We can coalesce two adjacent dimensions if:
  // shape[n] == 1 or shape[n] * stride[n] == stride[n + 1]
  // Note that all tensors should meet the conditions simultaneously
  auto can_coalesce = [&](int64_t dim0, int64_t dim1) {
    auto shape0 = shape_[dim0];
    auto shape1 = shape_[dim1];
    if (shape0 == 1 || shape1 == 1) {
      return true;
    }
    for (const auto i : irange(ntensors())) {
      auto& stride = operands_[i].stride_bytes;
      if (shape0 * stride[dim0] != stride[dim1]) {
        return false;
      }
    }
    return true;
  };

  // replace each operands stride at dim0 with its stride at dim1
  auto replace_stride = [&](int64_t dim0, int64_t dim1) {
    for (const auto i : irange(ntensors())) {
      auto& stride = operands_[i].stride_bytes;
      stride[dim0] = stride[dim1];
    }
  };

  // The logic here is dual-pointer, starting from the `prev_dim` pointer,
  // traversing each dimension afterwards, and trying to coalesce as many
  // dimensions as possible
  int64_t prev_dim = 0;
  for (const auto dim : irange(1, ndim())) {
    if (can_coalesce(prev_dim, dim)) {
      if (shape_[prev_dim] == 1) {
        replace_stride(prev_dim, dim);
      }
      shape_[prev_dim] *= shape_[dim];
    } else {
      prev_dim++;
      if (prev_dim != dim) {
        replace_stride(prev_dim, dim);
        shape_[prev_dim] = shape_[dim];
      }
    }
  }

  // Finally shrink.
  shape_.resize(prev_dim + 1);
  for (const auto i : irange(ntensors())) {
    operands_[i].stride_bytes.resize(ndim());
  }
  has_coalesced_dimensions_ = true;
}

void TensorIterator::build() {
  // set is_out and is_in_out flags on appropriate tensors
  mark_outs();
  // compute the common broadcasted shape
  compute_common_shape();
  // mark outputs for resizing if necessary
  mark_resize_outs();
  // compute the result device
  check_device();
  // try fast setup output tensor, if failed, fallback to normal setup
  if (can_do_fast_setup()) {
    fast_set_up();
  } else {
    // compute each tensor's stride after broadcasting
    compute_strides();
    // re-order dimensions to improve coalescing
    reorder_dimensions();
    // allocate the output tensor if it's not provided
    allocate_or_resize_outputs();
    // coalesce adjacent dimensions when possible
    coalesce_dimensions();
  }
  for (auto& op : operands_) {
    TORCH_CHECK(op.tensor().defined(), "tensor should be defined");
    op.data = op.tensor().data_ptr();
  }
}

// Set output when output is not defined or should be resize
void TensorIterator::configure_output(OperandInfo& op, IntArrayRef sizes,
                                      IntArrayRef strides, bool requires_grad) {
  if (!op.tensor().defined()) {
    if (strides.empty()) {
      op.tensor(empty(sizes, common_device_, requires_grad));
    } else {
      TORCH_CHECK(false, "TODO: stride set is not supported now.");
    }
  } else if (op.should_resize) {
    TORCH_CHECK(false, "TODO: resize is not supported now.");
  }
}

DimCounter::DimCounter(IntArrayRef shape, Range range)
    : shape(shape), range(range), values(shape.size()), offset(range.begin) {
  std::fill(values.begin(), values.end(), 0);
  if (range.begin == 0) {
    return;
  }

  int64_t linear_offset = range.begin;
  int64_t ndim = values.size();
  for (const auto dim : irange(ndim)) {
    int64_t size = shape[dim];
    if (size > 0) {
      // linear_offset * stride to calculate the total offset,
      // in order to directly find the begin location of the current range.
      values[dim] = linear_offset % size;
      linear_offset /= size;
    }
  }
  TORCH_CHECK(linear_offset == 0, "linear offset should be 0.");
}

void DimCounter::increment(const std::array<int64_t, 2>& step) {
  offset += step[0] * step[1];
  int64_t ndim = values.size();
  int64_t overflow = step[0];
  int64_t i = 0;
  if (step[1] != 1) {
    TORCH_CHECK(step[0] == shape[0] && values[0] == 0,
                "step[0] should == shape[0] && values[0] should == 0.");
    i = 1;
    overflow = step[1];
  }
  for (; i < ndim && overflow > 0; i++) {
    auto size = shape[i];
    auto prev = values[i];
    auto value = prev + overflow;
    if (value >= size) {
      overflow = 1;
      value -= size;
      TORCH_CHECK(value < size, "value should < size.");
    } else {
      overflow = 0;
    }
    values[i] = value;
  }
  TORCH_CHECK(overflow == 0 || overflow == 1, "overflow should == 0 or 1.");
}

std::array<int64_t, 2> DimCounter::max_2d_step() const {
  int64_t step0 = std::min(shape[0] - values[0], range.end - offset);
  int64_t step1 = 1;
  if (step0 == shape[0] && !shape.empty()) {
    step1 = std::min(shape[1] - values[1], (range.end - offset) / shape[0]);
  }
  return {step0, step1};
}

}  // namespace microtorch