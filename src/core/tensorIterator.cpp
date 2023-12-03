#include "tensorIterator.hpp"

#include "ops.hpp"

namespace microtorch {

namespace {

// `get_strides_bytes` stores the strides of all tensors in order (from low
// dimension to high dimension) for easier value calculation. For example,
// `output.stride_bytes` = [4, 256], `input.stride_bytes` = [80, 4] the result
// is [4, 80, 256, 4]
inline void get_strides_bytes(int64_t* strides_bytes,
                              ArrayRef<OperandInfo> operands, int64_t ndim) {
  for (const auto dim : irange(ndim)) {
    for (const auto arg : irange(operands.size())) {
      *strides_bytes++ = operands[arg].stride_bytes[dim];
    }
  }
  // Always at least 2d strides to support 2d for_each loops
  if (ndim < 2) {
    std::fill_n(strides_bytes, (2 - ndim) * operands.size(), 0);
  }
}

// Calculating the starting address of each tensor under the current range
inline void get_data_ptrs(char** ptrs, int64_t ntensors,
                          IntArrayRef strides_bytes, IntArrayRef dim_offsets) {
  // sum of the product of the offset and stride_byte on all dimensions.
  // For example, `dim_offsets` = [46, 666, 8], `output_stride` (corresponding
  // to `strides_bytes` dimension = [0,2,4]) = [4, 256, 512000]. The starting
  // address of output is `base + 46 * 4 + 666 * 256 + 8 * 512000 = base +
  // 4266680`.
  for (const auto i : irange(dim_offsets.size())) {
    int64_t offset = dim_offsets[i];
    for (const auto t : irange(ntensors)) {
      ptrs[t] += offset * strides_bytes[i * ntensors + t];
    }
  }
}

inline void serial_for_each_(IntArrayRef shape, IntArrayRef strides_bytes,
                             PtrArrayRef tensor_ptrs,
                             typename TensorIterator::loop2d_t loop,
                             Range range) {
  char** tensor_base_ptrs = tensor_ptrs.data();
  auto ntensors = tensor_ptrs.size();
  const auto ndim = shape.size();
  if (ndim <= 1) {
    if (range.begin == 0) {
      loop(tensor_base_ptrs, strides_bytes.data(), range.size(), 1);
    } else {
      PtrArrayRef ptrs(tensor_base_ptrs, tensor_base_ptrs + ntensors);
      get_data_ptrs(ptrs.data(), ntensors, strides_bytes, {range.begin});
      loop(ptrs.data(), strides_bytes.data(), range.size(), 1);
    }
  } else {
    // `ptrs` stores the addresses that need to be processed in current batch.
    PtrArrayRef ptrs(ntensors);
    auto counter = DimCounter(shape, range);
    // `is_done` judges whether the offset is greater than range.end.
    while (!counter.is_done()) {
      std::copy(tensor_base_ptrs, tensor_base_ptrs + ntensors, ptrs.data());
      get_data_ptrs(ptrs.data(), ntensors, strides_bytes, counter.dim_offsets_);
      std::array<int64_t, 2> steps = counter.get_max_2d_steps();
      loop(ptrs.data(), strides_bytes.data(), steps[0], steps[1]);
      counter.increment(steps[0], steps[1]);
    }
  }
}

}  // namespace

void TensorIterator::parallel_reduce(loop2d_t loop) {
  TORCH_CHECK(ntensors() == 2,
              "parallel_reduce only supports one input and one output");
  serial_for_each(loop, {0, numel()});
  // TODO: we may support parallel in the future
  // parallel_dim_reduction(*this, loop);
}

// checks that all tensors are on the same device
void TensorIterator::check_device() {
  // consider the device of first input as the common device
  common_device_ = operands_[num_outputs_].tensor().device();
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

void TensorIterator::for_each(TensorIterator::loop2d_t loop,
                              int64_t grain_size) {
  // TODO: we may support parallel in the future
  return serial_for_each(loop, {0, numel()});
}

void TensorIterator::serial_for_each(TensorIterator::loop2d_t loop,
                                     Range range) const {
  if (range.size() == 0) return;

  PtrArrayRef tensor_ptrs(ntensors());
  IntArrayRef strides_bytes(ntensors() * (ndim() > 2 ? ndim() : 2));

  // convert data ptrs to char* type, and store in `tensor_ptrs`.
  std::transform(
      operands_.begin(), operands_.end(), tensor_ptrs.data(),
      [](const OperandInfo& op) { return static_cast<char*>(op.data); });
  // extract op.stride_bytes
  get_strides_bytes(strides_bytes.data(), operands_, ndim());
  serial_for_each_(shape_, strides_bytes, tensor_ptrs, loop, range);
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
  // coalescing dimensions to 1
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

// Compute the op's `stride_bytes`
// eg: a float tensor with shape[2, 3], strides[3, 1] and we get [12, 4]
// eg2: a float tensor with shape[1, 3], strides[0, 1] and we get [0, 4]
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

// Sort the dimensions based on strides in ascending order.
// strides[0] is the fastest moving dimension instead of strides[ndim - 1].
// Eg: An input tensor with shape=[3, 2], stride_bytes=[8, 4] -> [4, 8]
void TensorIterator::reorder_dimensions() {
  perm_.resize(ndim());
  if (ndim() == 1) {
    perm_[0] = 0;
    return;
  }

  // initialize perm with n-1, n-2, ..., 1, 0
  std::iota(perm_.rbegin(), perm_.rend(), 0);

  // check whether two dims should swap
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
        // move reduced dimensions for output to the front strides
        if ((stride0 == 0) != (stride1 == 0)) {
          return stride1 == 0 ? 1 : -1;
        }
      }
      // move on to the input if one of the dimensions is broadcasted
      if (stride0 == 0 || stride1 == 0) {
        continue;
      } else if (stride0 < stride1) {
        return -1;
      } else if (stride0 > stride1) {
        return 1;
      } else {
        // case when equal strides, use shape to compare
        auto t_dim0 = shape_[dim0];
        auto t_dim1 = shape_[dim1];
        if (t_dim0 > t_dim1) {
          return 1;
        }
      }
    }
    return 0;
  };

  // calculate for perm_
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
      // for ambiguous comparison, skip
    }
  }

  // perform re-ordering of shape and strides
  auto apply_perm = [this](IntArrayRef data) {
    auto res = IntArrayRef(data.size());
    for (const auto i : irange(perm_.size())) {
      res[i] = data[perm_[i]];
    }
    return res;
  };

  // Update shape and strides
  shape_ = apply_perm(shape_);
  for (auto& op : operands_) {
    if (!op.stride_bytes.empty()) {
      op.stride_bytes = apply_perm(op.stride_bytes);
    }
  }
}

void TensorIterator::allocate_or_resize_outputs() {
  // Invert the permutation caused by reorder_dimensions.
  auto invert_perm = [this](IntArrayRef data) {
    TORCH_INTERNAL_ASSERT(data.size() == perm_.size());
    auto res = IntArrayRef(data.size());
    for (const auto i : irange(data.size())) {
      res[perm_[i]] = data[i];
    }
    return res;
  };

  for (const auto i : irange(num_outputs_)) {
    auto& op = operands_[i];
    if (!op.tensor().defined() || op.should_resize) {
      op.init_stride_bytes(shape_);
      // check if permutation is just an inverted order: contiguous output
      bool fully_inverted = true;
      for (const auto j : irange(ndim())) {
        if (perm_[j] != ndim() - j - 1) {
          fully_inverted = false;
          break;
        }
      }
      // TODO: we may directly record the original shape instead of compute it
      // again?
      auto original_shape = invert_perm(shape_);
      if (fully_inverted) {
        configure_output(op, original_shape, {}, op.optional_requires_grad());
      } else {
        auto original_strides = invert_perm(op.stride_bytes);
        int64_t element_size = op.tensor().element_size();
        for (const auto dim : irange(ndim())) {
          original_strides[dim] /= element_size;
        }
        configure_output(op, original_shape, original_strides,
                         op.optional_requires_grad());
      }
    } else if (op.tensor().defined()) {
      configure_output(op, op.tensor().shape(), {},
                       op.tensor().requires_grad());
    } else {
      TORCH_INTERNAL_ASSERT(false);
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
  // shape[n] / shape[n+1] == 1 or
  // shape[n] * stride[n] == stride[n + 1] for all of the tensors
  auto can_coalesce = [&](int64_t dim0, int64_t dim1) {
    if (shape_[dim0] == 1 || shape_[dim1] == 1) {
      return true;
    }
    for (const auto i : irange(ntensors())) {
      auto& stride = operands_[i].stride_bytes;
      if (shape_[dim0] * stride[dim0] != stride[dim1]) {
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

  // Starting from the `prev_dim` pointer, traversing each dimension afterwards,
  // and trying to coalesce as many dimensions as possible
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
    : shape_(shape),
      range_(range),
      dim_offsets_(shape.size()),
      offset_(range.begin) {
  std::fill(dim_offsets_.begin(), dim_offsets_.end(), 0);
  if (range.begin == 0) return;

  int64_t linear_offset = range.begin;
  for (const auto i : irange(dim_offsets_.size())) {
    int64_t size = shape[i];
    if (size > 0) {
      // calculating the dim_offsets
      // For example, `range.begin` = 1066670, `shape` = [64, 2000, 10],
      // Then `dim_offsets_` stores [46, 666, 8].
      dim_offsets_[i] = linear_offset % size;
      linear_offset /= size;
    }
  }
  TORCH_INTERNAL_ASSERT(linear_offset == 0);
}

// Get the steps that should be processed in current batch.
// Try to fetch the **maximum** range of steps
std::array<int64_t, 2> DimCounter::get_max_2d_steps() const {
  // If the offset is already close to end, fetch the remaining data
  int64_t step0 = std::min(shape_[0] - dim_offsets_[0], range_.end - offset_);
  int64_t step1 = 1;
  // eg, range = {1066670, 1280000}, shape = [64, 2000, 10]
  // offset = 1066670, dim_offsets = [46, 666, 8]
  // round1:
  // return {18, 1}, then updates offset to 1066688, dim_offsets = [0, 667, 8]
  // round2:
  // return {64, 1333}, updates offset to 1152000, dim_offsets = [0, 0, 9]
  // round3:
  // return {64, 2000}, updates offset to 1280000, dim_offsets = [0, 0, 0]
  if (step0 == shape_[0] && !shape_.empty()) {
    step1 = std::min(shape_[1] - dim_offsets_[1],
                     (range_.end - offset_) / shape_[0]);
  }
  return {step0, step1};
}

// updates offset and dim_offsets according to the steps we fetched
void DimCounter::increment(int64_t step0, int64_t step1) {
  offset_ += step0 * step1;
  int64_t ndim = dim_offsets_.size();
  int64_t overflow = step0;
  int64_t i = 0;
  if (step1 != 1) {
    TORCH_INTERNAL_ASSERT(step0 == shape_[0] && dim_offsets_[0] == 0);
    i = 1;
    overflow = step1;
  }
  // traverse through the dim_offsets, if we can make the dim_offset to 0
  // do it and add an overflow to next dim_offsets
  for (; i < ndim && overflow > 0; i++) {
    auto dim = shape_[i];
    auto dim_offset = dim_offsets_[i] + overflow;
    if (dim_offset >= dim) {
      overflow = 1;
      dim_offset -= dim;
      TORCH_INTERNAL_ASSERT(dim_offset < dim);
    } else {
      overflow = 0;
    }
    dim_offsets_[i] = dim_offset;
  }
  TORCH_INTERNAL_ASSERT(overflow == 0 || overflow == 1);
}

}  // namespace microtorch