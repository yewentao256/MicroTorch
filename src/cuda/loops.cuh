/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#include "cuda.hpp"
#include "tensorIterator.hpp"
#include "funcTraits.hpp"
#include "load.hpp"

#define HOST_DEVICE __device__ __host__

// A fixed-size array type usable from both host and device code.
template <typename T, int size_>
struct Array {
  T data[size_];

  HOST_DEVICE T operator[](int i) const {
    return data[i];
  }
  HOST_DEVICE T& operator[](int i) {
    return data[i];
  }
#if defined(USE_ROCM)
  HOST_DEVICE Array() = default;
  HOST_DEVICE Array(const Array&) = default;
  HOST_DEVICE Array& operator=(const Array&) = default;
#else
  Array() = default;
  Array(const Array&) = default;
  Array& operator=(const Array&) = default;
#endif
  static constexpr int size(){return size_;}
  // Fill the array with x.
  HOST_DEVICE Array(T x) {
    for (int i = 0; i < size_; i++) {
      data[i] = x;
    }
  }
};

constexpr int MAX_DIMS = 8;
// constants from
// (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications)
// The maximum number of threads per multiprocessor is 1024 for Turing
// architecture (7.5), 1536 for Geforce Ampere (8.6)/Jetson Orin (8.7), and
// 2048 for all other architectures. You'll get warnings if you exceed these
// constants. Hence, the following macros adjust the input values from the user
// to resolve potential warnings.
#if __CUDA_ARCH__ == 750
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 1024;
#elif __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 870 || __CUDA_ARCH__ == 890
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 1536;
#else
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 2048;
#endif
// CUDA_MAX_THREADS_PER_BLOCK is same for all architectures currently
constexpr uint32_t CUDA_MAX_THREADS_PER_BLOCK = 1024;
// CUDA_THREADS_PER_BLOCK_FALLBACK is the "canonical fallback" choice of block
// size. 256 is a good number for this fallback and should give good occupancy
// and versatility across all architectures.
constexpr uint32_t CUDA_THREADS_PER_BLOCK_FALLBACK = 256;
// NOTE: if you are thinking of constexpr-ify the inputs to launch bounds, it
//       turns out that although __launch_bounds__ can take constexpr, it
//       can't take a constexpr that has anything to do with templates.
//       Currently we use launch_bounds that depend on template arguments in
//       Loops.cuh, Reduce.cuh and LossCTC.cuh. Hence, MAX_THREADS_PER_BLOCK
//       and MIN_BLOCKS_PER_SM are kept as macros.
// Suppose you were planning to write __launch_bounds__(a, b), based on your
// performance tuning on a modern GPU. Instead, you should write
// __launch_bounds__(MAX_THREADS_PER_BLOCK(a), MIN_BLOCKS_PER_SM(a, b)),
// which will also properly respect limits on old architectures.
#define MAX_THREADS_PER_BLOCK(val)               \
  (((val) <= CUDA_MAX_THREADS_PER_BLOCK) ? (val) \
                                         : CUDA_THREADS_PER_BLOCK_FALLBACK)
#define MIN_BLOCKS_PER_SM(threads_per_block, blocks_per_sm)            \
  ((((threads_per_block) * (blocks_per_sm) <= CUDA_MAX_THREADS_PER_SM) \
        ? (blocks_per_sm)                                              \
        : ((CUDA_MAX_THREADS_PER_SM + (threads_per_block)-1) /         \
           (threads_per_block))))

#define LAUNCH_BOUNDS(max_threads_per_block, min_blocks_per_sm) \
  __launch_bounds__(                                            \
      (MAX_THREADS_PER_BLOCK((max_threads_per_block))),         \
      (MIN_BLOCKS_PER_SM((max_threads_per_block), (min_blocks_per_sm))))

namespace microtorch {

// A utility class to implement integer division by multiplication, given a fixed
// divisor.
//
// WARNING: The fast divider algorithm is only implemented for unsigned int;
//          otherwise we default to plain integer division.  For unsigned int,
//          we further assume that the dividend is at most INT32_MAX.  Thus,
//          IntDivider must NOT be used for general integer division.
//
//          This reduced range is enough for our purpose, and it allows us to
//          slightly simplify the computation.
//
// (NOTE: Below, "2^k" denotes exponentiation, i.e., 1<<k.)
//
// For any N-bit unsigned integer d (> 0), we can find a "magic number" m (2^N
// <= m < 2^(N+1)) and shift s such that:
//
//    \floor(n / d) = \floor((m * n) / 2^(N+s)).
//
// Given such m and s, the integer division can be then implemented as:
//
//    let m' = m - 2^N  // 0 <= m' < 2^N
//
//    fast_integer_division(n):
//      // Multiply two N-bit unsigned integers: the result is a 2N-bit unsigned
//      // integer.  Then take the higher N bits.
//      t = (m' * n) >> N
//
//      // Here we use the fact that n is less than 2^(N-1): otherwise the value
//      // of (t + n) may not fit in an N-bit integer.
//      return (t + n) >> s
//
// Finding such a magic number is surprisingly easy:
//
//    s  = \ceil(\log_2 d)
//    m' = \floor(2^N * (2^s - d) / d) + 1  // Need 2N-bit integer arithmetic.
//
// See also:
//    - Division by Invariant Integers Using Multiplication,
//      Torbjörn Granlund and Peter L. Montgomery, 1994.
//
//    - http://www.hackersdelight.org/magic.htm
//
//    - http://ridiculousfish.com/blog/posts/labor-of-division-episode-i.html

// Result of div/mod operation stored together.
template <typename Value>
struct DivMod {
  Value div, mod;

  HOST_DEVICE DivMod(Value div, Value mod) : div(div), mod(mod) { }
};

// Base case: we only have an implementation for uint32_t for now.  For
// everything else, we use plain division.
template <typename Value>
struct IntDivider {
  IntDivider() = default;
  IntDivider(Value d) : divisor(d) { }

  HOST_DEVICE inline Value div(Value n) const { return n / divisor; }
  HOST_DEVICE inline Value mod(Value n) const { return n % divisor; }
  HOST_DEVICE inline DivMod<Value> divmod(Value n) const {
    return DivMod<Value>(n / divisor, n % divisor);
  }

  Value divisor;
};

// Implement fast integer division.
template <>
struct IntDivider<unsigned int> {
  static_assert(sizeof(unsigned int) == 4, "Assumes 32-bit unsigned int.");

  IntDivider() = default;

  IntDivider(unsigned int d) : divisor(d) {
    TORCH_INTERNAL_ASSERT(divisor >= 1 && divisor <= INT32_MAX);

    // TODO: gcc/clang has __builtin_clz() but it's not portable.
    for (shift = 0; shift < 32; shift++) if ((1U << shift) >= divisor) break;

    uint64_t one = 1;
    uint64_t magic = ((one << 32) * ((one << shift) - divisor)) / divisor + 1;
    m1 = magic;
    TORCH_INTERNAL_ASSERT(m1 > 0 && m1 == magic);  // m1 must fit in 32 bits.
  }

  HOST_DEVICE inline unsigned int div(unsigned int n) const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    // 't' is the higher 32-bits of unsigned 32-bit multiplication of 'n' and
    // 'm1'.
    unsigned int t = __umulhi(n, m1);
    return (t + n) >> shift;
#else
    // Using uint64_t so that the addition does not overflow.
    uint64_t t = ((uint64_t) n * m1) >> 32;
    return (t + n) >> shift;
#endif
  }

  HOST_DEVICE inline unsigned int mod(unsigned int n) const {
    return n - div(n) * divisor;
  }

  HOST_DEVICE inline DivMod<unsigned int> divmod(unsigned int n) const {
    unsigned int q = div(n);
    return DivMod<unsigned int>(q, n - q * divisor);
  }

  unsigned int divisor;  // d above.
  unsigned int m1;  // Magic number: m' above.
  unsigned int shift;  // Shift amounts.
};

template <int NARGS, typename index_t = uint32_t>
struct OffsetCalculator {
  // The offset for each argument. Wrapper around fixed-size array.
  // On CUDA, zero sized array is not allowed, so when we are handling nullary
  // operators, we need to create a size 1 offset to avoid compiler failure.
  // This size 1 offset is just a placeholder, and we will not use it.
  using offset_type = Array<index_t, std::max<int>(NARGS, 1)>;

  // if element_sizes is nullptr, then the strides will be in bytes, otherwise
  // the strides will be in # of elements.
  OffsetCalculator(int dims, const int64_t* sizes,
                   const int64_t* const* strides)
      : dims_(dims) {
    TORCH_CHECK(dims <= MAX_DIMS, "tensor has too many (>", MAX_DIMS, ") dims");
    for (int i = 0; i < dims; i++) {
      sizes_[i] = IntDivider<index_t>(sizes[i]);
      for (int arg = 0; arg < NARGS; arg++) {
        // TODO: after the vector is fixed, restore here 
        // strides_[i][arg] = strides[arg][i];
        strides_[i][arg] = 4;
      }
    }
    printf("stride_bytes in OffsetCalculator: [%d, %d, %d]\n", strides_[0][0], strides_[0][1], strides_[0][2]);
  }

  HOST_DEVICE offset_type get(index_t linear_idx) const {
    offset_type offsets;

    for (int arg = 0; arg < NARGS; arg++) {
      offsets[arg] = 0;
    }

    for (int dim = 0; dim < MAX_DIMS; ++dim) {
      if (dim == dims_) break;
      auto divmod = sizes_[dim].divmod(linear_idx);
      linear_idx = divmod.div;

      for (int arg = 0; arg < NARGS; arg++) {
        offsets[arg] += divmod.mod * strides_[dim][arg];
      }
    }
    return offsets;
  }

  int dims_;
  IntDivider<index_t> sizes_[MAX_DIMS];
  index_t strides_[MAX_DIMS][std::max<int>(NARGS, 1)];
};

// Make an OffsetCalculator with byte offsets
template <int N>
static OffsetCalculator<N, uint32_t> make_offset_calculator(
    const TensorIterator& iter) {
  TORCH_INTERNAL_ASSERT(N <= iter.ntensors());
  std::array<const int64_t*, N> strides;
  for (int i = 0; i < N; i++) {
    // TODO(fix me): since strides is a vector, it can not be use directly
    strides[i] = iter.strides(i).data();
  }
  return OffsetCalculator<N, uint32_t>(
      iter.ndim(), iter.shape().data(), strides.data());
}

template <int nt, int vt, typename func_t>
LAUNCH_BOUNDS(nt, 4)
__global__ void elementwise_kernel(int N, func_t f) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int idx = nv * blockIdx.x + tid;
#pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      f(idx);
      idx += nt;
    }
  }
}

template <typename traits, typename func_t, typename index_t, size_t... INDEX>
HOST_DEVICE typename traits::result_type invoke_impl(
    const func_t& f, char* const data[], const index_t strides[],
    std::index_sequence<INDEX...>) {
  (void)strides;
  // if INDEX = 0, 1, ... the code will be expanded to
  // f(load<arg0_type>(data[0] + strides[0]), load<arg1_type>(data[1] + strides[1]), ...)
  printf("stride_bytes in invoke: [%d, %d, %d]\n", strides[0], strides[1], strides[2]);
  return f(load<typename traits::template arg<INDEX>::type>(
      data[INDEX] + strides[INDEX])...);
}

template <typename func_t, typename index_t,
          typename traits = FuncTraits<func_t>>
HOST_DEVICE typename traits::result_type invoke(const func_t& f,
                                                        char* const data[],
                                                        const index_t strides[]) {
  using Indices = std::make_index_sequence<traits::num_args>;
  return invoke_impl<traits>(f, data, strides, Indices{});
}

template <int nt, int vt, typename func_t>
static void launch_legacy_kernel(int64_t N, const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) return;
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  // TODO: stream management
  // auto stream = getCurrentCUDAStream();
  elementwise_kernel<nt, vt, func_t><<<grid, block, 0>>>(N, f);
  CUDA_ERROR_CHECK();
}

template <typename func_t>
void gpu_kernel(TensorIterator& iter, const func_t& f) {
  if (iter.numel() == 0) return;
  TORCH_INTERNAL_ASSERT(iter.common_device().is_cuda());
  TORCH_INTERNAL_ASSERT(iter.numel() < std::numeric_limits<int32_t>::max());

  using traits = FuncTraits<func_t>;
  using rtype = typename traits::result_type;
  constexpr int ntensors = traits::num_args + 1;

  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::num_args);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);

  Array<char*, ntensors> tensor_ptrs;
  for (int i = 0; i < ntensors; i++) {
    tensor_ptrs[i] = (char*)iter.tensor(i).data_ptr();
  }

  int64_t numel = iter.numel();

  // TODO: vectorized_kernel
  //   if (iter.is_contiguous()) {
  //     return launch_vectorized_kernel(numel, f, tensor_ptrs);
  //   }
  auto offset_calc = make_offset_calculator<traits::num_args + 1>(iter);
  constexpr int unroll_factor = sizeof(rtype) >= 4 ? 2 : 4;
  launch_legacy_kernel<128, unroll_factor>(numel, [=] HOST_DEVICE(int idx) {
    auto offsets = offset_calc.get(idx);
    rtype* out = (rtype*)(tensor_ptrs[0] + offsets[0]);
    *out = invoke(f, &tensor_ptrs.data[1], &offsets.data[1]);
  });
}

}  // namespace microtorch
