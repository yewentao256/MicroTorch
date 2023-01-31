
#include "ops.hpp"
#include "tensor.hpp"
namespace tinytorch {

// Tensor Create operators
Tensor zero(size_t size) {
  Tensor t(size);
  for (size_t i = 0; i < t.size(); i++) {
    t[i] = 0;
  }
  return t;
}
Tensor ones(size_t size) {
  Tensor t(size);
  for (size_t i = 0; i < t.size(); i++) {
    t[i] = 1;
  }
  return t;
}
Tensor rand(size_t size) {
  Tensor t(size);
  static std::mt19937 mersenne_engine{572547235};
  std::uniform_real_distribution<float> dist{0.f, 1.f};

  for (size_t i = 0; i < t.size(); i++) {
    t[i] = dist(mersenne_engine);
  }
  return t;
}

std::string repr(Tensor t) {
  std::ostringstream s;
  s << t;
  return s.str();
}

std::ostream &operator<<(std::ostream &stream, Tensor t) {
  size_t size = t.size();
  stream << "<tinytorch.Tensor size=" << size << ", device=" << t.arch() << ">: [";
  if (size > 20) {
    // only print several numbers
    for (size_t i = 0; i < 10; i++) {
      stream << std::setw(8) << t[i] << " ";
    }
    stream << " ... ";
    for (size_t i = size - 10; i < size; i++) {
      stream << std::setw(8) << t[i] << " ";
    }
  } else{
    for (size_t i = 0; i < size; i++) {
      stream << std::setw(8) << t[i] << " ";
    }
  }
  stream << "]";
  return stream;
}

Tensor add_impl(Tensor a, Tensor b) {
  Tensor result(a.size());
  for (size_t i = 0; i < a.size(); i++) {
    result[i] = a[i] + b[i];
  }
  return result;
}

std::vector<Tensor> add_backward_impl(Tensor grad_output) {
  Tensor result_a(grad_output.size());
  Tensor result_b(grad_output.size());
  for (size_t i = 0; i < grad_output.size(); i++) {
    // y = a + b, y'(a) = 1 * grad
    result_a[i] = grad_output[i];
    result_b[i] = grad_output[i];
  }
  return {result_a, result_b};
}

Tensor sub_impl(Tensor a, Tensor b) {
  Tensor result(a.size());
  for (size_t i = 0; i < a.size(); i++) {
    result[i] = a[i] - b[i];
  }
  return result;
}

std::vector<Tensor> sub_backward_impl(Tensor grad_output) {
  Tensor result_a(grad_output.size());
  Tensor result_b(grad_output.size());
  for (size_t i = 0; i < grad_output.size(); i++) {
    // y = a - b, y'(a) = 1 * grad, y'(b) = -1 * grad
    result_a[i] = grad_output[i];
    result_b[i] = -grad_output[i];
  }
  return {result_a, result_b};
}
Tensor mult_impl(Tensor a, Tensor b) {
  Tensor result(a.size());
  for (size_t i = 0; i < a.size(); i++) {
    result[i] = a[i] * b[i];
  }
  return result;
}

std::vector<Tensor> mult_backward_impl(Tensor a, Tensor b, Tensor grad_output) {
  Tensor result_a(a.size());
  Tensor result_b(a.size());
  for (size_t i = 0; i < a.size(); i++) {
    // y = a * b, y'(a) = b * grad
    result_a[i] = b[i] * grad_output[i];
    result_b[i] = a[i] * grad_output[i];
  }
  return {result_a, result_b};
}

Tensor square_impl(Tensor a) {
  Tensor result(a.size());
  for (size_t i = 0; i < a.size(); i++) {
    result[i] = a[i] * a[i];
  }
  return result;
}
std::vector<Tensor> square_backward_impl(Tensor a, Tensor grad_output) {
  Tensor result(a.size());
  for (size_t i = 0; i < a.size(); i++) {
    // y = a^2, y'(a) = 2 * a * grad
    result[i] = 2 * a[i] * grad_output[i];
  }
  return {result};
}

Tensor sum_impl(Tensor a) {
  Tensor result = zero(1);
  for (size_t i = 0; i < a.size(); i++) {
    result[0] += a[i];
  }
  return result;
}
std::vector<Tensor> sum_backward_impl(size_t input_size, Tensor grad_output) {
  assert(grad_output.size() == 1);
  Tensor result(input_size);
  for (size_t i = 0; i < input_size; i++) {
    // y = a + b + c ..., y'(a) = 1 * grad
    result[i] = grad_output[0];
  }
  return {result};
}
}  // namespace tinytorch
