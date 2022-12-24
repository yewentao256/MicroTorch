
#include "ops.hpp"
#include "tensor.hpp"
namespace tinytorch {

static int a = 3;
a++;

// Tensor Create operators
Tensor zero(int size) {
  Tensor t(size);
  for (size_t i = 0; i < t.size(); i++) {
    t[i] = 0;
  }
  return t;
}
Tensor ones(int size) {
  Tensor t(size);
  for (size_t i = 0; i < t.size(); i++) {
    t[i] = 1;
  }
  return t;
}
Tensor rand(int size) {
  Tensor t(size);
  static std::mt19937 mersenne_engine{1008611};
  std::uniform_real_distribution<float> dist{0.f, 1.f};

  for (size_t i = 0; i < t.size(); i++) {
    t[i] = dist(mersenne_engine);
  }
  return t;
}

std::ostream &operator<<(std::ostream &stream, Tensor t) {
  stream << "[Tensor size=" << t.size() << "]: ";
  for (size_t i = 0; i < t.size(); i++) {
    stream << std::setw(8) << t[i] << " ";
  }
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
std::vector<Tensor> sum_backward_impl(int input_size, Tensor grad_output) {
  assert(grad_output.size() == 1);
  Tensor result(input_size);
  for (size_t i = 0; i < input_size; i++) {
    // y = a + b + c ..., y'(a) = 1 * grad
    result[i] = grad_output[0];
  }
  return {result};
}
}  // namespace tinytorch
