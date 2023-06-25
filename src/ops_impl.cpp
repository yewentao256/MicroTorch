
#include "ops.hpp"
#include "tensor.hpp"
namespace tinytorch {

// Tensor Create operators
Tensor zeros(size_t size, std::string device) {
  Tensor t(size);
  for (size_t i = 0; i < t.size(); i++) {
    t[i] = 0;
  }
  if (device == "cuda") {
    return t.cuda();    // TODO: zeros for cuda ops
  }
  return t;
}
Tensor ones(size_t size, std::string device) {
  Tensor t(size);
  for (size_t i = 0; i < t.size(); i++) {
    t[i] = 1;
  }
  if (device == "cuda") {
    return t.cuda();    // TODO: zeros for cuda ops
  }
  return t;
}
Tensor rand(size_t size, std::string device) {
  // TODO: device cuda support, t("cpu"), then return t.cuda()
  Tensor t(size);
  static std::mt19937 mersenne_engine{572547235};
  std::uniform_real_distribution<float> dist{0.f, 1.f};

  for (size_t i = 0; i < t.size(); i++) {
    t[i] = dist(mersenne_engine);
  }
  if (device == "cuda") {
    return t.cuda();    // TODO: zeros for cuda ops
  }
  return t;
}

std::ostream& print_with_size(std::ostream& stream, Tensor t, size_t print_size, const std::string& name = "name") {
  size_t size = t.numel();
  // if (t.arch() == "cuda") {
    
  // }
  stream << "<tinytorch.Tensor[" << name << "] size=" << size << ", device=" << t.arch()
         << ">: [";
  if (size > print_size) {
    // 只打印前print_size/2个和后print_size/2个元素
    for (size_t i = 0; i < print_size / 2; i++) {
      stream << std::setw(8) << t[i] << " ";
    }
    stream << " ... ";
    for (size_t i = size - print_size / 2; i < size; i++) {
      stream << std::setw(8) << t[i] << " ";
    }
  } else {
    for (size_t i = 0; i < size; i++) {
      stream << std::setw(8) << t[i] << " ";
    }
  }
  stream << "]";
  return stream;
}

std::string repr(Tensor t, size_t print_size, const std::string& name) {
  std::ostringstream s;
  print_with_size(s, t, print_size, name);
  return s.str();
}

std::ostream& operator<<(std::ostream& stream, Tensor t) {
  return print_with_size(stream, t, 20);  // 默认打印20个元素
}

template <>
void add_impl<Host>(Context& ctx, Tensor& a, Tensor& b, Tensor& out) {
  for (size_t i = 0; i < a.size(); i++) {
    out[i] = a[i] + b[i];
  }
}

template <>
void add_backward_impl<Host>(Context& ctx, Tensor& dy, Tensor& dx_1,
                             Tensor& dx_2) {
  for (size_t i = 0; i < dy.size(); i++) {
    // y = a + b, y'(a) = 1 * grad
    dx_1[i] = dy[i];
    dx_2[i] = dy[i];
  }
}

template <>
void sub_impl<Host>(Context& ctx, Tensor& a, Tensor& b, Tensor& out) {
  for (size_t i = 0; i < a.size(); i++) {
    out[i] = a[i] - b[i];
  }
}

std::vector<Tensor> sub_backward_impl(Tensor& dy) {
  Tensor result_a(dy.size());
  Tensor result_b(dy.size());
  for (size_t i = 0; i < dy.size(); i++) {
    // y = a - b, y'(a) = 1 * grad, y'(b) = -1 * grad
    result_a[i] = dy[i];
    result_b[i] = -dy[i];
  }
  return {result_a, result_b};
}

template <>
void mult_impl<Host>(Context& ctx, Tensor& a, Tensor& b, Tensor& out) {
  for (size_t i = 0; i < a.size(); i++) {
    out[i] = a[i] * b[i];
  }
}

std::vector<Tensor> mult_backward_impl(Tensor a, Tensor b, Tensor& dy) {
  Tensor result_a(a.size());
  Tensor result_b(a.size());
  for (size_t i = 0; i < a.size(); i++) {
    // y = a * b, y'(a) = b * grad
    result_a[i] = b[i] * dy[i];
    result_b[i] = a[i] * dy[i];
  }
  return {result_a, result_b};
}

template <>
void square_impl<Host>(Context& ctx, Tensor& a, Tensor& out) {
  for (size_t i = 0; i < a.size(); i++) {
    out[i] = a[i] * a[i];
  }
}
std::vector<Tensor> square_backward_impl(Tensor a, Tensor& dy) {
  Tensor result(a.size());
  for (size_t i = 0; i < a.size(); i++) {
    // y = a^2, y'(a) = 2 * a * grad
    result[i] = 2 * a[i] * dy[i];
  }
  return {result};
}

template <>
void sum_impl<Host>(Context& ctx, Tensor& a, Tensor& out) {
  for (size_t i = 0; i < a.size(); i++) {
    out[0] += a[i];
  }
}
std::vector<Tensor> sum_backward_impl(size_t input_size, Tensor& dy) {
  TORCH_CHECK(dy.size() == 1, "dy size should equal to 1");
  Tensor result(input_size);
  for (size_t i = 0; i < input_size; i++) {
    // y = a + b + c ..., y'(a) = 1 * grad
    result[i] = dy[0];
  }
  return {result};
}
}  // namespace tinytorch
