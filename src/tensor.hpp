#pragma once

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <vector>

namespace tinytorch {

struct Edge;

struct TensorImpl {
  std::vector<float> data;
  std::vector<float> grad;  // for .backward
  TensorImpl(int size) : data(size) {}
  TensorImpl(std::vector<float> data) : data(data) {}
  std::shared_ptr<Edge> edge;
};

struct Tensor {
 private:
  std::shared_ptr<TensorImpl> impl_;

 public:
  Tensor(int size = 0) : impl_(std::make_shared<TensorImpl>(size)) {}
  Tensor(std::vector<float> data) : impl_(std::make_shared<TensorImpl>(data)) {}

  // operator override
  float &operator[](int idx) { return (*impl_).data[idx]; }

  // tensor functions
  int size() { return (*impl_).data.size(); }
  void resize(int size) {
    assert(impl_);
    (*impl_).data.resize(size, 0);
    (*impl_).grad.resize(size, 0);
  }
  void clearGrad() { (*impl_).grad.clear(); }

  Tensor grad() { return Tensor((*impl_).grad); }

  void addInplace(Tensor t) {
    resize(t.size());
    for (size_t i = 0; i < size(); i++) {
      (*impl_).data[i] += t[i];
    }
  }
  void addGradInplace(Tensor g) {
    resize(g.size());
    for (size_t i = 0; i < size(); i++) {
      (*impl_).grad[i] += g[i];
    }
  }
  std::shared_ptr<Edge> getEdge() { return (*impl_).edge; };
  void setEdge(std::shared_ptr<Edge> edge) { (*impl_).edge = edge; };

  float* data() {return impl_->data.data();};
};

}  // namespace tinytorch