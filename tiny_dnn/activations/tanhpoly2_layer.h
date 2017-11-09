/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <string>
#include <utility>

#include "tiny_dnn/activations/activation_layer.h"
#include "tiny_dnn/layers/layer.h"

namespace tiny_dnn {

class tanhpoly2_layer : public activation_layer {
 public:
  using activation_layer::activation_layer;

  float_t stretch = 1.0;

  std::string layer_type() const override { return "tanhpoly2-activation"; }

  void forward_activation(const vec_t &x, vec_t &y) override {
    for (size_t j = 0; j < x.size(); j++) {
      float_t t  = x[j] / stretch;
      float_t t2 = t*t;
      float_t ts = (3.0*t2-10.0);
      // (1.0/8.0)*t*((3.0*t*t-10.0)*t*t+15.0)
      y[j]  = (t <= -1.0) ? -1.0 : ( (t >= 1.0) ? 1.0 :  0.125*t*(ts*t2 + 15.0) );
    }
  }

  void backward_activation(const vec_t &x,
                           const vec_t &y,
                           vec_t &dx,
                           const vec_t &dy) override {
    for (size_t j = 0; j < x.size(); j++) {
      // dx = dy * (gradient of tanh)
      //(15/8)*(t^2-1)^2; t = x/a
      //dx[j] = dy[j] * (float_t(1) - sqr(y[j]));
      
      float_t t    = x[j] / stretch;
      float_t q    = t*t - 1.0;
      float_t grad = (t <= -1.0) ? 0.0 : ( (t >= 1.0) ? 0.0 :  1.875*q*q );

      dx[j] = dy[j] * grad;
    }
  }

  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(-0.8), float_t(0.8));
  }

  friend struct serialization_buddy;
private:

};

}  // namespace tiny_dnn