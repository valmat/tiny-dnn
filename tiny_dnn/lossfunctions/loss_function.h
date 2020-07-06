/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <vector>

#include "tiny_dnn/util/util.h"

namespace tiny_dnn {

// mean-squared-error loss function for regression
class mse {
 public:
  static float_t f(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    float_t d{0.0};

    for (size_t i = 0; i < y.size(); ++i) d += (y[i] - t[i]) * (y[i] - t[i]);

    return d / static_cast<float_t>(y.size());
  }

  static vec_t df(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    vec_t d(t.size());
    float_t factor = float_t(2) / static_cast<float_t>(t.size());

    for (size_t i = 0; i < y.size(); ++i) d[i] = factor * (y[i] - t[i]);

    return d;
  }
};

// absolute loss function for regression
class absolute {
 public:
  static float_t f(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    float_t d{0};

    for (size_t i = 0; i < y.size(); ++i) d += std::abs(y[i] - t[i]);

    return d / static_cast<float_t>(y.size());
  }

  static vec_t df(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    vec_t d(t.size());
    float_t factor = float_t(1) / static_cast<float_t>(t.size());

    for (size_t i = 0; i < y.size(); ++i) {
      float_t sign = y[i] - t[i];
      if (sign < float_t{0.f})
        d[i] = -factor;
      else if (sign > float_t{0.f})
        d[i] = factor;
      else
        d[i] = {0};
    }

    return d;
  }
};

// absolute loss with epsilon range for regression
// epsilon range [-eps, eps] with eps = 1./fraction
template <int fraction>
class absolute_eps {
 public:
  static float_t f(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    float_t d{0};
    const float_t eps = float_t(1) / fraction;

    for (size_t i = 0; i < y.size(); ++i) {
      float_t diff = std::abs(y[i] - t[i]);
      if (diff > eps) d += diff;
    }
    return d / static_cast<float_t>(y.size());
  }

  static vec_t df(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    vec_t d(t.size());
    const float_t factor = float_t(1) / static_cast<float_t>(t.size());
    const float_t eps    = float_t(1) / fraction;

    for (size_t i = 0; i < y.size(); ++i) {
      float_t sign = y[i] - t[i];
      if (sign < -eps)
        d[i] = -factor;
      else if (sign > eps)
        d[i] = factor;
      else
        d[i] = 0.f;
    }
    return d;
  }
};

// cross-entropy loss function for (multiple independent) binary classifications
class cross_entropy {
 public:
  static float_t f(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    float_t d{0};

    for (size_t i = 0; i < y.size(); ++i)
      d += -t[i] * std::log(y[i]) -
           (float_t(1) - t[i]) * std::log(float_t(1) - y[i]);

    return d;
  }

  static vec_t df(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    vec_t d(t.size());

    for (size_t i = 0; i < y.size(); ++i)
      d[i]        = (y[i] - t[i]) / (y[i] * (float_t(1) - y[i]));

    return d;
  }
};

// cross-entropy loss function for multi-class classification
class cross_entropy_multiclass {
 public:
  static float_t f(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    float_t d{0.0};

    for (size_t i = 0; i < y.size(); ++i) d += -t[i] * std::log(y[i]);

    return d;
  }

  static vec_t df(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    vec_t d(t.size());

    for (size_t i = 0; i < y.size(); ++i) d[i] = -t[i] / y[i];

    return d;
  }
};

// helper struct
struct _norms
{
    float_t yt = 0.0;
    float_t y2 = 0.0;
    float_t t2 = 0.0;
    float_t y_abs;
    float_t t_abs;

  _norms (const vec_t &y, const vec_t &t) noexcept
  {
    assert(y.size() == t.size());
    float_t m {0.5};
    for (size_t i = 0; i < y.size(); ++i) {
      float_t ti = (t[i] - m);
      float_t yi = (y[i] - m);
      yt += ti * yi;
      t2 += ti * ti;
      y2 += yi * yi;
    }
    y_abs = sqrt(y2);
    t_abs = sqrt(t2);
  }
};

// Loss function based on cosine distances
class cosine {
 public:
  // ( 1 - <y, t> / (|t|*|y|) ) / 2  ( in [0, 1] )
  static float_t f(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    
    float_t m {0.5};
    auto [yt, y2, t2, y_abs, t_abs] = _norms(y, t);

    return (0.0 == y_abs || 0.0 == t_abs) ?
      m :
      m * ( 1.0 - yt / (t_abs * y_abs) );
  }

  // (<y, t> * (y) - |y|^2 * (t)  ) / (|t|*|y|^3) / 2
  static vec_t df(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    vec_t grad(t.size());

    float_t m {0.5};
    auto [yt, y2, t2, y_abs, t_abs] = _norms(y, t);

    float_t d = y2 * y_abs * t_abs;
    for (size_t i = 0; i < y.size(); ++i) {
      float_t ti = (t[i] - m);
      float_t yi = (y[i] - m);
      grad[i]    = (0.0 == y_abs || 0.0 == t_abs) ?
        0.0 :
        (m * (yt * yi - y2 * ti) / d);
    }    

    return grad;
  }
};

// fined cosine loss function
// Modification of previous with adding fine
template<int fine_fract>
class fined_cosine
{
 public:
  // ( 1 - <y, t> / (|t|*|y|)) / 2 + fine * |y|
  static float_t f(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    float_t fine{1.0 / fine_fract};

    float_t m {0.5};
    [[maybe_unused]]
    auto [yt, y2, t2, y_abs, t_abs] = _norms(y, t);

    return (0.0 == y_abs || 0.0 == t_abs) ?
      (m + fine * y_abs) :
      (m * ( 1.0 - yt / (t_abs*y_abs) ) + fine * y_abs);
  }

  // (<y, t> * (y) - |y|^2 * (t)  ) / (|t|*|y|^3) / 2 + fine * (y) / |y|
  static vec_t df(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    vec_t grad(t.size());
    float_t fine{1.0 / fine_fract};

    float_t m {0.5};
    [[maybe_unused]]
    auto [yt, y2, t2, y_abs, t_abs] = _norms(y, t);

    float_t d = y2 * y_abs * t_abs;
    for (size_t i = 0; i < y.size(); ++i) {
      float_t ti = (t[i] - m);
      float_t yi = (y[i] - m);
      grad[i]    = (0.0 == y_abs || 0.0 == t_abs) ?
        ( (0.0 == y_abs) ? 0.0 : fine * yi / y_abs ) :
        ( m * (yt * yi - y2 * ti) / d + fine * yi / y_abs );
    }    

    return grad;
  }
};

using fined_cosine01 = fined_cosine<10>;

// gain loss function
// This is a modification of the cosine distances
class gain {
 public:
  // 2 - <y, t> / <t,t>
  static float_t f(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    float_t yt{0.0};
    float_t t2{0.0};

    for (size_t i = 0; i < y.size(); ++i) {
      float_t ti = 2.0 * (t[i] - 0.5);
      float_t yi = 2.0 * (y[i] - 0.5);
      yt        += ti * yi;
      t2        += ti * ti;
    }

    return (0.0 == t2) ? 2.0 : (2.0 - yt / t2);
  }

  // - (t) / <t,t>
  static vec_t df(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    vec_t grad(t.size());

    float_t t2{0};
    for (size_t i = 0; i < t.size(); ++i) {
      float_t ti = 2.0 * (t[i] - 0.5);
      t2        += ti * ti;
    }    

    for (size_t i = 0; i < t.size(); ++i) {
      float_t ti = 2.0 * (t[i] - 0.5);
      grad[i]    = (0.0 == t2) ? 0.0 :  (- 2.0 * ti / t2);
    }

    return grad;
  }
};


// fined gain loss function
// Modification of previous with adding fine
template<int fine_fract>
class fined_gain
{
 public:
  // 1 - (<y, t> - fine*|y|) / (<t,t> - fine*|t|)
  static float_t f(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    float_t fine{1.0 / fine_fract};

    [[maybe_unused]]
    auto [yt, y2, t2, y_abs, t_abs] = _norms(y, t);

    float_t d = t2 - fine * t_abs;
    return (0.0 == d) ?
      float_t(1.0) :
      float_t(1.0) - (yt - fine * y_abs) / d;
  }

  //  - ( (y) - fine*(y)/|y|) / (<t,t> - fine*|t|)
  static vec_t df(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    float_t fine{1.0 / fine_fract};
    vec_t grad(t.size());

    float_t t2{0.0};
    float_t y2{0.0};
    float_t m {0.5};
    for (size_t i = 0; i < t.size(); ++i) {
      float_t ti = (t[i] - m);
      float_t yi = (y[i] - m);
      t2        += ti * ti;
      y2        += yi * yi;
    } 
    
    float_t y_abs = sqrt(y2);
    float_t t_abs = sqrt(t2);

    float_t d  = t2 - fine * t_abs;
    for (size_t i = 0; i < y.size(); ++i) {
      grad[i] = (0.0 == d || 0.0 == y_abs) ? 
        0.0 :
        (y[i] - m) * (fine / y_abs - 1.0) / d;
    }

    return grad;
  }
};

using fined_gain01 = fined_gain<10>;


template <typename E>
vec_t gradient(const vec_t &y, const vec_t &t) {
  assert(y.size() == t.size());
  return E::df(y, t);
}

template <typename E>
std::vector<vec_t> gradient(const std::vector<vec_t> &y,
                            const std::vector<vec_t> &t) {
  std::vector<vec_t> grads(y.size());

  assert(y.size() == t.size());

  for (size_t i = 0; i < y.size(); i++) grads[i] = gradient<E>(y[i], t[i]);

  return grads;
}

inline void apply_cost_if_defined(std::vector<vec_t> &sample_gradient,
                                  const std::vector<vec_t> &sample_cost) {
  if (sample_gradient.size() == sample_cost.size()) {
    // @todo consider adding parallelism
    const size_t channel_count = sample_gradient.size();
    for (size_t channel = 0; channel < channel_count; ++channel) {
      if (sample_gradient[channel].size() == sample_cost[channel].size()) {
        const size_t element_count = sample_gradient[channel].size();

        // @todo optimize? (use AVX or so)
        for (size_t element = 0; element < element_count; ++element) {
          sample_gradient[channel][element] *= sample_cost[channel][element];
        }
      }
    }
  }
}

// gradient for a minibatch
template <typename E>
std::vector<tensor_t> gradient(const std::vector<tensor_t> &y,
                               const std::vector<tensor_t> &t,
                               const std::vector<tensor_t> &t_cost) {
  const size_t sample_count  = y.size();
  const size_t channel_count = y[0].size();

  std::vector<tensor_t> gradients(sample_count);

  CNN_UNREFERENCED_PARAMETER(channel_count);
  assert(y.size() == t.size());
  assert(t_cost.empty() || t_cost.size() == t.size());

  // @todo add parallelism
  for (size_t sample = 0; sample < sample_count; ++sample) {
    assert(y[sample].size() == channel_count);
    assert(t[sample].size() == channel_count);
    assert(t_cost.empty() || t_cost[sample].empty() ||
           t_cost[sample].size() == channel_count);

    gradients[sample] = gradient<E>(y[sample], t[sample]);

    if (sample < t_cost.size()) {
      apply_cost_if_defined(gradients[sample], t_cost[sample]);
    }
  }

  return gradients;
}

}  // namespace tiny_dnn
