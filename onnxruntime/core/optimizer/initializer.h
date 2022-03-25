// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <vector>
#include <cmath>

#include "core/common/common.h"
#include "core/common/path.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/onnx_protobuf.h"
#include "core/util/math.h"

namespace onnxruntime {

class DataTransferManager;
struct SessionOptions;

class Initializer final {
 public:
  // Construct an initializer with the provided name and data type, with all values initialized to 0
  Initializer(ONNX_NAMESPACE::TensorProto_DataType data_type,
              const std::string& name,
              gsl::span<const int64_t> dims) : dims_(dims.begin(), dims.end()), size_(0) {
    data_type_ = data_type;
    name_ = name;
    size_ = std::accumulate(dims_.begin(), dims_.end(), int64_t(1), std::multiplies<int64_t>{});

    switch (data_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
        float16_data_.assign(static_cast<size_t>(size_), math::floatToHalf(0.f));
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16: {
        // Reuse float16 field
        float16_data_.assign(static_cast<size_t>(size_), BFloat16(0.f).val);
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        float_data_.assign(static_cast<size_t>(size_), 0.0f);
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        double_data_.assign(static_cast<size_t>(size_), 0.0);
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
        int8_data_.assign(static_cast<size_t>(size_), 0);
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {
        uint8_data_.assign(static_cast<size_t>(size_), 0);
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
        int32_data_.assign(static_cast<size_t>(size_), 0);
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
        int64_data_.assign(static_cast<size_t>(size_), 0);
        break;
      }
      default:
        ORT_THROW("data type ", data_type_, "is not supported.");
        break;
    }
  }

  Initializer(const ONNX_NAMESPACE::TensorProto& tensor_proto, const DataTransferManager& dt_manager,
              const SessionOptions& sess_options, const Path& model_path);

  ~Initializer() = default;

  void ToProto(ONNX_NAMESPACE::TensorProto& tensor_proto_result) {
    ONNX_NAMESPACE::TensorProto tensor_proto;
    if (!name_.empty()) {
      tensor_proto.set_name(name_);
    }

    for (auto d : dims_) {
      tensor_proto.add_dims(d);
    }

    tensor_proto.set_data_type(data_type_);

    if (!raw_data_.empty()) {
      tensor_proto.set_raw_data(raw_data_.data(), raw_data_.size());
    } else {
      switch (data_type_) {
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
        case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16: {
          for (int i = 0; i < size_; i++) {
            tensor_proto.add_int32_data(float16_data_[i]);
          }
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
          for (int i = 0; i < size_; i++) {
            tensor_proto.add_float_data(float_data_[i]);
          }
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
          for (int i = 0; i < size_; i++) {
            tensor_proto.add_double_data(double_data_[i]);
          }
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
          for (int i = 0; i < size_; i++) {
            tensor_proto.add_int32_data(int8_data_[i]);
          }
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {
          for (int i = 0; i < size_; i++) {
            tensor_proto.add_int32_data(uint8_data_[i]);
          }
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
          for (int i = 0; i < size_; i++) {
            tensor_proto.add_int32_data(int32_data_[i]);
          }
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
          for (int i = 0; i < size_; i++) {
            tensor_proto.add_int64_data(int64_data_[i]);
          }
          break;
        }
        default:
          ORT_NOT_IMPLEMENTED(__FUNCTION__, "data type is not supported");
          break;
      }
    }
    tensor_proto_result.Swap(&tensor_proto);
  }

  ONNX_NAMESPACE::TensorProto ToFP16(const std::string name) {
    ONNX_NAMESPACE::TensorProto tensor_proto;
    tensor_proto.set_name(name);

    for (auto d : dims_) {
      tensor_proto.add_dims(d);
    }

    tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);

    switch (data_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
        uint16_t* dst = data<uint16_t>();
        for (int i = 0; i < size_; i++) {
          tensor_proto.add_int32_data(dst[i]);
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        float* dst = data<float>();
        for (int i = 0; i < size_; i++) {
          tensor_proto.add_int32_data(math::floatToHalf(dst[i]));
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        double* dst = data<double>();
        for (int i = 0; i < size_; i++) {
          tensor_proto.add_int32_data(math::doubleToHalf(dst[i]));
        }
        break;
      }
      default:
        ORT_NOT_IMPLEMENTED(__FUNCTION__, "data type is not supported");
        break;
    }
    return tensor_proto;
  }

  ONNX_NAMESPACE::TensorProto ToBFloat16(const std::string name) {
    ONNX_NAMESPACE::TensorProto tensor_proto;
    tensor_proto.set_name(name);

    for (auto d : dims_) {
      tensor_proto.add_dims(d);
    }

    tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16);

    switch (data_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        float* dst = data<float>();
        for (int i = 0; i < size_; i++) {
          tensor_proto.add_int32_data(BFloat16(dst[i]).val);
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        double* dst = data<double>();
        for (int i = 0; i < size_; i++) {
          tensor_proto.add_int32_data(math::doubleToHalf(dst[i]));
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16: {
        uint16_t* dst = data<uint16_t>();
        for (int i = 0; i < size_; i++) {
          tensor_proto.add_int32_data(dst[i]);
        }
        break;
      }
      default:
        ORT_NOT_IMPLEMENTED(__FUNCTION__, "data type is not supported");
        break;
    }
    return tensor_proto;
  }

  int data_type() const {
    return data_type_;
  }

  int& data_type() {
    return data_type_;
  }

  const std::string& name() {
    return name_;
  }

  template <typename T>
  T* data() {
    if (!raw_data_.empty()) {
      return reinterpret_cast<T*>(raw_data_.data());
    }
    switch (data_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16: {
        return reinterpret_cast<T*>(float16_data_.data());
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        return reinterpret_cast<T*>(float_data_.data());
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        return reinterpret_cast<T*>(double_data_.data());
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
        return reinterpret_cast<T*>(int8_data_.data());
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {
        return reinterpret_cast<T*>(uint8_data_.data());
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
        return reinterpret_cast<T*>(int32_data_.data());
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
        return reinterpret_cast<T*>(int64_data_.data());
        break;
      }
      default:
        break;
    }

    return nullptr;
  }

  template <typename T>
  const T* data() const {
    if (!raw_data_.empty()) {
      return reinterpret_cast<const T*>(raw_data_.data());
    }
    switch (data_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16: {
        return reinterpret_cast<const T*>(float16_data_.data());
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        return reinterpret_cast<const T*>(float_data_.data());
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        return reinterpret_cast<const T*>(double_data_.data());
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
        return reinterpret_cast<const T*>(int8_data_.data());
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {
        return reinterpret_cast<const T*>(uint8_data_.data());
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
        return reinterpret_cast<const T*>(int32_data_.data());
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
        return reinterpret_cast<const T*>(int64_data_.data());
        break;
      }
      default:
        break;
    }

    return nullptr;
  }

  gsl::span<const int64_t> dims() const {
    return dims_;
  }

  int64_t size() const { return size_; }

  Initializer& add(float value) {
    int64_t n = size();
    switch (data_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
        uint16_t* dst = data<uint16_t>();
        for (int i = 0; i < n; i++) {
          dst[i] = math::floatToHalf(math::halfToFloat(dst[i]) + value);
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16: {
        uint16_t* dst = data<uint16_t>();
        for (int i = 0; i < n; i++) {
          dst[i] = BFloat16((reinterpret_cast<BFloat16*>(dst + i))->ToFloat() + value).val;
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        float* dst = data<float>();
        for (int i = 0; i < n; i++) {
          dst[i] += value;
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        double* dst = data<double>();
        for (int i = 0; i < n; i++) {
          dst[i] += value;
        }
        break;
      }
      default:
        break;
    }
    return *this;
  }

  Initializer& add(const Initializer& other) {
    int64_t n = size();
    switch (data_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
        uint16_t* dst = data<uint16_t>();
        const uint16_t* src = other.data<uint16_t>();
        for (int i = 0; i < n; i++) {
          dst[i] = math::floatToHalf(math::halfToFloat(dst[i]) + math::halfToFloat(src[i]));
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16: {
        uint16_t* dst = data<uint16_t>();
        const uint16_t* src = other.data<uint16_t>();
        for (int i = 0; i < n; i++) {
          dst[i] = BFloat16((reinterpret_cast<BFloat16*>(dst + i))->ToFloat() + (reinterpret_cast<const BFloat16*>(src + i))->ToFloat()).val;
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        float* dst = data<float>();
        const float* src = other.data<float>();
        for (int i = 0; i < n; i++) {
          dst[i] += src[i];
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        double* dst = data<double>();
        const double* src = other.data<double>();
        for (int i = 0; i < n; i++) {
          dst[i] += src[i];
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
        int32_t* dst = data<int32_t>();
        const int32_t* src = other.data<int32_t>();
        for (int i = 0; i < n; i++) {
          dst[i] += src[i];
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
        int64_t* dst = data<int64_t>();
        const int64_t* src = other.data<int64_t>();
        for (int i = 0; i < n; i++) {
          dst[i] += src[i];
        }
        break;
      }
      default:
        break;
    }
    return *this;
  }
  Initializer& sub(const Initializer& other) {
    int64_t n = size();
    switch (data_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
        uint16_t* dst = data<uint16_t>();
        const uint16_t* src = other.data<uint16_t>();
        for (int i = 0; i < n; i++) {
          dst[i] = math::floatToHalf(math::halfToFloat(dst[i]) - math::halfToFloat(src[i]));
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16: {
        uint16_t* dst = data<uint16_t>();
        const uint16_t* src = other.data<uint16_t>();
        for (int i = 0; i < n; i++) {
          dst[i] = BFloat16((reinterpret_cast<BFloat16*>(dst + i))->ToFloat() - (reinterpret_cast<const BFloat16*>(src + i))->ToFloat()).val;
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        float* dst = data<float>();
        const float* src = other.data<float>();
        for (int i = 0; i < n; i++) {
          dst[i] -= src[i];
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        double* dst = data<double>();
        const double* src = other.data<double>();
        for (int i = 0; i < n; i++) {
          dst[i] -= src[i];
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
        int32_t* dst = data<int32_t>();
        const int32_t* src = other.data<int32_t>();
        for (int i = 0; i < n; i++) {
          dst[i] -= src[i];
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
        int64_t* dst = data<int64_t>();
        const int64_t* src = other.data<int64_t>();
        for (int i = 0; i < n; i++) {
          dst[i] -= src[i];
        }
        break;
      }
      default:
        break;
    }
    return *this;
  }

  Initializer& mul(const Initializer& other) {
    int64_t n = size();
    switch (data_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
        uint16_t* dst = data<uint16_t>();
        const uint16_t* src = other.data<uint16_t>();
        for (int i = 0; i < n; i++) {
          dst[i] = math::floatToHalf(math::halfToFloat(dst[i]) * math::halfToFloat(src[i]));
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16: {
        uint16_t* dst = data<uint16_t>();
        const uint16_t* src = other.data<uint16_t>();
        for (int i = 0; i < n; i++) {
          dst[i] = BFloat16((reinterpret_cast<BFloat16*>(dst + i))->ToFloat() * (reinterpret_cast<const BFloat16*>(src + i))->ToFloat()).val;
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        float* dst = data<float>();
        const float* src = other.data<float>();
        for (int i = 0; i < n; i++) {
          dst[i] *= src[i];
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        double* dst = data<double>();
        const double* src = other.data<double>();
        for (int i = 0; i < n; i++) {
          dst[i] *= src[i];
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
        int32_t* dst = data<int32_t>();
        const int32_t* src = other.data<int32_t>();
        for (int i = 0; i < n; i++) {
          dst[i] *= src[i];
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
        int64_t* dst = data<int64_t>();
        const int64_t* src = other.data<int64_t>();
        for (int i = 0; i < n; i++) {
          dst[i] *= src[i];
        }
        break;
      }
      default:
        break;
    }
    return *this;
  }
  Initializer& div(const Initializer& other) {
    int64_t n = size();
    switch (data_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
        uint16_t* dst = data<uint16_t>();
        const uint16_t* src = other.data<uint16_t>();
        for (int i = 0; i < n; i++) {
          dst[i] = math::floatToHalf(math::halfToFloat(dst[i]) / math::halfToFloat(src[i]));
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16: {
        uint16_t* dst = data<uint16_t>();
        const uint16_t* src = other.data<uint16_t>();
        for (int i = 0; i < n; i++) {
          dst[i] = BFloat16((reinterpret_cast<BFloat16*>(dst + i))->ToFloat() / (reinterpret_cast<const BFloat16*>(src + i))->ToFloat()).val;
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        float* dst = data<float>();
        const float* src = other.data<float>();
        for (int i = 0; i < n; i++) {
          dst[i] /= src[i];
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        double* dst = data<double>();
        const double* src = other.data<double>();
        for (int i = 0; i < n; i++) {
          dst[i] /= src[i];
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
        int32_t* dst = data<int32_t>();
        const int32_t* src = other.data<int32_t>();
        for (int i = 0; i < n; i++) {
          dst[i] /= src[i];
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
        int64_t* dst = data<int64_t>();
        const int64_t* src = other.data<int64_t>();
        for (int i = 0; i < n; i++) {
          dst[i] /= src[i];
        }
        break;
      }
      default:
        break;
    }
    return *this;
  }

  Initializer& sqrt() {
    int64_t n = size();
    switch (data_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
        uint16_t* dst = data<uint16_t>();
        for (int i = 0; i < n; i++) {
          dst[i] = math::floatToHalf(std::sqrt(math::halfToFloat(dst[i])));
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16: {
        uint16_t* dst = data<uint16_t>();
        for (int i = 0; i < n; i++) {
          dst[i] = BFloat16(std::sqrt((reinterpret_cast<BFloat16*>(dst + i))->ToFloat())).val;
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        float* dst = data<float>();
        for (int i = 0; i < n; i++) {
          dst[i] = std::sqrt(dst[i]);
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        double* dst = data<double>();
        for (int i = 0; i < n; i++) {
          dst[i] = std::sqrt(dst[i]);
        }
        break;
      }
      default:
        break;
    }
    return *this;
  }

  inline void scale_by_axis(const Initializer& other, int axis) {
    int64_t num = 1;
    for (size_t k = axis; k < dims_.size(); k++) {
      num *= dims_[k];
    }

    int64_t n = size() / num;
    switch (data_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
        uint16_t* dst = data<uint16_t>();
        const uint16_t* src = other.data<uint16_t>();
        for (int i = 0; i < n; i++) {
          int index = other.size() == 1 ? 0 : i;
          for (int64_t j = 0; j < num; j++) {
            auto k = i * num + j;
            dst[k] = math::floatToHalf(math::halfToFloat(dst[k]) * math::halfToFloat(src[index]));
          }
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16: {
        uint16_t* dst = data<uint16_t>();
        const uint16_t* src = other.data<uint16_t>();
        for (int i = 0; i < n; i++) {
          int index = other.size() == 1 ? 0 : i;
          for (int64_t j = 0; j < num; j++) {
            auto k = i * num + j;
            dst[k] = BFloat16((reinterpret_cast<BFloat16*>(dst + k))->ToFloat() * (reinterpret_cast<const BFloat16*>(src + index))->ToFloat()).val;
          }
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        float* dst = data<float>();
        const float* src = other.data<float>();
        for (int i = 0; i < n; i++) {
          int index = other.size() == 1 ? 0 : i;
          for (int64_t j = 0; j < num; j++) {
            dst[i * num + j] *= src[index];
          }
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        double* dst = data<double>();
        const double* src = other.data<double>();
        for (int i = 0; i < n; i++) {
          int index = other.size() == 1 ? 0 : i;
          for (int64_t j = 0; j < num; j++) {
            dst[i * num + j] *= src[index];
          }
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
        int32_t* dst = data<int32_t>();
        const int32_t* src = other.data<int32_t>();
        for (int i = 0; i < n; i++) {
          int index = other.size() == 1 ? 0 : i;
          for (int64_t j = 0; j < num; j++) {
            dst[i * num + j] *= src[index];
          }
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
        int64_t* dst = data<int64_t>();
        const int64_t* src = other.data<int64_t>();
        for (int i = 0; i < n; i++) {
          int index = other.size() == 1 ? 0 : i;
          for (int64_t j = 0; j < num; j++) {
            dst[i * num + j] *= src[index];
          }
        }
        break;
      }
      default:
        break;
    }
  }

 private:
#if !defined(ORT_MINIMAL_BUILD)
  static Status ReadExternalRawData(
      const ONNX_NAMESPACE::TensorProto& tensor_proto, const Path& model_path, std::vector<char>& raw_data);
#endif

  int data_type_;
  std::string name_;
  TensorShapeVector dims_;
  int64_t size_;

  std::vector<char> raw_data_;
  std::vector<float> float_data_;
  std::vector<uint16_t> float16_data_;
  std::vector<double> double_data_;
  std::vector<int8_t> int8_data_;
  std::vector<uint8_t> uint8_data_;
  std::vector<int32_t> int32_data_;
  std::vector<int64_t> int64_data_;
};

}  // namespace onnxruntime
