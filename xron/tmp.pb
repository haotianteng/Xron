node {
  name: "Placeholder"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 400
        }
        dim {
          size: 400
        }
      }
    }
  }
}
node {
  name: "Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\220\001\000\000\220\001\000\000"
      }
    }
  }
}
node {
  name: "strided_slice/stack"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "strided_slice/stack_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "strided_slice/stack_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "strided_slice"
  op: "StridedSlice"
  input: "Shape"
  input: "strided_slice/stack"
  input: "strided_slice/stack_1"
  input: "strided_slice/stack_2"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "begin_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "ellipsis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "end_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "new_axis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "shrink_axis_mask"
    value {
      i: 1
    }
  }
}
node {
  name: "Reshape/shape/1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "Reshape/shape/2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 400
      }
    }
  }
}
node {
  name: "Reshape/shape/3"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "Reshape/shape"
  op: "Pack"
  input: "strided_slice"
  input: "Reshape/shape/1"
  input: "Reshape/shape/2"
  input: "Reshape/shape/3"
  attr {
    key: "N"
    value {
      i: 4
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "Reshape"
  op: "Reshape"
  input: "Placeholder"
  input: "Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "res_layer1/branch1/conv1/weights"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer1/branch1/conv1/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer1/branch1/conv1/weights/read"
  op: "Identity"
  input: "res_layer1/branch1/conv1/weights"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer1/branch1/conv1/weights"
      }
    }
  }
}
node {
  name: "res_layer1/branch1/conv1/conv1"
  op: "Conv2D"
  input: "Reshape"
  input: "res_layer1/branch1/conv1/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer1/branch1/conv1_bn/conv1_bn_moments/mean/reduction_indices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 3
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "res_layer1/branch1/conv1_bn/conv1_bn_moments/mean"
  op: "Mean"
  input: "res_layer1/branch1/conv1/conv1"
  input: "res_layer1/branch1/conv1_bn/conv1_bn_moments/mean/reduction_indices"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer1/branch1/conv1_bn/conv1_bn_moments/StopGradient"
  op: "StopGradient"
  input: "res_layer1/branch1/conv1_bn/conv1_bn_moments/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch1/conv1_bn/conv1_bn_moments/SquaredDifference"
  op: "SquaredDifference"
  input: "res_layer1/branch1/conv1/conv1"
  input: "res_layer1/branch1/conv1_bn/conv1_bn_moments/StopGradient"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch1/conv1_bn/conv1_bn_moments/variance/reduction_indices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 3
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "res_layer1/branch1/conv1_bn/conv1_bn_moments/variance"
  op: "Mean"
  input: "res_layer1/branch1/conv1_bn/conv1_bn_moments/SquaredDifference"
  input: "res_layer1/branch1/conv1_bn/conv1_bn_moments/variance/reduction_indices"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer1/branch1/conv1_bn/conv1_bn_moments/Squeeze"
  op: "Squeeze"
  input: "res_layer1/branch1/conv1_bn/conv1_bn_moments/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "squeeze_dims"
    value {
      list {
        i: 0
        i: 1
        i: 2
      }
    }
  }
}
node {
  name: "res_layer1/branch1/conv1_bn/conv1_bn_moments/Squeeze_1"
  op: "Squeeze"
  input: "res_layer1/branch1/conv1_bn/conv1_bn_moments/variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "squeeze_dims"
    value {
      list {
        i: 0
        i: 1
        i: 2
      }
    }
  }
}
node {
  name: "res_layer1/branch1/conv1_bn/conv1_bn_scale"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer1/branch1/conv1_bn/conv1_bn_scale"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer1/branch1/conv1_bn/conv1_bn_scale/read"
  op: "Identity"
  input: "res_layer1/branch1/conv1_bn/conv1_bn_scale"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer1/branch1/conv1_bn/conv1_bn_scale"
      }
    }
  }
}
node {
  name: "res_layer1/branch1/conv1_bn/conv1_bn_offset"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer1/branch1/conv1_bn/conv1_bn_offset"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer1/branch1/conv1_bn/conv1_bn_offset/read"
  op: "Identity"
  input: "res_layer1/branch1/conv1_bn/conv1_bn_offset"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer1/branch1/conv1_bn/conv1_bn_offset"
      }
    }
  }
}
node {
  name: "res_layer1/branch1/conv1_bn/batchnorm/add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 9.999999747378752e-06
      }
    }
  }
}
node {
  name: "res_layer1/branch1/conv1_bn/batchnorm/add"
  op: "Add"
  input: "res_layer1/branch1/conv1_bn/conv1_bn_moments/Squeeze_1"
  input: "res_layer1/branch1/conv1_bn/batchnorm/add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch1/conv1_bn/batchnorm/Rsqrt"
  op: "Rsqrt"
  input: "res_layer1/branch1/conv1_bn/batchnorm/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch1/conv1_bn/batchnorm/mul"
  op: "Mul"
  input: "res_layer1/branch1/conv1_bn/batchnorm/Rsqrt"
  input: "res_layer1/branch1/conv1_bn/conv1_bn_scale/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch1/conv1_bn/batchnorm/mul_1"
  op: "Mul"
  input: "res_layer1/branch1/conv1/conv1"
  input: "res_layer1/branch1/conv1_bn/batchnorm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch1/conv1_bn/batchnorm/mul_2"
  op: "Mul"
  input: "res_layer1/branch1/conv1_bn/conv1_bn_moments/Squeeze"
  input: "res_layer1/branch1/conv1_bn/batchnorm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch1/conv1_bn/batchnorm/sub"
  op: "Sub"
  input: "res_layer1/branch1/conv1_bn/conv1_bn_offset/read"
  input: "res_layer1/branch1/conv1_bn/batchnorm/mul_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch1/conv1_bn/batchnorm/add_1"
  op: "Add"
  input: "res_layer1/branch1/conv1_bn/batchnorm/mul_1"
  input: "res_layer1/branch1/conv1_bn/batchnorm/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch2/conv2a/weights"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer1/branch2/conv2a/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer1/branch2/conv2a/weights/read"
  op: "Identity"
  input: "res_layer1/branch2/conv2a/weights"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer1/branch2/conv2a/weights"
      }
    }
  }
}
node {
  name: "res_layer1/branch2/conv2a/conv2a"
  op: "Conv2D"
  input: "Reshape"
  input: "res_layer1/branch2/conv2a/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer1/branch2/conv2a_bn/conv2a_bn_moments/mean/reduction_indices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 3
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "res_layer1/branch2/conv2a_bn/conv2a_bn_moments/mean"
  op: "Mean"
  input: "res_layer1/branch2/conv2a/conv2a"
  input: "res_layer1/branch2/conv2a_bn/conv2a_bn_moments/mean/reduction_indices"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer1/branch2/conv2a_bn/conv2a_bn_moments/StopGradient"
  op: "StopGradient"
  input: "res_layer1/branch2/conv2a_bn/conv2a_bn_moments/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch2/conv2a_bn/conv2a_bn_moments/SquaredDifference"
  op: "SquaredDifference"
  input: "res_layer1/branch2/conv2a/conv2a"
  input: "res_layer1/branch2/conv2a_bn/conv2a_bn_moments/StopGradient"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch2/conv2a_bn/conv2a_bn_moments/variance/reduction_indices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 3
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "res_layer1/branch2/conv2a_bn/conv2a_bn_moments/variance"
  op: "Mean"
  input: "res_layer1/branch2/conv2a_bn/conv2a_bn_moments/SquaredDifference"
  input: "res_layer1/branch2/conv2a_bn/conv2a_bn_moments/variance/reduction_indices"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer1/branch2/conv2a_bn/conv2a_bn_moments/Squeeze"
  op: "Squeeze"
  input: "res_layer1/branch2/conv2a_bn/conv2a_bn_moments/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "squeeze_dims"
    value {
      list {
        i: 0
        i: 1
        i: 2
      }
    }
  }
}
node {
  name: "res_layer1/branch2/conv2a_bn/conv2a_bn_moments/Squeeze_1"
  op: "Squeeze"
  input: "res_layer1/branch2/conv2a_bn/conv2a_bn_moments/variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "squeeze_dims"
    value {
      list {
        i: 0
        i: 1
        i: 2
      }
    }
  }
}
node {
  name: "res_layer1/branch2/conv2a_bn/conv2a_bn_scale"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer1/branch2/conv2a_bn/conv2a_bn_scale"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer1/branch2/conv2a_bn/conv2a_bn_scale/read"
  op: "Identity"
  input: "res_layer1/branch2/conv2a_bn/conv2a_bn_scale"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer1/branch2/conv2a_bn/conv2a_bn_scale"
      }
    }
  }
}
node {
  name: "res_layer1/branch2/conv2a_bn/conv2a_bn_offset"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer1/branch2/conv2a_bn/conv2a_bn_offset"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer1/branch2/conv2a_bn/conv2a_bn_offset/read"
  op: "Identity"
  input: "res_layer1/branch2/conv2a_bn/conv2a_bn_offset"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer1/branch2/conv2a_bn/conv2a_bn_offset"
      }
    }
  }
}
node {
  name: "res_layer1/branch2/conv2a_bn/batchnorm/add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 9.999999747378752e-06
      }
    }
  }
}
node {
  name: "res_layer1/branch2/conv2a_bn/batchnorm/add"
  op: "Add"
  input: "res_layer1/branch2/conv2a_bn/conv2a_bn_moments/Squeeze_1"
  input: "res_layer1/branch2/conv2a_bn/batchnorm/add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch2/conv2a_bn/batchnorm/Rsqrt"
  op: "Rsqrt"
  input: "res_layer1/branch2/conv2a_bn/batchnorm/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch2/conv2a_bn/batchnorm/mul"
  op: "Mul"
  input: "res_layer1/branch2/conv2a_bn/batchnorm/Rsqrt"
  input: "res_layer1/branch2/conv2a_bn/conv2a_bn_scale/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch2/conv2a_bn/batchnorm/mul_1"
  op: "Mul"
  input: "res_layer1/branch2/conv2a/conv2a"
  input: "res_layer1/branch2/conv2a_bn/batchnorm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch2/conv2a_bn/batchnorm/mul_2"
  op: "Mul"
  input: "res_layer1/branch2/conv2a_bn/conv2a_bn_moments/Squeeze"
  input: "res_layer1/branch2/conv2a_bn/batchnorm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch2/conv2a_bn/batchnorm/sub"
  op: "Sub"
  input: "res_layer1/branch2/conv2a_bn/conv2a_bn_offset/read"
  input: "res_layer1/branch2/conv2a_bn/batchnorm/mul_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch2/conv2a_bn/batchnorm/add_1"
  op: "Add"
  input: "res_layer1/branch2/conv2a_bn/batchnorm/mul_1"
  input: "res_layer1/branch2/conv2a_bn/batchnorm/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch2/conv2a_relu/relu"
  op: "Relu"
  input: "res_layer1/branch2/conv2a_bn/batchnorm/add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch2/conv2b/weights"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer1/branch2/conv2b/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 3
        }
        dim {
          size: 256
        }
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer1/branch2/conv2b/weights/read"
  op: "Identity"
  input: "res_layer1/branch2/conv2b/weights"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer1/branch2/conv2b/weights"
      }
    }
  }
}
node {
  name: "res_layer1/branch2/conv2b/conv2b"
  op: "Conv2D"
  input: "res_layer1/branch2/conv2a_relu/relu"
  input: "res_layer1/branch2/conv2b/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer1/branch2/conv2b_bn/conv2b_bn_moments/mean/reduction_indices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 3
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "res_layer1/branch2/conv2b_bn/conv2b_bn_moments/mean"
  op: "Mean"
  input: "res_layer1/branch2/conv2b/conv2b"
  input: "res_layer1/branch2/conv2b_bn/conv2b_bn_moments/mean/reduction_indices"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer1/branch2/conv2b_bn/conv2b_bn_moments/StopGradient"
  op: "StopGradient"
  input: "res_layer1/branch2/conv2b_bn/conv2b_bn_moments/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch2/conv2b_bn/conv2b_bn_moments/SquaredDifference"
  op: "SquaredDifference"
  input: "res_layer1/branch2/conv2b/conv2b"
  input: "res_layer1/branch2/conv2b_bn/conv2b_bn_moments/StopGradient"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch2/conv2b_bn/conv2b_bn_moments/variance/reduction_indices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 3
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "res_layer1/branch2/conv2b_bn/conv2b_bn_moments/variance"
  op: "Mean"
  input: "res_layer1/branch2/conv2b_bn/conv2b_bn_moments/SquaredDifference"
  input: "res_layer1/branch2/conv2b_bn/conv2b_bn_moments/variance/reduction_indices"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer1/branch2/conv2b_bn/conv2b_bn_moments/Squeeze"
  op: "Squeeze"
  input: "res_layer1/branch2/conv2b_bn/conv2b_bn_moments/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "squeeze_dims"
    value {
      list {
        i: 0
        i: 1
        i: 2
      }
    }
  }
}
node {
  name: "res_layer1/branch2/conv2b_bn/conv2b_bn_moments/Squeeze_1"
  op: "Squeeze"
  input: "res_layer1/branch2/conv2b_bn/conv2b_bn_moments/variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "squeeze_dims"
    value {
      list {
        i: 0
        i: 1
        i: 2
      }
    }
  }
}
node {
  name: "res_layer1/branch2/conv2b_bn/conv2b_bn_scale"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer1/branch2/conv2b_bn/conv2b_bn_scale"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer1/branch2/conv2b_bn/conv2b_bn_scale/read"
  op: "Identity"
  input: "res_layer1/branch2/conv2b_bn/conv2b_bn_scale"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer1/branch2/conv2b_bn/conv2b_bn_scale"
      }
    }
  }
}
node {
  name: "res_layer1/branch2/conv2b_bn/conv2b_bn_offset"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer1/branch2/conv2b_bn/conv2b_bn_offset"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer1/branch2/conv2b_bn/conv2b_bn_offset/read"
  op: "Identity"
  input: "res_layer1/branch2/conv2b_bn/conv2b_bn_offset"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer1/branch2/conv2b_bn/conv2b_bn_offset"
      }
    }
  }
}
node {
  name: "res_layer1/branch2/conv2b_bn/batchnorm/add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 9.999999747378752e-06
      }
    }
  }
}
node {
  name: "res_layer1/branch2/conv2b_bn/batchnorm/add"
  op: "Add"
  input: "res_layer1/branch2/conv2b_bn/conv2b_bn_moments/Squeeze_1"
  input: "res_layer1/branch2/conv2b_bn/batchnorm/add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch2/conv2b_bn/batchnorm/Rsqrt"
  op: "Rsqrt"
  input: "res_layer1/branch2/conv2b_bn/batchnorm/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch2/conv2b_bn/batchnorm/mul"
  op: "Mul"
  input: "res_layer1/branch2/conv2b_bn/batchnorm/Rsqrt"
  input: "res_layer1/branch2/conv2b_bn/conv2b_bn_scale/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch2/conv2b_bn/batchnorm/mul_1"
  op: "Mul"
  input: "res_layer1/branch2/conv2b/conv2b"
  input: "res_layer1/branch2/conv2b_bn/batchnorm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch2/conv2b_bn/batchnorm/mul_2"
  op: "Mul"
  input: "res_layer1/branch2/conv2b_bn/conv2b_bn_moments/Squeeze"
  input: "res_layer1/branch2/conv2b_bn/batchnorm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch2/conv2b_bn/batchnorm/sub"
  op: "Sub"
  input: "res_layer1/branch2/conv2b_bn/conv2b_bn_offset/read"
  input: "res_layer1/branch2/conv2b_bn/batchnorm/mul_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch2/conv2b_bn/batchnorm/add_1"
  op: "Add"
  input: "res_layer1/branch2/conv2b_bn/batchnorm/mul_1"
  input: "res_layer1/branch2/conv2b_bn/batchnorm/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch2/conv2b_relu/relu"
  op: "Relu"
  input: "res_layer1/branch2/conv2b_bn/batchnorm/add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch2/conv2c/weights"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer1/branch2/conv2c/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 256
        }
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer1/branch2/conv2c/weights/read"
  op: "Identity"
  input: "res_layer1/branch2/conv2c/weights"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer1/branch2/conv2c/weights"
      }
    }
  }
}
node {
  name: "res_layer1/branch2/conv2c/conv2c"
  op: "Conv2D"
  input: "res_layer1/branch2/conv2b_relu/relu"
  input: "res_layer1/branch2/conv2c/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer1/branch2/conv2c_bn/conv2c_bn_moments/mean/reduction_indices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 3
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "res_layer1/branch2/conv2c_bn/conv2c_bn_moments/mean"
  op: "Mean"
  input: "res_layer1/branch2/conv2c/conv2c"
  input: "res_layer1/branch2/conv2c_bn/conv2c_bn_moments/mean/reduction_indices"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer1/branch2/conv2c_bn/conv2c_bn_moments/StopGradient"
  op: "StopGradient"
  input: "res_layer1/branch2/conv2c_bn/conv2c_bn_moments/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch2/conv2c_bn/conv2c_bn_moments/SquaredDifference"
  op: "SquaredDifference"
  input: "res_layer1/branch2/conv2c/conv2c"
  input: "res_layer1/branch2/conv2c_bn/conv2c_bn_moments/StopGradient"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch2/conv2c_bn/conv2c_bn_moments/variance/reduction_indices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 3
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "res_layer1/branch2/conv2c_bn/conv2c_bn_moments/variance"
  op: "Mean"
  input: "res_layer1/branch2/conv2c_bn/conv2c_bn_moments/SquaredDifference"
  input: "res_layer1/branch2/conv2c_bn/conv2c_bn_moments/variance/reduction_indices"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer1/branch2/conv2c_bn/conv2c_bn_moments/Squeeze"
  op: "Squeeze"
  input: "res_layer1/branch2/conv2c_bn/conv2c_bn_moments/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "squeeze_dims"
    value {
      list {
        i: 0
        i: 1
        i: 2
      }
    }
  }
}
node {
  name: "res_layer1/branch2/conv2c_bn/conv2c_bn_moments/Squeeze_1"
  op: "Squeeze"
  input: "res_layer1/branch2/conv2c_bn/conv2c_bn_moments/variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "squeeze_dims"
    value {
      list {
        i: 0
        i: 1
        i: 2
      }
    }
  }
}
node {
  name: "res_layer1/branch2/conv2c_bn/conv2c_bn_scale"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer1/branch2/conv2c_bn/conv2c_bn_scale"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer1/branch2/conv2c_bn/conv2c_bn_scale/read"
  op: "Identity"
  input: "res_layer1/branch2/conv2c_bn/conv2c_bn_scale"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer1/branch2/conv2c_bn/conv2c_bn_scale"
      }
    }
  }
}
node {
  name: "res_layer1/branch2/conv2c_bn/conv2c_bn_offset"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer1/branch2/conv2c_bn/conv2c_bn_offset"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer1/branch2/conv2c_bn/conv2c_bn_offset/read"
  op: "Identity"
  input: "res_layer1/branch2/conv2c_bn/conv2c_bn_offset"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer1/branch2/conv2c_bn/conv2c_bn_offset"
      }
    }
  }
}
node {
  name: "res_layer1/branch2/conv2c_bn/batchnorm/add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 9.999999747378752e-06
      }
    }
  }
}
node {
  name: "res_layer1/branch2/conv2c_bn/batchnorm/add"
  op: "Add"
  input: "res_layer1/branch2/conv2c_bn/conv2c_bn_moments/Squeeze_1"
  input: "res_layer1/branch2/conv2c_bn/batchnorm/add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch2/conv2c_bn/batchnorm/Rsqrt"
  op: "Rsqrt"
  input: "res_layer1/branch2/conv2c_bn/batchnorm/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch2/conv2c_bn/batchnorm/mul"
  op: "Mul"
  input: "res_layer1/branch2/conv2c_bn/batchnorm/Rsqrt"
  input: "res_layer1/branch2/conv2c_bn/conv2c_bn_scale/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch2/conv2c_bn/batchnorm/mul_1"
  op: "Mul"
  input: "res_layer1/branch2/conv2c/conv2c"
  input: "res_layer1/branch2/conv2c_bn/batchnorm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch2/conv2c_bn/batchnorm/mul_2"
  op: "Mul"
  input: "res_layer1/branch2/conv2c_bn/conv2c_bn_moments/Squeeze"
  input: "res_layer1/branch2/conv2c_bn/batchnorm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch2/conv2c_bn/batchnorm/sub"
  op: "Sub"
  input: "res_layer1/branch2/conv2c_bn/conv2c_bn_offset/read"
  input: "res_layer1/branch2/conv2c_bn/batchnorm/mul_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/branch2/conv2c_bn/batchnorm/add_1"
  op: "Add"
  input: "res_layer1/branch2/conv2c_bn/batchnorm/mul_1"
  input: "res_layer1/branch2/conv2c_bn/batchnorm/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/plus/add"
  op: "Add"
  input: "res_layer1/branch1/conv1_bn/batchnorm/add_1"
  input: "res_layer1/branch2/conv2c_bn/batchnorm/add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer1/plus/final_relu"
  op: "Relu"
  input: "res_layer1/plus/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer2/branch1/conv1/weights"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer2/branch1/conv1/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 256
        }
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer2/branch1/conv1/weights/read"
  op: "Identity"
  input: "res_layer2/branch1/conv1/weights"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer2/branch1/conv1/weights"
      }
    }
  }
}
node {
  name: "res_layer2/branch1/conv1/conv1"
  op: "Conv2D"
  input: "res_layer1/plus/final_relu"
  input: "res_layer2/branch1/conv1/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer2/branch2/conv2a/weights"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer2/branch2/conv2a/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 256
        }
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer2/branch2/conv2a/weights/read"
  op: "Identity"
  input: "res_layer2/branch2/conv2a/weights"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer2/branch2/conv2a/weights"
      }
    }
  }
}
node {
  name: "res_layer2/branch2/conv2a/conv2a"
  op: "Conv2D"
  input: "res_layer1/plus/final_relu"
  input: "res_layer2/branch2/conv2a/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer2/branch2/conv2a_bn/conv2a_bn_moments/mean/reduction_indices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 3
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "res_layer2/branch2/conv2a_bn/conv2a_bn_moments/mean"
  op: "Mean"
  input: "res_layer2/branch2/conv2a/conv2a"
  input: "res_layer2/branch2/conv2a_bn/conv2a_bn_moments/mean/reduction_indices"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer2/branch2/conv2a_bn/conv2a_bn_moments/StopGradient"
  op: "StopGradient"
  input: "res_layer2/branch2/conv2a_bn/conv2a_bn_moments/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer2/branch2/conv2a_bn/conv2a_bn_moments/SquaredDifference"
  op: "SquaredDifference"
  input: "res_layer2/branch2/conv2a/conv2a"
  input: "res_layer2/branch2/conv2a_bn/conv2a_bn_moments/StopGradient"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer2/branch2/conv2a_bn/conv2a_bn_moments/variance/reduction_indices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 3
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "res_layer2/branch2/conv2a_bn/conv2a_bn_moments/variance"
  op: "Mean"
  input: "res_layer2/branch2/conv2a_bn/conv2a_bn_moments/SquaredDifference"
  input: "res_layer2/branch2/conv2a_bn/conv2a_bn_moments/variance/reduction_indices"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer2/branch2/conv2a_bn/conv2a_bn_moments/Squeeze"
  op: "Squeeze"
  input: "res_layer2/branch2/conv2a_bn/conv2a_bn_moments/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "squeeze_dims"
    value {
      list {
        i: 0
        i: 1
        i: 2
      }
    }
  }
}
node {
  name: "res_layer2/branch2/conv2a_bn/conv2a_bn_moments/Squeeze_1"
  op: "Squeeze"
  input: "res_layer2/branch2/conv2a_bn/conv2a_bn_moments/variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "squeeze_dims"
    value {
      list {
        i: 0
        i: 1
        i: 2
      }
    }
  }
}
node {
  name: "res_layer2/branch2/conv2a_bn/conv2a_bn_scale"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer2/branch2/conv2a_bn/conv2a_bn_scale"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer2/branch2/conv2a_bn/conv2a_bn_scale/read"
  op: "Identity"
  input: "res_layer2/branch2/conv2a_bn/conv2a_bn_scale"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer2/branch2/conv2a_bn/conv2a_bn_scale"
      }
    }
  }
}
node {
  name: "res_layer2/branch2/conv2a_bn/conv2a_bn_offset"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer2/branch2/conv2a_bn/conv2a_bn_offset"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer2/branch2/conv2a_bn/conv2a_bn_offset/read"
  op: "Identity"
  input: "res_layer2/branch2/conv2a_bn/conv2a_bn_offset"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer2/branch2/conv2a_bn/conv2a_bn_offset"
      }
    }
  }
}
node {
  name: "res_layer2/branch2/conv2a_bn/batchnorm/add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 9.999999747378752e-06
      }
    }
  }
}
node {
  name: "res_layer2/branch2/conv2a_bn/batchnorm/add"
  op: "Add"
  input: "res_layer2/branch2/conv2a_bn/conv2a_bn_moments/Squeeze_1"
  input: "res_layer2/branch2/conv2a_bn/batchnorm/add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer2/branch2/conv2a_bn/batchnorm/Rsqrt"
  op: "Rsqrt"
  input: "res_layer2/branch2/conv2a_bn/batchnorm/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer2/branch2/conv2a_bn/batchnorm/mul"
  op: "Mul"
  input: "res_layer2/branch2/conv2a_bn/batchnorm/Rsqrt"
  input: "res_layer2/branch2/conv2a_bn/conv2a_bn_scale/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer2/branch2/conv2a_bn/batchnorm/mul_1"
  op: "Mul"
  input: "res_layer2/branch2/conv2a/conv2a"
  input: "res_layer2/branch2/conv2a_bn/batchnorm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer2/branch2/conv2a_bn/batchnorm/mul_2"
  op: "Mul"
  input: "res_layer2/branch2/conv2a_bn/conv2a_bn_moments/Squeeze"
  input: "res_layer2/branch2/conv2a_bn/batchnorm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer2/branch2/conv2a_bn/batchnorm/sub"
  op: "Sub"
  input: "res_layer2/branch2/conv2a_bn/conv2a_bn_offset/read"
  input: "res_layer2/branch2/conv2a_bn/batchnorm/mul_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer2/branch2/conv2a_bn/batchnorm/add_1"
  op: "Add"
  input: "res_layer2/branch2/conv2a_bn/batchnorm/mul_1"
  input: "res_layer2/branch2/conv2a_bn/batchnorm/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer2/branch2/conv2a_relu/relu"
  op: "Relu"
  input: "res_layer2/branch2/conv2a_bn/batchnorm/add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer2/branch2/conv2b/weights"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer2/branch2/conv2b/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 3
        }
        dim {
          size: 256
        }
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer2/branch2/conv2b/weights/read"
  op: "Identity"
  input: "res_layer2/branch2/conv2b/weights"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer2/branch2/conv2b/weights"
      }
    }
  }
}
node {
  name: "res_layer2/branch2/conv2b/conv2b"
  op: "Conv2D"
  input: "res_layer2/branch2/conv2a_relu/relu"
  input: "res_layer2/branch2/conv2b/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer2/branch2/conv2b_bn/conv2b_bn_moments/mean/reduction_indices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 3
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "res_layer2/branch2/conv2b_bn/conv2b_bn_moments/mean"
  op: "Mean"
  input: "res_layer2/branch2/conv2b/conv2b"
  input: "res_layer2/branch2/conv2b_bn/conv2b_bn_moments/mean/reduction_indices"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer2/branch2/conv2b_bn/conv2b_bn_moments/StopGradient"
  op: "StopGradient"
  input: "res_layer2/branch2/conv2b_bn/conv2b_bn_moments/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer2/branch2/conv2b_bn/conv2b_bn_moments/SquaredDifference"
  op: "SquaredDifference"
  input: "res_layer2/branch2/conv2b/conv2b"
  input: "res_layer2/branch2/conv2b_bn/conv2b_bn_moments/StopGradient"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer2/branch2/conv2b_bn/conv2b_bn_moments/variance/reduction_indices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 3
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "res_layer2/branch2/conv2b_bn/conv2b_bn_moments/variance"
  op: "Mean"
  input: "res_layer2/branch2/conv2b_bn/conv2b_bn_moments/SquaredDifference"
  input: "res_layer2/branch2/conv2b_bn/conv2b_bn_moments/variance/reduction_indices"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer2/branch2/conv2b_bn/conv2b_bn_moments/Squeeze"
  op: "Squeeze"
  input: "res_layer2/branch2/conv2b_bn/conv2b_bn_moments/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "squeeze_dims"
    value {
      list {
        i: 0
        i: 1
        i: 2
      }
    }
  }
}
node {
  name: "res_layer2/branch2/conv2b_bn/conv2b_bn_moments/Squeeze_1"
  op: "Squeeze"
  input: "res_layer2/branch2/conv2b_bn/conv2b_bn_moments/variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "squeeze_dims"
    value {
      list {
        i: 0
        i: 1
        i: 2
      }
    }
  }
}
node {
  name: "res_layer2/branch2/conv2b_bn/conv2b_bn_scale"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer2/branch2/conv2b_bn/conv2b_bn_scale"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer2/branch2/conv2b_bn/conv2b_bn_scale/read"
  op: "Identity"
  input: "res_layer2/branch2/conv2b_bn/conv2b_bn_scale"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer2/branch2/conv2b_bn/conv2b_bn_scale"
      }
    }
  }
}
node {
  name: "res_layer2/branch2/conv2b_bn/conv2b_bn_offset"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer2/branch2/conv2b_bn/conv2b_bn_offset"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer2/branch2/conv2b_bn/conv2b_bn_offset/read"
  op: "Identity"
  input: "res_layer2/branch2/conv2b_bn/conv2b_bn_offset"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer2/branch2/conv2b_bn/conv2b_bn_offset"
      }
    }
  }
}
node {
  name: "res_layer2/branch2/conv2b_bn/batchnorm/add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 9.999999747378752e-06
      }
    }
  }
}
node {
  name: "res_layer2/branch2/conv2b_bn/batchnorm/add"
  op: "Add"
  input: "res_layer2/branch2/conv2b_bn/conv2b_bn_moments/Squeeze_1"
  input: "res_layer2/branch2/conv2b_bn/batchnorm/add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer2/branch2/conv2b_bn/batchnorm/Rsqrt"
  op: "Rsqrt"
  input: "res_layer2/branch2/conv2b_bn/batchnorm/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer2/branch2/conv2b_bn/batchnorm/mul"
  op: "Mul"
  input: "res_layer2/branch2/conv2b_bn/batchnorm/Rsqrt"
  input: "res_layer2/branch2/conv2b_bn/conv2b_bn_scale/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer2/branch2/conv2b_bn/batchnorm/mul_1"
  op: "Mul"
  input: "res_layer2/branch2/conv2b/conv2b"
  input: "res_layer2/branch2/conv2b_bn/batchnorm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer2/branch2/conv2b_bn/batchnorm/mul_2"
  op: "Mul"
  input: "res_layer2/branch2/conv2b_bn/conv2b_bn_moments/Squeeze"
  input: "res_layer2/branch2/conv2b_bn/batchnorm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer2/branch2/conv2b_bn/batchnorm/sub"
  op: "Sub"
  input: "res_layer2/branch2/conv2b_bn/conv2b_bn_offset/read"
  input: "res_layer2/branch2/conv2b_bn/batchnorm/mul_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer2/branch2/conv2b_bn/batchnorm/add_1"
  op: "Add"
  input: "res_layer2/branch2/conv2b_bn/batchnorm/mul_1"
  input: "res_layer2/branch2/conv2b_bn/batchnorm/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer2/branch2/conv2b_relu/relu"
  op: "Relu"
  input: "res_layer2/branch2/conv2b_bn/batchnorm/add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer2/branch2/conv2c/weights"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer2/branch2/conv2c/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 256
        }
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer2/branch2/conv2c/weights/read"
  op: "Identity"
  input: "res_layer2/branch2/conv2c/weights"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer2/branch2/conv2c/weights"
      }
    }
  }
}
node {
  name: "res_layer2/branch2/conv2c/conv2c"
  op: "Conv2D"
  input: "res_layer2/branch2/conv2b_relu/relu"
  input: "res_layer2/branch2/conv2c/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer2/branch2/conv2c_bn/conv2c_bn_moments/mean/reduction_indices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 3
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "res_layer2/branch2/conv2c_bn/conv2c_bn_moments/mean"
  op: "Mean"
  input: "res_layer2/branch2/conv2c/conv2c"
  input: "res_layer2/branch2/conv2c_bn/conv2c_bn_moments/mean/reduction_indices"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer2/branch2/conv2c_bn/conv2c_bn_moments/StopGradient"
  op: "StopGradient"
  input: "res_layer2/branch2/conv2c_bn/conv2c_bn_moments/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer2/branch2/conv2c_bn/conv2c_bn_moments/SquaredDifference"
  op: "SquaredDifference"
  input: "res_layer2/branch2/conv2c/conv2c"
  input: "res_layer2/branch2/conv2c_bn/conv2c_bn_moments/StopGradient"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer2/branch2/conv2c_bn/conv2c_bn_moments/variance/reduction_indices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 3
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "res_layer2/branch2/conv2c_bn/conv2c_bn_moments/variance"
  op: "Mean"
  input: "res_layer2/branch2/conv2c_bn/conv2c_bn_moments/SquaredDifference"
  input: "res_layer2/branch2/conv2c_bn/conv2c_bn_moments/variance/reduction_indices"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer2/branch2/conv2c_bn/conv2c_bn_moments/Squeeze"
  op: "Squeeze"
  input: "res_layer2/branch2/conv2c_bn/conv2c_bn_moments/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "squeeze_dims"
    value {
      list {
        i: 0
        i: 1
        i: 2
      }
    }
  }
}
node {
  name: "res_layer2/branch2/conv2c_bn/conv2c_bn_moments/Squeeze_1"
  op: "Squeeze"
  input: "res_layer2/branch2/conv2c_bn/conv2c_bn_moments/variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "squeeze_dims"
    value {
      list {
        i: 0
        i: 1
        i: 2
      }
    }
  }
}
node {
  name: "res_layer2/branch2/conv2c_bn/conv2c_bn_scale"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer2/branch2/conv2c_bn/conv2c_bn_scale"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer2/branch2/conv2c_bn/conv2c_bn_scale/read"
  op: "Identity"
  input: "res_layer2/branch2/conv2c_bn/conv2c_bn_scale"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer2/branch2/conv2c_bn/conv2c_bn_scale"
      }
    }
  }
}
node {
  name: "res_layer2/branch2/conv2c_bn/conv2c_bn_offset"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer2/branch2/conv2c_bn/conv2c_bn_offset"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer2/branch2/conv2c_bn/conv2c_bn_offset/read"
  op: "Identity"
  input: "res_layer2/branch2/conv2c_bn/conv2c_bn_offset"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer2/branch2/conv2c_bn/conv2c_bn_offset"
      }
    }
  }
}
node {
  name: "res_layer2/branch2/conv2c_bn/batchnorm/add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 9.999999747378752e-06
      }
    }
  }
}
node {
  name: "res_layer2/branch2/conv2c_bn/batchnorm/add"
  op: "Add"
  input: "res_layer2/branch2/conv2c_bn/conv2c_bn_moments/Squeeze_1"
  input: "res_layer2/branch2/conv2c_bn/batchnorm/add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer2/branch2/conv2c_bn/batchnorm/Rsqrt"
  op: "Rsqrt"
  input: "res_layer2/branch2/conv2c_bn/batchnorm/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer2/branch2/conv2c_bn/batchnorm/mul"
  op: "Mul"
  input: "res_layer2/branch2/conv2c_bn/batchnorm/Rsqrt"
  input: "res_layer2/branch2/conv2c_bn/conv2c_bn_scale/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer2/branch2/conv2c_bn/batchnorm/mul_1"
  op: "Mul"
  input: "res_layer2/branch2/conv2c/conv2c"
  input: "res_layer2/branch2/conv2c_bn/batchnorm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer2/branch2/conv2c_bn/batchnorm/mul_2"
  op: "Mul"
  input: "res_layer2/branch2/conv2c_bn/conv2c_bn_moments/Squeeze"
  input: "res_layer2/branch2/conv2c_bn/batchnorm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer2/branch2/conv2c_bn/batchnorm/sub"
  op: "Sub"
  input: "res_layer2/branch2/conv2c_bn/conv2c_bn_offset/read"
  input: "res_layer2/branch2/conv2c_bn/batchnorm/mul_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer2/branch2/conv2c_bn/batchnorm/add_1"
  op: "Add"
  input: "res_layer2/branch2/conv2c_bn/batchnorm/mul_1"
  input: "res_layer2/branch2/conv2c_bn/batchnorm/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer2/plus/add"
  op: "Add"
  input: "res_layer2/branch1/conv1/conv1"
  input: "res_layer2/branch2/conv2c_bn/batchnorm/add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer2/plus/final_relu"
  op: "Relu"
  input: "res_layer2/plus/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer3/branch1/conv1/weights"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer3/branch1/conv1/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 256
        }
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer3/branch1/conv1/weights/read"
  op: "Identity"
  input: "res_layer3/branch1/conv1/weights"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer3/branch1/conv1/weights"
      }
    }
  }
}
node {
  name: "res_layer3/branch1/conv1/conv1"
  op: "Conv2D"
  input: "res_layer2/plus/final_relu"
  input: "res_layer3/branch1/conv1/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer3/branch2/conv2a/weights"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer3/branch2/conv2a/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 256
        }
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer3/branch2/conv2a/weights/read"
  op: "Identity"
  input: "res_layer3/branch2/conv2a/weights"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer3/branch2/conv2a/weights"
      }
    }
  }
}
node {
  name: "res_layer3/branch2/conv2a/conv2a"
  op: "Conv2D"
  input: "res_layer2/plus/final_relu"
  input: "res_layer3/branch2/conv2a/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer3/branch2/conv2a_bn/conv2a_bn_moments/mean/reduction_indices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 3
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "res_layer3/branch2/conv2a_bn/conv2a_bn_moments/mean"
  op: "Mean"
  input: "res_layer3/branch2/conv2a/conv2a"
  input: "res_layer3/branch2/conv2a_bn/conv2a_bn_moments/mean/reduction_indices"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer3/branch2/conv2a_bn/conv2a_bn_moments/StopGradient"
  op: "StopGradient"
  input: "res_layer3/branch2/conv2a_bn/conv2a_bn_moments/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer3/branch2/conv2a_bn/conv2a_bn_moments/SquaredDifference"
  op: "SquaredDifference"
  input: "res_layer3/branch2/conv2a/conv2a"
  input: "res_layer3/branch2/conv2a_bn/conv2a_bn_moments/StopGradient"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer3/branch2/conv2a_bn/conv2a_bn_moments/variance/reduction_indices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 3
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "res_layer3/branch2/conv2a_bn/conv2a_bn_moments/variance"
  op: "Mean"
  input: "res_layer3/branch2/conv2a_bn/conv2a_bn_moments/SquaredDifference"
  input: "res_layer3/branch2/conv2a_bn/conv2a_bn_moments/variance/reduction_indices"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer3/branch2/conv2a_bn/conv2a_bn_moments/Squeeze"
  op: "Squeeze"
  input: "res_layer3/branch2/conv2a_bn/conv2a_bn_moments/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "squeeze_dims"
    value {
      list {
        i: 0
        i: 1
        i: 2
      }
    }
  }
}
node {
  name: "res_layer3/branch2/conv2a_bn/conv2a_bn_moments/Squeeze_1"
  op: "Squeeze"
  input: "res_layer3/branch2/conv2a_bn/conv2a_bn_moments/variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "squeeze_dims"
    value {
      list {
        i: 0
        i: 1
        i: 2
      }
    }
  }
}
node {
  name: "res_layer3/branch2/conv2a_bn/conv2a_bn_scale"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer3/branch2/conv2a_bn/conv2a_bn_scale"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer3/branch2/conv2a_bn/conv2a_bn_scale/read"
  op: "Identity"
  input: "res_layer3/branch2/conv2a_bn/conv2a_bn_scale"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer3/branch2/conv2a_bn/conv2a_bn_scale"
      }
    }
  }
}
node {
  name: "res_layer3/branch2/conv2a_bn/conv2a_bn_offset"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer3/branch2/conv2a_bn/conv2a_bn_offset"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer3/branch2/conv2a_bn/conv2a_bn_offset/read"
  op: "Identity"
  input: "res_layer3/branch2/conv2a_bn/conv2a_bn_offset"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer3/branch2/conv2a_bn/conv2a_bn_offset"
      }
    }
  }
}
node {
  name: "res_layer3/branch2/conv2a_bn/batchnorm/add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 9.999999747378752e-06
      }
    }
  }
}
node {
  name: "res_layer3/branch2/conv2a_bn/batchnorm/add"
  op: "Add"
  input: "res_layer3/branch2/conv2a_bn/conv2a_bn_moments/Squeeze_1"
  input: "res_layer3/branch2/conv2a_bn/batchnorm/add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer3/branch2/conv2a_bn/batchnorm/Rsqrt"
  op: "Rsqrt"
  input: "res_layer3/branch2/conv2a_bn/batchnorm/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer3/branch2/conv2a_bn/batchnorm/mul"
  op: "Mul"
  input: "res_layer3/branch2/conv2a_bn/batchnorm/Rsqrt"
  input: "res_layer3/branch2/conv2a_bn/conv2a_bn_scale/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer3/branch2/conv2a_bn/batchnorm/mul_1"
  op: "Mul"
  input: "res_layer3/branch2/conv2a/conv2a"
  input: "res_layer3/branch2/conv2a_bn/batchnorm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer3/branch2/conv2a_bn/batchnorm/mul_2"
  op: "Mul"
  input: "res_layer3/branch2/conv2a_bn/conv2a_bn_moments/Squeeze"
  input: "res_layer3/branch2/conv2a_bn/batchnorm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer3/branch2/conv2a_bn/batchnorm/sub"
  op: "Sub"
  input: "res_layer3/branch2/conv2a_bn/conv2a_bn_offset/read"
  input: "res_layer3/branch2/conv2a_bn/batchnorm/mul_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer3/branch2/conv2a_bn/batchnorm/add_1"
  op: "Add"
  input: "res_layer3/branch2/conv2a_bn/batchnorm/mul_1"
  input: "res_layer3/branch2/conv2a_bn/batchnorm/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer3/branch2/conv2a_relu/relu"
  op: "Relu"
  input: "res_layer3/branch2/conv2a_bn/batchnorm/add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer3/branch2/conv2b/weights"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer3/branch2/conv2b/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 3
        }
        dim {
          size: 256
        }
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer3/branch2/conv2b/weights/read"
  op: "Identity"
  input: "res_layer3/branch2/conv2b/weights"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer3/branch2/conv2b/weights"
      }
    }
  }
}
node {
  name: "res_layer3/branch2/conv2b/conv2b"
  op: "Conv2D"
  input: "res_layer3/branch2/conv2a_relu/relu"
  input: "res_layer3/branch2/conv2b/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer3/branch2/conv2b_bn/conv2b_bn_moments/mean/reduction_indices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 3
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "res_layer3/branch2/conv2b_bn/conv2b_bn_moments/mean"
  op: "Mean"
  input: "res_layer3/branch2/conv2b/conv2b"
  input: "res_layer3/branch2/conv2b_bn/conv2b_bn_moments/mean/reduction_indices"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer3/branch2/conv2b_bn/conv2b_bn_moments/StopGradient"
  op: "StopGradient"
  input: "res_layer3/branch2/conv2b_bn/conv2b_bn_moments/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer3/branch2/conv2b_bn/conv2b_bn_moments/SquaredDifference"
  op: "SquaredDifference"
  input: "res_layer3/branch2/conv2b/conv2b"
  input: "res_layer3/branch2/conv2b_bn/conv2b_bn_moments/StopGradient"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer3/branch2/conv2b_bn/conv2b_bn_moments/variance/reduction_indices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 3
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "res_layer3/branch2/conv2b_bn/conv2b_bn_moments/variance"
  op: "Mean"
  input: "res_layer3/branch2/conv2b_bn/conv2b_bn_moments/SquaredDifference"
  input: "res_layer3/branch2/conv2b_bn/conv2b_bn_moments/variance/reduction_indices"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer3/branch2/conv2b_bn/conv2b_bn_moments/Squeeze"
  op: "Squeeze"
  input: "res_layer3/branch2/conv2b_bn/conv2b_bn_moments/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "squeeze_dims"
    value {
      list {
        i: 0
        i: 1
        i: 2
      }
    }
  }
}
node {
  name: "res_layer3/branch2/conv2b_bn/conv2b_bn_moments/Squeeze_1"
  op: "Squeeze"
  input: "res_layer3/branch2/conv2b_bn/conv2b_bn_moments/variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "squeeze_dims"
    value {
      list {
        i: 0
        i: 1
        i: 2
      }
    }
  }
}
node {
  name: "res_layer3/branch2/conv2b_bn/conv2b_bn_scale"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer3/branch2/conv2b_bn/conv2b_bn_scale"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer3/branch2/conv2b_bn/conv2b_bn_scale/read"
  op: "Identity"
  input: "res_layer3/branch2/conv2b_bn/conv2b_bn_scale"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer3/branch2/conv2b_bn/conv2b_bn_scale"
      }
    }
  }
}
node {
  name: "res_layer3/branch2/conv2b_bn/conv2b_bn_offset"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer3/branch2/conv2b_bn/conv2b_bn_offset"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer3/branch2/conv2b_bn/conv2b_bn_offset/read"
  op: "Identity"
  input: "res_layer3/branch2/conv2b_bn/conv2b_bn_offset"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer3/branch2/conv2b_bn/conv2b_bn_offset"
      }
    }
  }
}
node {
  name: "res_layer3/branch2/conv2b_bn/batchnorm/add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 9.999999747378752e-06
      }
    }
  }
}
node {
  name: "res_layer3/branch2/conv2b_bn/batchnorm/add"
  op: "Add"
  input: "res_layer3/branch2/conv2b_bn/conv2b_bn_moments/Squeeze_1"
  input: "res_layer3/branch2/conv2b_bn/batchnorm/add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer3/branch2/conv2b_bn/batchnorm/Rsqrt"
  op: "Rsqrt"
  input: "res_layer3/branch2/conv2b_bn/batchnorm/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer3/branch2/conv2b_bn/batchnorm/mul"
  op: "Mul"
  input: "res_layer3/branch2/conv2b_bn/batchnorm/Rsqrt"
  input: "res_layer3/branch2/conv2b_bn/conv2b_bn_scale/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer3/branch2/conv2b_bn/batchnorm/mul_1"
  op: "Mul"
  input: "res_layer3/branch2/conv2b/conv2b"
  input: "res_layer3/branch2/conv2b_bn/batchnorm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer3/branch2/conv2b_bn/batchnorm/mul_2"
  op: "Mul"
  input: "res_layer3/branch2/conv2b_bn/conv2b_bn_moments/Squeeze"
  input: "res_layer3/branch2/conv2b_bn/batchnorm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer3/branch2/conv2b_bn/batchnorm/sub"
  op: "Sub"
  input: "res_layer3/branch2/conv2b_bn/conv2b_bn_offset/read"
  input: "res_layer3/branch2/conv2b_bn/batchnorm/mul_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer3/branch2/conv2b_bn/batchnorm/add_1"
  op: "Add"
  input: "res_layer3/branch2/conv2b_bn/batchnorm/mul_1"
  input: "res_layer3/branch2/conv2b_bn/batchnorm/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer3/branch2/conv2b_relu/relu"
  op: "Relu"
  input: "res_layer3/branch2/conv2b_bn/batchnorm/add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer3/branch2/conv2c/weights"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer3/branch2/conv2c/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 256
        }
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer3/branch2/conv2c/weights/read"
  op: "Identity"
  input: "res_layer3/branch2/conv2c/weights"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer3/branch2/conv2c/weights"
      }
    }
  }
}
node {
  name: "res_layer3/branch2/conv2c/conv2c"
  op: "Conv2D"
  input: "res_layer3/branch2/conv2b_relu/relu"
  input: "res_layer3/branch2/conv2c/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer3/branch2/conv2c_bn/conv2c_bn_moments/mean/reduction_indices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 3
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "res_layer3/branch2/conv2c_bn/conv2c_bn_moments/mean"
  op: "Mean"
  input: "res_layer3/branch2/conv2c/conv2c"
  input: "res_layer3/branch2/conv2c_bn/conv2c_bn_moments/mean/reduction_indices"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer3/branch2/conv2c_bn/conv2c_bn_moments/StopGradient"
  op: "StopGradient"
  input: "res_layer3/branch2/conv2c_bn/conv2c_bn_moments/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer3/branch2/conv2c_bn/conv2c_bn_moments/SquaredDifference"
  op: "SquaredDifference"
  input: "res_layer3/branch2/conv2c/conv2c"
  input: "res_layer3/branch2/conv2c_bn/conv2c_bn_moments/StopGradient"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer3/branch2/conv2c_bn/conv2c_bn_moments/variance/reduction_indices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 3
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "res_layer3/branch2/conv2c_bn/conv2c_bn_moments/variance"
  op: "Mean"
  input: "res_layer3/branch2/conv2c_bn/conv2c_bn_moments/SquaredDifference"
  input: "res_layer3/branch2/conv2c_bn/conv2c_bn_moments/variance/reduction_indices"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: true
    }
  }
}
node {
  name: "res_layer3/branch2/conv2c_bn/conv2c_bn_moments/Squeeze"
  op: "Squeeze"
  input: "res_layer3/branch2/conv2c_bn/conv2c_bn_moments/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "squeeze_dims"
    value {
      list {
        i: 0
        i: 1
        i: 2
      }
    }
  }
}
node {
  name: "res_layer3/branch2/conv2c_bn/conv2c_bn_moments/Squeeze_1"
  op: "Squeeze"
  input: "res_layer3/branch2/conv2c_bn/conv2c_bn_moments/variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "squeeze_dims"
    value {
      list {
        i: 0
        i: 1
        i: 2
      }
    }
  }
}
node {
  name: "res_layer3/branch2/conv2c_bn/conv2c_bn_scale"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer3/branch2/conv2c_bn/conv2c_bn_scale"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer3/branch2/conv2c_bn/conv2c_bn_scale/read"
  op: "Identity"
  input: "res_layer3/branch2/conv2c_bn/conv2c_bn_scale"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer3/branch2/conv2c_bn/conv2c_bn_scale"
      }
    }
  }
}
node {
  name: "res_layer3/branch2/conv2c_bn/conv2c_bn_offset"
  op: "VariableV2"
  device: "/device:CPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer3/branch2/conv2c_bn/conv2c_bn_offset"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "res_layer3/branch2/conv2c_bn/conv2c_bn_offset/read"
  op: "Identity"
  input: "res_layer3/branch2/conv2c_bn/conv2c_bn_offset"
  device: "/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@res_layer3/branch2/conv2c_bn/conv2c_bn_offset"
      }
    }
  }
}
node {
  name: "res_layer3/branch2/conv2c_bn/batchnorm/add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 9.999999747378752e-06
      }
    }
  }
}
node {
  name: "res_layer3/branch2/conv2c_bn/batchnorm/add"
  op: "Add"
  input: "res_layer3/branch2/conv2c_bn/conv2c_bn_moments/Squeeze_1"
  input: "res_layer3/branch2/conv2c_bn/batchnorm/add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer3/branch2/conv2c_bn/batchnorm/Rsqrt"
  op: "Rsqrt"
  input: "res_layer3/branch2/conv2c_bn/batchnorm/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer3/branch2/conv2c_bn/batchnorm/mul"
  op: "Mul"
  input: "res_layer3/branch2/conv2c_bn/batchnorm/Rsqrt"
  input: "res_layer3/branch2/conv2c_bn/conv2c_bn_scale/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer3/branch2/conv2c_bn/batchnorm/mul_1"
  op: "Mul"
  input: "res_layer3/branch2/conv2c/conv2c"
  input: "res_layer3/branch2/conv2c_bn/batchnorm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer3/branch2/conv2c_bn/batchnorm/mul_2"
  op: "Mul"
  input: "res_layer3/branch2/conv2c_bn/conv2c_bn_moments/Squeeze"
  input: "res_layer3/branch2/conv2c_bn/batchnorm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer3/branch2/conv2c_bn/batchnorm/sub"
  op: "Sub"
  input: "res_layer3/branch2/conv2c_bn/conv2c_bn_offset/read"
  input: "res_layer3/branch2/conv2c_bn/batchnorm/mul_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer3/branch2/conv2c_bn/batchnorm/add_1"
  op: "Add"
  input: "res_layer3/branch2/conv2c_bn/batchnorm/mul_1"
  input: "res_layer3/branch2/conv2c_bn/batchnorm/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer3/plus/add"
  op: "Add"
  input: "res_layer3/branch1/conv1/conv1"
  input: "res_layer3/branch2/conv2c_bn/batchnorm/add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "res_layer3/plus/final_relu"
  op: "Relu"
  input: "res_layer3/plus/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "fea_rs/shape/1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 400
      }
    }
  }
}
node {
  name: "fea_rs/shape/2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 256
      }
    }
  }
}
node {
  name: "fea_rs/shape"
  op: "Pack"
  input: "strided_slice"
  input: "fea_rs/shape/1"
  input: "fea_rs/shape/2"
  attr {
    key: "N"
    value {
      i: 3
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "fea_rs"
  op: "Reshape"
  input: "res_layer3/plus/final_relu"
  input: "fea_rs/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
library {
}
versions {
  producer: 27
}
