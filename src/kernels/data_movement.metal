// Copyright (c) 2026. Licensed under the MIT License.

#include <metal_stdlib>
using namespace metal;

kernel void mps_copy_bytes(device const uchar* input [[buffer(0)]],
                           device uchar* output [[buffer(1)]],
                           constant uint& n [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
  if (gid < n) output[gid] = input[gid];
}

struct TransposeParams {
  uint rank;
  uint element_size;
  uint n;
  uint output_dims[8];
  uint input_strides[8];
  uint permutation[8];
};

kernel void mps_transpose_bytes(device const uchar* input [[buffer(0)]],
                                device uchar* output [[buffer(1)]],
                                constant TransposeParams& p [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n) return;
  uint remaining = gid;
  uint input_index = 0;
  for (int output_axis = int(p.rank) - 1; output_axis >= 0; --output_axis) {
    uint coordinate = remaining % p.output_dims[output_axis];
    remaining /= p.output_dims[output_axis];
    input_index += coordinate * p.input_strides[p.permutation[output_axis]];
  }
  uint input_byte = input_index * p.element_size;
  uint output_byte = gid * p.element_size;
  for (uint byte = 0; byte < p.element_size; ++byte) {
    output[output_byte + byte] = input[input_byte + byte];
  }
}

struct ConcatSliceParams {
  uint element_size;
  uint outer;
  uint input_axis;
  uint output_axis;
  uint inner;
  uint axis_offset;
  uint n;
};

kernel void mps_concat_slice_bytes(device const uchar* input [[buffer(0)]],
                                   device uchar* output [[buffer(1)]],
                                   constant ConcatSliceParams& p [[buffer(2)]],
                                   uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n) return;
  uint axis_inner = p.input_axis * p.inner;
  uint outer_index = gid / axis_inner;
  uint remainder = gid % axis_inner;
  uint axis_index = remainder / p.inner;
  uint inner_index = remainder % p.inner;
  uint output_index =
      (outer_index * p.output_axis + p.axis_offset + axis_index) * p.inner + inner_index;
  uint input_byte = gid * p.element_size;
  uint output_byte = output_index * p.element_size;
  for (uint byte = 0; byte < p.element_size; ++byte) {
    output[output_byte + byte] = input[input_byte + byte];
  }
}
