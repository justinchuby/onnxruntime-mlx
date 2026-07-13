// Copyright (c) 2026. Licensed under the MIT License.

#include <metal_stdlib>
using namespace metal;

struct GatherBlockQuantizedParams {
  uint rows;
  uint row_width;
  uint packed_row_width;
  uint blocks_per_row;
  uint block_size;
  uint indices_count;
  uint n;
};

template <typename IndexT>
inline int gather_index(device const IndexT* indices, uint row,
                        constant GatherBlockQuantizedParams& p) {
  int index = int(indices[row]);
  return index < 0 ? index + int(p.rows) : index;
}

template <typename ScaleT, typename OutputT>
inline void gather_q4_one(device const uchar* data, int row, uint column,
                          device const ScaleT* scales, device const uchar* zero_points,
                          device OutputT* output, uint gid,
                          constant GatherBlockQuantizedParams& p) {
  uint packed_index = uint(row) * p.packed_row_width + (column >> 1);
  uchar packed = data[packed_index];
  int q = (column & 1) == 0 ? int(packed & 0x0f) : int((packed >> 4) & 0x0f);
  uint block = column / p.block_size;
  int zp = 8;
  if (zero_points != nullptr) {
    uint zp_index = uint(row) * ((p.blocks_per_row + 1) >> 1) + (block >> 1);
    uchar packed_zp = zero_points[zp_index];
    zp = (block & 1) == 0 ? int(packed_zp & 0x0f) : int((packed_zp >> 4) & 0x0f);
  }
  float scale = float(scales[uint(row) * p.blocks_per_row + block]);
  output[gid] = OutputT(float(q - zp) * scale);
}

#define MPS_GATHER_Q4_NO_ZP(NAME, INDEX_T, SCALE_T, OUTPUT_T)                         \
  kernel void NAME(device const uchar* data [[buffer(0)]],                            \
                   device const INDEX_T* indices [[buffer(1)]],                       \
                   device const SCALE_T* scales [[buffer(2)]],                        \
                   device OUTPUT_T* output [[buffer(3)]],                             \
                   constant GatherBlockQuantizedParams& p [[buffer(4)]],              \
                   uint gid [[thread_position_in_grid]]) {                            \
    if (gid >= p.n) return;                                                           \
    uint output_row = gid / p.row_width;                                              \
    uint column = gid % p.row_width;                                                  \
    int row = gather_index(indices, output_row, p);                                   \
    gather_q4_one(data, row, column, scales, nullptr, output, gid, p);                \
  }

#define MPS_GATHER_Q4_ZP(NAME, INDEX_T, SCALE_T, OUTPUT_T)                            \
  kernel void NAME(device const uchar* data [[buffer(0)]],                            \
                   device const INDEX_T* indices [[buffer(1)]],                       \
                   device const SCALE_T* scales [[buffer(2)]],                        \
                   device const uchar* zero_points [[buffer(3)]],                     \
                   device OUTPUT_T* output [[buffer(4)]],                             \
                   constant GatherBlockQuantizedParams& p [[buffer(5)]],              \
                   uint gid [[thread_position_in_grid]]) {                            \
    if (gid >= p.n) return;                                                           \
    uint output_row = gid / p.row_width;                                              \
    uint column = gid % p.row_width;                                                  \
    int row = gather_index(indices, output_row, p);                                   \
    gather_q4_one(data, row, column, scales, zero_points, output, gid, p);            \
  }

MPS_GATHER_Q4_NO_ZP(mps_gather_q4_i64_f32, long, float, float)
MPS_GATHER_Q4_NO_ZP(mps_gather_q4_i32_f32, int, float, float)
MPS_GATHER_Q4_NO_ZP(mps_gather_q4_i64_f16, long, half, half)
MPS_GATHER_Q4_NO_ZP(mps_gather_q4_i32_f16, int, half, half)
MPS_GATHER_Q4_ZP(mps_gather_q4_i64_f32_zp, long, float, float)
MPS_GATHER_Q4_ZP(mps_gather_q4_i32_f32_zp, int, float, float)
MPS_GATHER_Q4_ZP(mps_gather_q4_i64_f16_zp, long, half, half)
MPS_GATHER_Q4_ZP(mps_gather_q4_i32_f16_zp, int, half, half)
