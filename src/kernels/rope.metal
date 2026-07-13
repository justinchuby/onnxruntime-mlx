// Copyright (c) 2026. Licensed under the MIT License.

#include <metal_stdlib>
using namespace metal;

struct RopeParams {
  uint batch_size;
  uint sequence_length;
  uint num_heads;
  uint head_size;
  uint rotary_embedding_dim;
  uint cache_stride;
  uint max_sequence_length;
  uint rank3_bsh;
  uint interleaved;
  uint n;
};

template <typename T>
inline void rope_one(device const T* input, device const T* cos_cache,
                     device const T* sin_cache, device T* output, uint gid,
                     uint position, constant RopeParams& p) {
  uint d = gid % p.head_size;
  if (d >= p.rotary_embedding_dim) {
    output[gid] = input[gid];
    return;
  }

  uint pair_index;
  uint partner_d;
  float sign;
  if (p.interleaved != 0) {
    pair_index = d >> 1;
    partner_d = d ^ 1;
    sign = (d & 1) == 0 ? -1.0f : 1.0f;
  } else {
    uint half_dim = p.rotary_embedding_dim >> 1;
    pair_index = d % half_dim;
    partner_d = d < half_dim ? d + half_dim : d - half_dim;
    sign = d < half_dim ? -1.0f : 1.0f;
  }

  uint partner = gid - d + partner_d;
  uint cache_index = position * p.cache_stride + pair_index;
  float x = float(input[gid]);
  float other = float(input[partner]);
  float c = float(cos_cache[cache_index]);
  float s = float(sin_cache[cache_index]);
  output[gid] = T(x * c + sign * other * s);
}

inline uint rope_sequence_index(uint gid, constant RopeParams& p) {
  uint vector_index = gid / p.head_size;
  if (p.rank3_bsh != 0) {
    return (vector_index / p.num_heads) % p.sequence_length;
  }
  return vector_index % p.sequence_length;
}

inline uint rope_batch_index(uint gid, constant RopeParams& p) {
  uint vector_index = gid / p.head_size;
  if (p.rank3_bsh != 0) {
    return vector_index / (p.sequence_length * p.num_heads);
  }
  return vector_index / (p.num_heads * p.sequence_length);
}

#define MPS_ROPE_KERNELS(SUFFIX, TYPE)                                                \
  kernel void mps_rope_##SUFFIX(device const TYPE* input [[buffer(0)]],               \
                                device const TYPE* cos_cache [[buffer(1)]],           \
                                device const TYPE* sin_cache [[buffer(2)]],           \
                                device TYPE* output [[buffer(3)]],                    \
                                constant RopeParams& p [[buffer(4)]],                 \
                                uint gid [[thread_position_in_grid]]) {               \
    if (gid >= p.n) return;                                                           \
    uint b = rope_batch_index(gid, p);                                                \
    uint s = rope_sequence_index(gid, p);                                             \
    rope_one(input, cos_cache, sin_cache, output, gid, b * p.sequence_length + s, p); \
  }                                                                                  \
  kernel void mps_rope_pos_##SUFFIX(device const TYPE* input [[buffer(0)]],           \
                                    device const TYPE* cos_cache [[buffer(1)]],       \
                                    device const TYPE* sin_cache [[buffer(2)]],       \
                                    device const long* position_ids [[buffer(3)]],    \
                                    device TYPE* output [[buffer(4)]],                \
                                    constant RopeParams& p [[buffer(5)]],             \
                                    uint gid [[thread_position_in_grid]]) {           \
    if (gid >= p.n) return;                                                           \
    uint b = rope_batch_index(gid, p);                                                \
    uint s = rope_sequence_index(gid, p);                                             \
    uint position = uint(position_ids[b * p.sequence_length + s]);                    \
    rope_one(input, cos_cache, sin_cache, output, gid, position, p);                  \
  }

MPS_ROPE_KERNELS(f32, float)
MPS_ROPE_KERNELS(f16, half)
