// Copyright (c) 2026. Licensed under the MIT License.
//
// Recurrent op handlers (ai.onnx opset-17+): RNN, GRU, LSTM. See docs/OP_ARCHITECTURE.md §5.
//
// KEY INSIGHT — static-length unrolling. GRU/LSTM/RNN are SINGLE nodes (no nested subgraph) carrying
// the weight inputs W/R/(B)/(P) and a `hidden_size` attribute; the recurrence runs over the sequence
// axis of X ([seq_length, batch, input_size]). When seq_length is STATICALLY KNOWN from the input
// shape we UNROLL the recurrence into a fixed MLX graph: a host-side loop over t = 0..S-1 builds S
// steps of graph, each computing the gate pre-activations via matmuls (Xt·Wᵀ + H_{t-1}·Rᵀ + bias)
// followed by the gate activations. No control-flow primitive is needed — just S bounded steps.
//
//   * RNN  — Ht = f(Xt·Wᵀ + H_{t-1}·Rᵀ + Wb + Rb). f defaults to Tanh.
//   * GRU  — z/r/h gates; both `linear_before_reset` forms; f=Sigmoid, g=Tanh (defaults).
//   * LSTM — i/o/f/c gates + cell state; optional peepholes P; `input_forget`; f=Sigmoid, g=h=Tanh.
//
// Directions: forward, reverse (iterates t backwards), and bidirectional (the two directions run
// independently and their outputs are concatenated along the num_directions axis). `hidden_size`,
// `clip`, optional B / initial_h / (LSTM) initial_c / P are all supported.
//
// CLAIMED FORMS ONLY (everything else falls back to ORT CPU, which is always correct):
//   * float dtypes (fp32/fp16/bf16), all operands and outputs the same dtype;
//   * X rank-3 with a STATIC positive seq_length (dim 0) — a dynamic/symbolic seq length → CPU;
//   * default layout (layout=0); no `sequence_lens` input (variable-length masking → CPU);
//   * DEFAULT activations only (RNN=Tanh, GRU=Sigmoid/Tanh, LSTM=Sigmoid/Tanh/Tanh). Because the
//     STRINGS `activations` attribute is not carried into NodeDesc, the claim predicate validates it
//     against the per-op defaults and the handler hard-codes those activations; any non-default
//     activation set (e.g. Relu RNN, LeakyRelu gates) is left to ORT CPU.

#include <cctype>
#include <cstdint>
#include <string>
#include <vector>

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mlx {

namespace {

// ---- small handler helpers -----------------------------------------------------------------------

// A node input slot is present when it exists and is not an omitted optional.
bool Present(const NodeDesc& n, size_t i) {
  return i < n.inputs.size() && n.inputs[i].source != Src::Absent;
}

// A dtype-matched scalar (float value cast to `dt`), so mixing it into an fp16/bf16 graph keeps the
// graph dtype instead of promoting the product to fp32.
mlx_array Scalar(TranslationContext& ctx, float value, mlx_dtype dt) {
  mlx_array s = ctx.Keep(mlx_array_new_float32(value));
  return dt == MLX_FLOAT32 ? s : ctx.Astype(s, dt);
}

// Drop leading axis 0 by selecting index d: arr[d] (rank R -> rank R-1). Used to pick a direction's
// slab out of the [num_directions, ...] weight / state tensors.
mlx_array DirSlab(TranslationContext& ctx, mlx_array arr, int d) {
  const std::vector<int> sh = TranslationContext::ShapeOf(arr);
  std::vector<int> start(sh.size(), 0), stop = sh;
  start[0] = d;
  stop[0] = d + 1;
  mlx_array s = ctx.Slice(arr, start, stop);
  std::vector<int> ns(sh.begin() + 1, sh.end());
  return ctx.Reshape(s, ns);
}

// Columns [g*H, (g+1)*H) of a [rows, gates*H] gate block.
mlx_array GateCol(TranslationContext& ctx, mlx_array m, int g, int H) {
  const std::vector<int> sh = TranslationContext::ShapeOf(m);
  return ctx.Slice(m, {0, g * H}, {sh[0], (g + 1) * H});
}

// Sub-vector [a, b) of a 1-D array.
mlx_array Vec1D(TranslationContext& ctx, mlx_array v, int a, int b) {
  return ctx.Slice(v, {a}, {b});
}

// Rows [a, b) of a 2-D [rows, cols] array.
mlx_array Rows(TranslationContext& ctx, mlx_array m, int a, int b) {
  const std::vector<int> sh = TranslationContext::ShapeOf(m);
  return ctx.Slice(m, {a, 0}, {b, sh[1]});
}

mlx_array Sigmoid(TranslationContext& ctx, mlx_array x) {
  mlx_array r = ctx.Keep(mlx_array_new());
  MLX_CHECK(mlx_sigmoid(&r, x, ctx.stream()));
  return r;
}

mlx_array Tanh(TranslationContext& ctx, mlx_array x) {
  mlx_array r = ctx.Keep(mlx_array_new());
  MLX_CHECK(mlx_tanh(&r, x, ctx.stream()));
  return r;
}

mlx_array Zeros(TranslationContext& ctx, const std::vector<int>& shape, mlx_dtype dt) {
  mlx_array r = ctx.Keep(mlx_array_new());
  MLX_CHECK(mlx_zeros(&r, shape.data(), shape.size(), dt, ctx.stream()));
  return r;
}

mlx_array ExpandDims(TranslationContext& ctx, mlx_array a, int axis) {
  mlx_array r = ctx.Keep(mlx_array_new());
  MLX_CHECK(mlx_expand_dims(&r, a, axis, ctx.stream()));
  return r;
}

// Stack a list of same-shaped arrays along a new axis.
mlx_array Stack(TranslationContext& ctx, const std::vector<mlx_array>& arrs, int axis) {
  mlx_vector_array v = mlx_vector_array_new();
  for (mlx_array a : arrs) mlx_vector_array_append_value(v, a);
  mlx_array r = mlx_array_new();
  int rc = mlx_stack_axis(&r, v, axis, ctx.stream());
  mlx_vector_array_free(v);
  MLX_CHECK(rc);
  return ctx.Keep(r);
}

// Clamp `pre` to [-clip, clip] when clip > 0 (ONNX applies clip to the activation inputs). Returns
// `pre` unchanged when clip <= 0.
mlx_array Clip(TranslationContext& ctx, mlx_array pre, float clip, mlx_dtype dt) {
  if (clip <= 0.0f) return pre;
  mlx_array hi = ctx.Keep(mlx_array_new());
  MLX_CHECK(mlx_minimum(&hi, pre, Scalar(ctx, clip, dt), ctx.stream()));
  mlx_array lo = ctx.Keep(mlx_array_new());
  MLX_CHECK(mlx_maximum(&lo, hi, Scalar(ctx, -clip, dt), ctx.stream()));
  return lo;
}

// The number of gate blocks packed into W/R for each op (RNN=1, GRU=3, LSTM=4).
int GateCount(const std::string& op) {
  if (op == "LSTM") return 4;
  if (op == "GRU") return 3;
  return 1;  // RNN
}

// Per-timestep slab Xproj[t] -> [batch, gates*H] from a [S, batch, gates*H] tensor.
mlx_array StepSlab(TranslationContext& ctx, mlx_array xproj, int t, int B, int GH) {
  mlx_array s = ctx.Slice(xproj, {t, 0, 0}, {t + 1, B, GH});
  return ctx.Reshape(s, {B, GH});
}

// One direction's weight / state slabs (the [num_directions, ...] tensors, un-sliced).
struct DirInputs {
  mlx_array W;        // [num_dir, gates*H, I]
  mlx_array R;        // [num_dir, gates*H, H]
  bool has_bias = false;
  mlx_array B;        // [num_dir, 2*gates*H]
  bool has_init_h = false;
  mlx_array init_h;   // [num_dir, B, H]
  bool has_init_c = false;
  mlx_array init_c;   // [num_dir, B, H]  (LSTM)
  bool has_peephole = false;
  mlx_array P;        // [num_dir, 3*H]   (LSTM)
};

// One direction's unrolled recurrence. Produces the per-timestep hidden states Ys[0..S-1] (each
// [B,H], stored at the true timestep index), the final hidden state (into *final_h), and — for LSTM
// — the final cell state (into *final_c). `d` selects the direction slab of the weight/state
// tensors; `reverse` iterates t = S-1..0.
void RunDirection(TranslationContext& ctx, const std::string& op, const DirInputs& in, mlx_array X,
                  int d, bool reverse, int S, int B, int H, int GH, float clip,
                  int linear_before_reset, int input_forget, mlx_dtype dt,
                  std::vector<mlx_array>& Ys, mlx_array* final_h, mlx_array* final_c) {
  mlx_array Wd = DirSlab(ctx, in.W, d);        // [G*H, I]
  mlx_array Rd = DirSlab(ctx, in.R, d);        // [G*H, H]
  mlx_array WdT = ctx.Transpose(Wd, {1, 0});   // [I, G*H]
  mlx_array RdT = ctx.Transpose(Rd, {1, 0});   // [H, G*H]

  // Xproj = X · Wᵀ  (+ Wb), computed once for the whole sequence: [S,B,I] @ [I,G*H] -> [S,B,G*H].
  mlx_array Xproj = ctx.MatMul(X, WdT);
  mlx_array Rb{};  // recurrent bias vector [G*H], valid only when has_bias.
  const bool has_bias = in.has_bias;
  if (has_bias) {
    mlx_array Bd = DirSlab(ctx, in.B, d);      // [2*G*H]
    mlx_array Wb = Vec1D(ctx, Bd, 0, GH);      // [G*H]
    Rb = Vec1D(ctx, Bd, GH, 2 * GH);           // [G*H]
    Xproj = ctx.AddA(Xproj, Wb);
  }

  // Initial states.
  mlx_array H_prev = in.has_init_h ? DirSlab(ctx, in.init_h, d) : Zeros(ctx, {B, H}, dt);
  mlx_array C_prev{};
  if (op == "LSTM") {
    C_prev = in.has_init_c ? DirSlab(ctx, in.init_c, d) : Zeros(ctx, {B, H}, dt);
  }

  // Peepholes (LSTM).
  mlx_array Pi{}, Po{}, Pf{};
  if (op == "LSTM" && in.has_peephole) {
    mlx_array Pd = DirSlab(ctx, in.P, d);      // [3*H]
    Pi = Vec1D(ctx, Pd, 0, H);
    Po = Vec1D(ctx, Pd, H, 2 * H);
    Pf = Vec1D(ctx, Pd, 2 * H, 3 * H);
  }

  Ys.assign(S, mlx_array{});
  for (int step = 0; step < S; ++step) {
    const int t = reverse ? (S - 1 - step) : step;
    mlx_array xg = StepSlab(ctx, Xproj, t, B, GH);   // [B, G*H] (already carries Wb)
    mlx_array Hf = ctx.MatMul(H_prev, RdT);          // [B, G*H]
    if (has_bias) Hf = ctx.AddA(Hf, Rb);             // + Rb

    if (op == "RNN") {
      mlx_array pre = Clip(ctx, ctx.AddA(xg, Hf), clip, dt);
      mlx_array H_new = Tanh(ctx, pre);
      Ys[t] = H_new;
      H_prev = H_new;
    } else if (op == "GRU") {
      // gate order z, r, h
      mlx_array xz = GateCol(ctx, xg, 0, H), xr = GateCol(ctx, xg, 1, H), xh = GateCol(ctx, xg, 2, H);
      mlx_array hz = GateCol(ctx, Hf, 0, H), hr = GateCol(ctx, Hf, 1, H);
      mlx_array zt = Sigmoid(ctx, Clip(ctx, ctx.AddA(xz, hz), clip, dt));
      mlx_array rt = Sigmoid(ctx, Clip(ctx, ctx.AddA(xr, hr), clip, dt));
      mlx_array htpre{};
      if (linear_before_reset != 0) {
        // ht = g(xh + rt ∘ (H_{t-1}·Rhᵀ + Rbh)); Hf's h column already carries Rbh.
        mlx_array hh = GateCol(ctx, Hf, 2, H);
        htpre = ctx.AddA(xh, ctx.Mul(rt, hh));
      } else {
        // ht = g(xh + (rt ∘ H_{t-1})·Rhᵀ + Rbh)
        mlx_array RhT = ctx.Transpose(Rows(ctx, Rd, 2 * H, 3 * H), {1, 0});  // [H,H]
        mlx_array rH = ctx.Mul(rt, H_prev);
        mlx_array hh = ctx.MatMul(rH, RhT);
        if (has_bias) hh = ctx.AddA(hh, Vec1D(ctx, Rb, 2 * H, 3 * H));       // + Rbh
        htpre = ctx.AddA(xh, hh);
      }
      mlx_array ht = Tanh(ctx, Clip(ctx, htpre, clip, dt));
      // Ht = (1 - zt) ∘ ht + zt ∘ H_{t-1}
      mlx_array one_minus_z = ctx.SubA(Scalar(ctx, 1.0f, dt), zt);
      mlx_array H_new = ctx.AddA(ctx.Mul(one_minus_z, ht), ctx.Mul(zt, H_prev));
      Ys[t] = H_new;
      H_prev = H_new;
    } else {  // LSTM, gate order i, o, f, c
      mlx_array xi = GateCol(ctx, xg, 0, H), xo = GateCol(ctx, xg, 1, H);
      mlx_array xf = GateCol(ctx, xg, 2, H), xc = GateCol(ctx, xg, 3, H);
      mlx_array hi = GateCol(ctx, Hf, 0, H), ho = GateCol(ctx, Hf, 1, H);
      mlx_array hf = GateCol(ctx, Hf, 2, H), hc = GateCol(ctx, Hf, 3, H);

      mlx_array ipre = ctx.AddA(xi, hi);
      mlx_array fpre = ctx.AddA(xf, hf);
      if (in.has_peephole) {
        ipre = ctx.AddA(ipre, ctx.Mul(Pi, C_prev));
        fpre = ctx.AddA(fpre, ctx.Mul(Pf, C_prev));
      }
      mlx_array it = Sigmoid(ctx, Clip(ctx, ipre, clip, dt));
      // Couple the input and forget gates: ft = 1 - it (ORT: pf[i] = 1 - pi[i]). The forget gate's
      // own pre-activation (and peephole Pf) is unused in this mode.
      mlx_array ft = input_forget != 0 ? ctx.SubA(Scalar(ctx, 1.0f, dt), it)
                                       : Sigmoid(ctx, Clip(ctx, fpre, clip, dt));
      mlx_array ct = Tanh(ctx, Clip(ctx, ctx.AddA(xc, hc), clip, dt));
      mlx_array C_new = ctx.AddA(ctx.Mul(ft, C_prev), ctx.Mul(it, ct));

      mlx_array opre = ctx.AddA(xo, ho);
      if (in.has_peephole) opre = ctx.AddA(opre, ctx.Mul(Po, C_new));
      mlx_array ot = Sigmoid(ctx, Clip(ctx, opre, clip, dt));
      mlx_array H_new = ctx.Mul(ot, Tanh(ctx, C_new));

      Ys[t] = H_new;
      H_prev = H_new;
      C_prev = C_new;
    }
  }

  *final_h = H_prev;
  if (op == "LSTM" && final_c != nullptr) *final_c = C_prev;
}

// ---- the single handler shared by RNN / GRU / LSTM ---------------------------------------------

void RecurrentOp(TranslationContext& ctx, const NodeDesc& n) {
  const std::string& op = n.op_type;
  const int G = GateCount(op);

  const int H = static_cast<int>(n.ints.at("hidden_size"));
  const std::string direction =
      n.strings.count("direction") ? n.strings.at("direction") : std::string("forward");
  const bool bidir = direction == "bidirectional";
  const bool reverse = direction == "reverse";
  const float clip = n.floats.count("clip") ? n.floats.at("clip") : 0.0f;
  const int linear_before_reset =
      n.ints.count("linear_before_reset") ? static_cast<int>(n.ints.at("linear_before_reset")) : 0;
  const int input_forget =
      n.ints.count("input_forget") ? static_cast<int>(n.ints.at("input_forget")) : 0;

  mlx_array X = ctx.Resolve(n.inputs[0]);   // [S, B, I]
  const std::vector<int> xsh = TranslationContext::ShapeOf(X);
  const int S = xsh[0], B = xsh[1];
  const int GH = G * H;
  const mlx_dtype dt = mlx_array_dtype(X);

  DirInputs in;
  in.W = ctx.Resolve(n.inputs[1]);
  in.R = ctx.Resolve(n.inputs[2]);
  in.has_bias = Present(n, 3);
  if (in.has_bias) in.B = ctx.Resolve(n.inputs[3]);
  // input index 4 is sequence_lens (claimed only when absent).
  in.has_init_h = Present(n, 5);
  if (in.has_init_h) in.init_h = ctx.Resolve(n.inputs[5]);
  if (op == "LSTM") {
    in.has_init_c = Present(n, 6);
    if (in.has_init_c) in.init_c = ctx.Resolve(n.inputs[6]);
    in.has_peephole = Present(n, 7);
    if (in.has_peephole) in.P = ctx.Resolve(n.inputs[7]);
  }

  struct DirResult {
    std::vector<mlx_array> Ys;
    mlx_array final_h;
    mlx_array final_c;
  };
  std::vector<DirResult> results;

  auto run = [&](int d, bool rev) {
    DirResult r;
    mlx_array fc{};
    RunDirection(ctx, op, in, X, d, rev, S, B, H, GH, clip, linear_before_reset, input_forget, dt,
                 r.Ys, &r.final_h, op == "LSTM" ? &fc : nullptr);
    r.final_c = fc;
    results.push_back(std::move(r));
  };

  if (bidir) {
    run(0, /*rev=*/false);
    run(1, /*rev=*/true);
  } else {
    run(0, reverse);
  }

  const int num_dir = static_cast<int>(results.size());

  // Y : [S, num_dir, B, H]. Per direction: stack Ys along axis 0 -> [S,B,H], expand -> [S,1,B,H];
  // concat directions along axis 1.
  if (!n.outputs.empty() && !n.outputs[0].name.empty()) {
    mlx_array Y{};
    for (int d = 0; d < num_dir; ++d) {
      mlx_array ydir = Stack(ctx, results[d].Ys, 0);   // [S,B,H]
      ydir = ExpandDims(ctx, ydir, 1);                 // [S,1,B,H]
      Y = d == 0 ? ydir : ctx.Concat2(Y, ydir, 1);
    }
    ctx.Bind(n.outputs[0], Y);
  }

  // Y_h : [num_dir, B, H].
  if (n.outputs.size() >= 2 && !n.outputs[1].name.empty()) {
    mlx_array Yh{};
    for (int d = 0; d < num_dir; ++d) {
      mlx_array hd = ExpandDims(ctx, results[d].final_h, 0);  // [1,B,H]
      Yh = d == 0 ? hd : ctx.Concat2(Yh, hd, 0);
    }
    ctx.Bind(n.outputs[1], Yh);
  }

  // Y_c : [num_dir, B, H] (LSTM only).
  if (op == "LSTM" && n.outputs.size() >= 3 && !n.outputs[2].name.empty()) {
    mlx_array Yc{};
    for (int d = 0; d < num_dir; ++d) {
      mlx_array cd = ExpandDims(ctx, results[d].final_c, 0);  // [1,B,H]
      Yc = d == 0 ? cd : ctx.Concat2(Yc, cd, 0);
    }
    ctx.Bind(n.outputs[2], Yc);
  }
}

// ---- claim-time helpers -------------------------------------------------------------------------

// Read a STRINGS attribute (e.g. `activations`). Sets `present` to whether the node carries it.
// Returns false only on a genuine read failure of a present STRINGS attribute.
bool StringsAttribute(Ort::ConstNode node, const char* name, std::vector<std::string>& values,
                      bool& present) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK() || static_cast<const OrtOpAttr*>(attr) == nullptr ||
      attr.GetType() == ORT_OP_ATTR_UNDEFINED) {
    present = false;
    values.clear();
    return true;
  }
  present = true;
  return attr.GetType() == ORT_OP_ATTR_STRINGS && attr.GetValueArray(values).IsOK();
}

bool IEqual(const std::string& a, const char* b) {
  std::string lower;
  lower.reserve(a.size());
  for (char c : a) lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  return lower == b;
}

// The default (case-insensitive) activation names per op, repeated once per direction.
bool ActivationsAreDefault(const std::string& op, const std::vector<std::string>& acts,
                           int num_dir) {
  std::vector<const char*> base;
  if (op == "RNN") {
    base = {"tanh"};
  } else if (op == "GRU") {
    base = {"sigmoid", "tanh"};
  } else {  // LSTM
    base = {"sigmoid", "tanh", "tanh"};
  }
  const size_t expected = base.size() * static_cast<size_t>(num_dir);
  if (acts.size() != expected) return false;
  for (size_t i = 0; i < acts.size(); ++i) {
    if (!IEqual(acts[i], base[i % base.size()])) return false;
  }
  return true;
}

// Shared claim predicate for RNN / GRU / LSTM.
bool RecurrentClaim(Ort::ConstNode node, const std::string& op) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() < 3 || outputs.empty()) return false;

  // X (rank-3, static positive seq length), W, R — all the same float dtype.
  ONNXTensorElementDataType xt, wt, rt;
  std::vector<int64_t> xshape, wshape, rshape;
  if (!TensorInfo(inputs[0], xt, &xshape) || !TensorInfo(inputs[1], wt, &wshape) ||
      !TensorInfo(inputs[2], rt, &rshape)) {
    return false;
  }
  if (!IsMlxFloatType(xt) || wt != xt || rt != xt) return false;
  if (xshape.size() != 3) return false;
  if (xshape[0] <= 0) return false;  // dynamic / symbolic seq length -> CPU

  // Optional float inputs must share the float dtype when present.
  auto float_ok = [&](size_t i) {
    if (!SlotPresent(inputs, i)) return true;
    ONNXTensorElementDataType t;
    return TensorInfo(inputs[i], t) && t == xt;
  };
  if (!float_ok(3) || !float_ok(5) || !float_ok(6) || !float_ok(7)) return false;

  // sequence_lens (index 4) present => variable-length masking; leave to CPU.
  if (SlotPresent(inputs, 4)) return false;

  // Default layout only.
  if (IntAttribute(node, "layout", 0) != 0) return false;

  // hidden_size is required for the unroll.
  Ort::ConstOpAttr hs;
  if (!node.GetAttributeByName("hidden_size", hs).IsOK() ||
      static_cast<const OrtOpAttr*>(hs) == nullptr || hs.GetType() != ORT_OP_ATTR_INT) {
    return false;
  }

  // Direction determines num_directions; only forward/reverse/bidirectional are supported.
  Ort::ConstOpAttr dir_attr;
  std::string direction = "forward";
  if (node.GetAttributeByName("direction", dir_attr).IsOK() &&
      static_cast<const OrtOpAttr*>(dir_attr) != nullptr &&
      dir_attr.GetType() == ORT_OP_ATTR_STRING) {
    if (!dir_attr.GetValue(direction).IsOK()) return false;
  }
  int num_dir;
  if (direction == "forward" || direction == "reverse") {
    num_dir = 1;
  } else if (direction == "bidirectional") {
    num_dir = 2;
  } else {
    return false;
  }

  // Only DEFAULT activations are translatable (the STRINGS attribute is not carried into NodeDesc,
  // so the handler hard-codes the defaults). A present-but-non-default set -> CPU.
  std::vector<std::string> acts;
  bool acts_present = false;
  if (!StringsAttribute(node, "activations", acts, acts_present)) return false;
  if (acts_present && !ActivationsAreDefault(op, acts, num_dir)) return false;

  return true;
}

bool RnnClaim(Ort::ConstNode node) { return RecurrentClaim(node, "RNN"); }
bool GruClaim(Ort::ConstNode node) { return RecurrentClaim(node, "GRU"); }
bool LstmClaim(Ort::ConstNode node) { return RecurrentClaim(node, "LSTM"); }

}  // namespace

void RegisterRecurrentOps(OpRegistry& registry) {
  registry.Register({"", "RNN", kAnyOpset, kAnyOpset, &RecurrentOp, &RnnClaim});
  registry.Register({"", "GRU", kAnyOpset, kAnyOpset, &RecurrentOp, &GruClaim});
  registry.Register({"", "LSTM", kAnyOpset, kAnyOpset, &RecurrentOp, &LstmClaim});
}

}  // namespace ort_mlx
