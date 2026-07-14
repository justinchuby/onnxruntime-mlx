# Metal Tracing Design — onnxruntime-mlx

> GPU-level profiling and tracing for the MLX EP on Apple Silicon.
> Integrates with nxrt's unified tracing (§48) and multi-backend architecture (§48.8).

---

## 1. Problem

MLX EP runs Metal compute shaders on Apple GPU. We need:
- Per-kernel GPU timing (which shader took how long)
- Correlation back to ONNX op (which MatMul caused this shader dispatch)
- Memory bandwidth utilization
- Integration with our Perfetto trace output
- Low overhead (usable during benchmarking, not just debugging)

MLX itself only offers `mx.metal.start_capture()` for Xcode GPU Debugger — not
a programmatic trace API we can embed.

---

## 2. Apple Profiling Stack (What's Available)

| Layer | Apple API | Equivalent to | Use case |
|-------|-----------|---------------|----------|
| Code annotation | **os_signpost** | Intel ITT | Mark regions for Instruments |
| GPU kernel timing | **MTLCommandBuffer timestamps** | CUPTI Activity | Per-dispatch GPU time |
| GPU counters | **MTLCounterSampleBuffer** | CUPTI Metrics | Occupancy, bandwidth |
| Full capture | **MTLCaptureManager** | NSight | Xcode GPU Debugger |

---

## 3. Architecture

```
┌────────────────────────────────────────────────────────────┐
│  nxrt Unified Tracing (§48 TraceContext)                   │
├────────────────────────────────────────────────────────────┤
│  CompositeCollector (§48.8)                                │
│    ├─ PerfettoCollector     → .perfetto file               │
│    ├─ SignpostCollector      → Instruments (if attached)    │
│    └─ MetalTimingCollector  → GPU kernel timestamps        │
│                                ↓ merged into Perfetto      │
├────────────────────────────────────────────────────────────┤
│  MLX EP Dispatch Layer                                     │
│    → records correlation: NodeId ↔ command buffer index    │
│    → sets up timestamp sampling on MTLCommandBuffer        │
├────────────────────────────────────────────────────────────┤
│  Metal (GPU)                                               │
└────────────────────────────────────────────────────────────┘
```

---

## 4. GPU Kernel Timing (MTLCommandBuffer Timestamps)

Metal provides per-command-buffer timestamps without Xcode attached:

```swift
// When dispatching a compute kernel for an ONNX op:
let commandBuffer = commandQueue.makeCommandBuffer()!

// Encode compute work
let encoder = commandBuffer.makeComputeCommandEncoder()!
encoder.setComputePipelineState(matmulPipeline)
encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: blockSize)
encoder.endEncoding()

// After commit + completion:
commandBuffer.addCompletedHandler { cb in
    let gpuStart = cb.gpuStartTime   // seconds (Mach absolute time)
    let gpuEnd = cb.gpuEndTime       // seconds
    let duration = gpuEnd - gpuStart
    // → emit as TraceEvent to unified tracer
}

commandBuffer.commit()
```

**Rust FFI (via objc2/metal-rs):**

```rust
pub struct MetalTimingCollector {
    /// Correlation: command buffer index → (NodeId, op_type)
    correlation: DashMap<u64, KernelCorrelation>,
    /// Collected timing records
    records: Mutex<Vec<MetalKernelRecord>>,
}

#[derive(Debug, Clone)]
pub struct MetalKernelRecord {
    pub node_id: NodeId,
    pub op_type: String,
    pub kernel_name: String,        // Metal pipeline function name
    pub gpu_start_ns: u64,
    pub gpu_end_ns: u64,
    pub duration_ns: u64,
    pub command_buffer_index: u64,
}

impl MetalTimingCollector {
    /// Called by MLX EP when dispatching a kernel
    pub fn correlate(&self, cb_index: u64, node_id: NodeId, op_type: &str) {
        self.correlation.insert(cb_index, KernelCorrelation {
            node_id,
            op_type: op_type.to_string(),
        });
    }

    /// Called from command buffer completion handler
    pub fn record_completion(
        &self,
        cb_index: u64,
        gpu_start_ns: u64,
        gpu_end_ns: u64,
        kernel_name: &str,
    ) {
        if let Some(corr) = self.correlation.remove(&cb_index) {
            self.records.lock().unwrap().push(MetalKernelRecord {
                node_id: corr.1.node_id,
                op_type: corr.1.op_type,
                kernel_name: kernel_name.to_string(),
                gpu_start_ns,
                gpu_end_ns,
                duration_ns: gpu_end_ns - gpu_start_ns,
                command_buffer_index: cb_index,
            });
        }
    }
}
```

---

## 5. os_signpost Integration (Apple's ITT)

`os_signpost` is Apple's equivalent of Intel ITT — zero overhead when Instruments
isn't recording, full visibility when it is.

```rust
use os_signpost::{SignpostLog, SignpostInterval};

pub struct SignpostCollector {
    log: SignpostLog,
}

impl SignpostCollector {
    pub fn new() -> Self {
        Self {
            log: SignpostLog::new("com.nxrt.runtime", "Operations"),
        }
    }
}

impl TraceCollector for SignpostCollector {
    fn emit(&self, event: &TraceEvent) {
        match event.phase {
            TracePhase::Begin => {
                self.log.begin_interval(&event.name);
            }
            TracePhase::End => {
                self.log.end_interval(&event.name);
            }
            TracePhase::Instant => {
                self.log.emit_event(&event.name);
            }
            _ => {}
        }
    }

    fn flush(&self) -> Result<()> {
        Ok(()) // signpost doesn't need flush
    }
}
```

**What this gives you in Instruments:**
```
Instruments → Metal System Trace + os_signpost:

Timeline:
├─ nxrt.runtime (signpost)
│   ├─ MatMul_0      ████████████
│   ├─ LayerNorm_1        ███
│   └─ Softmax_2              █████
├─ GPU (Metal)
│   ├─ matmul_f16_kernel     ████████
│   └─ layernorm_kernel          ██
```

---

## 6. Metal GPU Counters (Detailed Profiling)

For deeper analysis (occupancy, bandwidth), Metal provides counter sampling:

```swift
// Create counter sample buffer
let descriptor = MTLCounterSampleBufferDescriptor()
descriptor.counterSet = device.counterSets?.first(where: { $0.name == "GPUTimestamp" })
descriptor.sampleCount = 1024
descriptor.storageMode = .shared
let counterBuffer = device.makeCounterSampleBuffer(descriptor: descriptor)!

// Sample at compute command boundaries
encoder.sampleCounters(sampleBuffer: counterBuffer, sampleIndex: 0, barrier: true)
// ... dispatch ...
encoder.sampleCounters(sampleBuffer: counterBuffer, sampleIndex: 1, barrier: true)
```

Available counter sets (device-dependent):
- `GPUTimestamp` — precise GPU timestamps
- `StatisticSet` — ALU utilization, memory bandwidth (M1+)

**Note:** Counter availability varies by chip. M1 has limited counters, M3+ has more.
Always query `device.counterSets` and gracefully degrade.

---

## 7. Perfetto Integration

Metal timing records merge into the same Perfetto trace as all other backends:

```
Process: nxrt (pid=1)
├─ Thread: runtime.compute         → op dispatch (§48)
├─ Thread: runtime.metal.gpu       → Metal kernel execution (this)   ← NEW
├─ Counter: metal.gpu_util         → GPU utilization %               ← NEW
├─ Counter: metal.bandwidth_util   → Memory bandwidth %              ← NEW
└─ Flow: dispatch → kernel         → correlation arrows
```

---

## 8. Profiling Modes

| Mode | Overhead | What you get | API |
|------|----------|-------------|-----|
| **Off** | 0% | Nothing | Default for production |
| **Timing** | 1-3% | Per-kernel GPU start/end | `commandBuffer.gpuStartTime/gpuEndTime` |
| **Signpost** | 0% (no Instruments) / 2% (recording) | Instruments visibility | `os_signpost` |
| **Counters** | 5-15% | ALU/bandwidth utilization | `MTLCounterSampleBuffer` |
| **Capture** | High (Xcode only) | Full GPU state inspection | `MTLCaptureManager` |

```python
import nxrt

session = nxrt.load("model.onnx", device="mlx")

# Timing mode (low overhead)
with nxrt.profiler.profile(gpu="timing") as prof:
    session.run(inputs)
for kernel in prof.gpu_kernels():
    print(f"{kernel.op_type}: {kernel.duration_ns/1000:.1f}μs")

# Export to Perfetto (includes Metal GPU tracks)
prof.export_perfetto("trace.perfetto")
```

---

## 9. Integration with MLX EP

```rust
// In the MLX EP dispatch path:
impl MlxExecutionProvider {
    fn execute_kernel(
        &self,
        node: &Node,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
    ) -> Result<()> {
        // 1. Register correlation
        let cb_index = self.next_command_buffer_index();
        if let Some(tracer) = &self.metal_timing {
            tracer.correlate(cb_index, node.id, &node.op_type);
        }

        // 2. Dispatch via MLX (which internally creates Metal command buffers)
        let mlx_outputs = mlx_dispatch(node, inputs)?;

        // 3. On completion → timing automatically recorded via completion handler
        // (set up once during EP initialization)

        copy_outputs(mlx_outputs, outputs)?;
        Ok(())
    }
}
```

---

## 10. Design Decisions

| Decision | Choice | Rationale |
|----------|--------|----------|
| Primary timing source | **MTLCommandBuffer gpuStart/EndTime** | Low overhead, no Xcode needed, always available |
| Apple ITT equivalent | **os_signpost** | Zero cost without Instruments, Apple-native |
| GPU counters | **Optional, M3+ only** | Limited availability on older chips |
| MTLCaptureManager | **Dev-only (not in release)** | Requires Xcode, high overhead |
| Perfetto integration | **Yes** (same trace file) | Unified view across all platforms |
| MLX internal tracing | **Don't depend on it** | MLX's capture is Xcode-only, not programmatic |

---

## 11. Implementation Priority

1. **MTLCommandBuffer timing** — immediate value, low effort
2. **Perfetto export with Metal tracks** — unified experience
3. **os_signpost** — free, helps Apple-native devs
4. **GPU counters** — stretch, chip-dependent
