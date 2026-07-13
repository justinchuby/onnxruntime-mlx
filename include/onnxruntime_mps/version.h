// Copyright (c) 2026. Licensed under the MIT License.
//
// Version information for the MLX-native ONNX Runtime execution provider plugin.
//
// NOTE: ORT_MPS_EP_NAME stays "MetalEP" for wire compatibility — onnx-genai registers and binds the
// device by this exact EP name (crates/onnx-genai-ort/src/session.rs). The vendor string carries the
// repo rename (onnxruntime-mps -> onnxruntime-mlx).

#pragma once

#define ORT_MPS_EP_NAME "MetalEP"
#define ORT_MPS_EP_VENDOR "onnxruntime-mlx"

// Apple's PCI-SIG vendor id (0x106B). Used as the OrtEpFactory vendor id.
#define ORT_MPS_EP_VENDOR_ID 0x106B

#define ORT_MPS_EP_VERSION_MAJOR 0
#define ORT_MPS_EP_VERSION_MINOR 1
#define ORT_MPS_EP_VERSION_PATCH 0
#define ORT_MPS_EP_VERSION "0.1.0"
