// Copyright (c) 2026. Licensed under the MIT License.
//
// OrtDataTransferImpl for the Metal EP. On Apple unified memory the EP's device buffers are
// shared-storage MTLBuffers whose contents are CPU-addressable, so host<->device and
// device<->device transfers are plain memcpy (see docs/DESIGN.md 2.6). Phase 2 may switch
// private-storage buffers to a blit encoder.

#pragma once

#include "plugin_ep_utils.h"

struct MetalDataTransfer : OrtDataTransferImpl, ApiPtrs {
  MetalDataTransfer(ApiPtrs apis, const OrtMemoryDevice* device_memory)
      : ApiPtrs(apis), device_memory_(device_memory) {
    ort_version_supported = ORT_API_VERSION;
    CanCopy = CanCopyImpl;
    CopyTensors = CopyTensorsImpl;
    Release = ReleaseImpl;
  }

  static bool ORT_API_CALL CanCopyImpl(const OrtDataTransferImpl* this_ptr,
                                       const OrtMemoryDevice* src,
                                       const OrtMemoryDevice* dst) noexcept;

  static OrtStatus* ORT_API_CALL CopyTensorsImpl(OrtDataTransferImpl* this_ptr,
                                                 const OrtValue** src_tensors,
                                                 OrtValue** dst_tensors,
                                                 OrtSyncStream** streams,
                                                 size_t num_tensors) noexcept;

  static void ORT_API_CALL ReleaseImpl(OrtDataTransferImpl* this_ptr) noexcept;

 private:
  const OrtMemoryDevice* device_memory_;  // the memory device this EP owns
};
