// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef __NVDS_TRANSFORM_H__
#define __NVDS_TRANSFORM_H__

#if defined(__cplusplus)
extern "C" {
#endif

#include <cuda.h>
#include <cuda_runtime_api.h>

/** API return values */
typedef enum {
  /** Success */
  NVDST_STATUS_SUCCESS = 0,
  /** Failure */
  NVDST_STATUS_FAILED = 1,
  /** Handle invalid */
  NVDST_STATUS_INVALID_HANDLE = 2,
  /** Parameter value invalid */
  NVDST_STATUS_INVALID_PARAM = 3,
  /** Insufficient input data to process */
  NVDST_STATUS_INSUFFICIENT_DATA = 4,
  /** Transform not supported */
  NVDST_STATUS_TRANSFORM_NOT_AVAILABLE = 5,
  /** Given buffer length too small to hold requested data */
  NVDST_STATUS_BUFFER_TOO_SMALL = 6,
} NvDst_Status;

/** Each transform instantiation is associated with an opaque handle. */
typedef void *NvDst_Handle;

/** Transform selector as string */
typedef const char *NvDst_TransformSelector;

/** Parameter selector as string */
typedef const char *NvDst_ParameterSelector;

/** @brief Create a new instance of an transform.
 *
 * @param[in] code  The selector code for the desired transform.
 * @param[out] transform  A handle to the transform instantiation.
 *
 * @return  Status values as enumerated in @ref NvDst_Status
 */
NvDst_Status NvDst_CreateTransform (NvDst_TransformSelector transform, NvDst_Handle *handle);

/** @brief Delete a previously instantiated transform.
 *
 * @param[in] transform  A handle to the transform to be deleted.
 *
 * @return  Status values as enumerated in @ref NvDst_Status
 */
NvDst_Status NvDst_DestroyTransform (NvDst_Handle handle);

/** @brief Set the value of the selected parameter (unsigned int, float, char*)
 *
 * @param[in]  transform   The transform to configure.
 * @param[in]  param_name  The selector of the transfor parameter to configure.
 * @param[in]  val         The value to be assigned to the selected transform parameter.
 *
 * @return  Status values as enumerated in @ref NvDst_Status
 */
NvDst_Status NvDst_SetU32(NvDst_Handle transform, NvDst_ParameterSelector param_name,
                                    unsigned int val);
NvDst_Status NvDst_SetFloat(NvDst_Handle transform, NvDst_ParameterSelector param_name,
                                    float val);
NvDst_Status NvDst_SetString(NvDst_Handle transform, NvDst_ParameterSelector param_name,
                                       const char* val);

/** @brief Get the value of the selected parameter (unsigned int, float, char*)
*
* @param[in]  transform   The transform handle.
* @param[in]  param_name  The selector of the transform parameter to read.
* @param[out] val         Buffer in which the parameter value will be assigned.
* @param[in]  max_length  The length in bytes of the buffer provided.
*
* @return  Status values as enumerated in @ref NvDst_Status
*/
NvDst_Status NvDst_GetU32(NvDst_Handle transform, NvDst_ParameterSelector param_name,
                                    unsigned int* val);
NvDst_Status NvDst_GetFloat(NvDst_Handle transform, NvDst_ParameterSelector param_name,
                                    float* val);
NvDst_Status NvDst_GetString(NvDst_Handle transform, NvDst_ParameterSelector param_name,
                                       char* val, int max_length);

/** @brief Initializes the DSP transform based on the set params.
 *
 * @param[in]  transform  The transform object handle.
 *
 * @return  Status values as enumerated in @ref NvDst_Status
 */
NvDst_Status NvDst_LoadTransform (NvDst_Handle handle, unsigned int frame_size,
                                       unsigned int num_channels);

/** @brief Return the exact amount of output buffer needed by NvDst_Run()
 *
 * @note To be called only after NvDst_Load(). If called before, it may return invalid results
 *
 * @param[in]  transform  The transform handle.
 * @param[in]  num_output_elements  The number of expected elements per channel to be allocated for the
 *                                              output buffer.
 *
 * @return  Status values as enumerated in @ref NvDst_Status
 */
NvDst_Status NvDst_GetExpectedOutputElements (NvDst_Handle handle, unsigned int *num_output_elements);

/** @brief Process the input buffer as per the Audio transform selected. e.g. mel spectrogram
 *
 * @param[in]  transform  The transform handle.
 * @param[in]  input      Input float buffer array. It points to an array of buffers where each buffer
 *                        holds audio data for a single stream (stream could be batch or channel).
 *                        Array size should be same as number of streams expected by the transform.
 * @param[out]  output    Output float buffer array. The layout is same as input. It points to an
 *                        array of buffers where each buffer has data corresponding to that
 *                        stream. The buffers have to be preallocated by caller. Size of each buffer
 *                        depends on the transform applied on the input.
 *
 * @return  Status values as enumerated in @ref NvDst_Status
 */
NvDst_Status NvDst_Run(NvDst_Handle transform, void** input, void** output,
                             unsigned int num_input_elements, unsigned int num_output_element,
                             unsigned int num_channels, cudaStream_t stream);

#if defined(__cplusplus)
}
#endif

#endif /* __NVDS_TRANSFORM_H__ */
