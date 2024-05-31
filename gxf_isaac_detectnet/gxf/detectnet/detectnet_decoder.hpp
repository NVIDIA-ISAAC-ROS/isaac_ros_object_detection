// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVIDIA_ISAAC_ROS_EXTENSIONS_DETECTNET_DECODER_HPP_
#define NVIDIA_ISAAC_ROS_EXTENSIONS_DETECTNET_DECODER_HPP_

#include "gxf/core/entity.hpp"
#include "gxf/core/gxf.h"
#include "gxf/core/parameter.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/core/parameter_parser_std.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"
#include "detection2_d_array_message.hpp"
#include "deepstream_utils/nvdsinferutils/dbscan/nvdsinfer_dbscan.hpp"


namespace nvidia
{
namespace isaac_ros
{

// GXF codelet that decodes detections from a tensor and converts it to a detections_2d_array
class DetectnetDecoder : public gxf::Codelet
{
public:
  gxf_result_t registerInterface(gxf::Registrar * registrar) noexcept override;
  gxf_result_t start() noexcept override;
  gxf_result_t tick() noexcept override;
  gxf_result_t stop() noexcept override {return GXF_SUCCESS;}

private:
  gxf::Parameter<gxf::Handle<gxf::Receiver>> tensorlist_receiver_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> detections_transmitter_;

  gxf::Parameter<std::vector<std::string>> label_list_;
  gxf::Parameter<bool> enable_confidence_threshold_;
  gxf::Parameter<bool> enable_bbox_area_threshold_;
  gxf::Parameter<bool> enable_dbscan_clustering_;
  gxf::Parameter<double> confidence_threshold_;
  gxf::Parameter<double> min_bbox_area_;
  gxf::Parameter<double> dbscan_confidence_threshold_;
  gxf::Parameter<double> dbscan_eps_;
  gxf::Parameter<int> dbscan_min_boxes_;
  gxf::Parameter<int> dbscan_enable_athr_filter_;
  gxf::Parameter<double> dbscan_threshold_athr_;
  gxf::Parameter<int> dbscan_clustering_algorithm_;
  gxf::Parameter<double> bounding_box_scale_;
  gxf::Parameter<double> bounding_box_offset_;

  // tuning parameters data structure dbscan library
  NvDsInferDBScanClusteringParams params_;
};

}  // namespace isaac_ros
}  // namespace nvidia

#endif  // NVIDIA_ISAAC_ROS_EXTENSIONS_DETECTNET_DECODER_HPP_
