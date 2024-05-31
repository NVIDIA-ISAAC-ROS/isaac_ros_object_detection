// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#ifndef __NVDSINFER_DBSCANCLUSTERING_HPP__
#define __NVDSINFER_DBSCANCLUSTERING_HPP__

#include <memory>
#include <vector>

/* Ignore errors from open source headers. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#if __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#endif
#if __GNUC__ >= 7
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#endif
#include <Eigen/Dense>
#pragma GCC diagnostic pop
#include "EigenDefs.hpp"

#include <nvdsinfer_dbscan.h>

/**
 * Holds the bounding box co-ordindates.
 */
typedef struct
{
    float x;
    float y;
    float width;
    float height;
} BBox;

typedef struct
{
    /// Bounding box of the detected object.
    BBox box;
    /// Variance of bounding boxes of the members of the cluster.
    BBox boxVariance;
    /// Total confidence of the members of the cluster.
    float totalConfidence;
    /// Maximum confidence of the members of the cluster.
    float maximumConfidence;
    /// Number of members of the cluster.
    uint32_t numMembers;
} ClusteredObject;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
* NvDsInferDBScan class
**/
struct NvDsInferDBScan {
private:
    typedef Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixXi;

    NvDsInferDBScan();
    ~NvDsInferDBScan();

    void clusterObjects(NvDsInferObjectDetectionInfo *detections, size_t &numDetections,
            NvDsInferDBScanClusteringParams *params, bool hybridClustering = false);

    void re_allocate (size_t num_detections);

    void buildDistanceMatrix(const NvDsInferObjectDetectionInfo input[],
            int32_t inputSize);

    void findNeighbors(int32_t boxIdx, int32_t inputSize, float eps);

    void joinNeighbors(int32_t boxDstIdx, int32_t boxSrcIdx);

    std::unique_ptr<MatrixXf> m_distanceMatrix;
    std::unique_ptr<RowMatrixXi> m_neighborListMatrix;
    std::unique_ptr<MatrixXb> m_neighborshipMatrix;
    std::vector<uint32_t> m_neighborCounts;
    std::vector<int32_t> m_labels;
    std::vector<bool> m_visited;
    size_t m_maxProposals;
    std::vector <ClusteredObject> m_clusteredObjects;

    friend NvDsInferDBScanHandle NvDsInferDBScanCreate();
    friend void NvDsInferDBScanDestroy(NvDsInferDBScanHandle handle);
    friend void NvDsInferDBScanCluster(NvDsInferDBScanHandle handle,
            NvDsInferDBScanClusteringParams *params,
            NvDsInferObjectDetectionInfo *objects, size_t *numObjects);
    friend void NvDsInferDBScanClusterHybrid(NvDsInferDBScanHandle handle,
            NvDsInferDBScanClusteringParams *params,
            NvDsInferObjectDetectionInfo *objects, size_t *numObjects);
};

#endif
