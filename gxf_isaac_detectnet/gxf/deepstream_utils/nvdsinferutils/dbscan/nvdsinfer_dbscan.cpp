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

#include <cstdint>

#pragma GCC diagnostic push
#if __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "nvdsinfer_dbscan.hpp"
#pragma GCC diagnostic pop

static const int INIT_MAX_PROPOSALS = 800;

NvDsInferDBScan::NvDsInferDBScan()
    : m_maxProposals(INIT_MAX_PROPOSALS)
{
    // Create a distance matrix for storing computed distance values
    m_distanceMatrix.reset(new MatrixXf(m_maxProposals, m_maxProposals));
    // Each row idx refers to an input sample, and each row is a list of neighbor
    // indicies for that sample
    m_neighborListMatrix.reset(new RowMatrixXi(m_maxProposals, m_maxProposals));
    // Matrix indicating neighborship between samples
    m_neighborshipMatrix.reset(new MatrixXb(m_maxProposals, m_maxProposals));

    // Each idx refers to an input sample, and stores the number of neighbors
    // for that sample
    m_neighborCounts.resize(m_maxProposals);
    std::fill(m_neighborCounts.begin(), m_neighborCounts.end(), 0);
    // Store label id for each sample
    m_labels.resize(m_maxProposals);
    std::fill(m_labels.begin(), m_labels.end(), 0);
    // Store an indicator for being visited for each sample.
    m_visited.resize(m_maxProposals);
    std::fill(m_visited.begin(), m_visited.end(), 0);

    m_clusteredObjects.resize (m_maxProposals);
}

void
NvDsInferDBScan::re_allocate(size_t num_detections)
{
    m_maxProposals = num_detections;
    m_distanceMatrix.reset(new MatrixXf(m_maxProposals, m_maxProposals));
    // Each row idx refers to an input sample, and each row is a list of neighbor indicies for that sample
    m_neighborListMatrix.reset(new RowMatrixXi(m_maxProposals, m_maxProposals));
    // Matrix indicating neighborship between samples
    m_neighborshipMatrix.reset(new MatrixXb(m_maxProposals, m_maxProposals));

    m_neighborCounts.resize(m_maxProposals);
    std::fill(m_neighborCounts.begin(), m_neighborCounts.end(), 0);
    m_labels.resize(m_maxProposals);
    std::fill(m_labels.begin(), m_labels.end(), 0);
    m_visited.resize(m_maxProposals);
    std::fill(m_visited.begin(), m_visited.end(), 0);

    m_clusteredObjects.resize (m_maxProposals);
}

NvDsInferDBScan::~NvDsInferDBScan()
{
}

static inline float
intersectionOverUnion(const NvDsInferObjectDetectionInfo &boxA,
        const NvDsInferObjectDetectionInfo &boxB)
{
    float areaA = boxA.width * boxA.height;
    float areaB = boxB.width * boxB.height;

    // Compute intersection bounding box
    float isectLeft = std::max(boxA.left, boxB.left);
    float isectTop = std::max(boxA.top, boxB.top);
    float isectRight = std::min(boxA.left + boxA.width, boxB.left + boxB.width);
    float isectBottom = std::min(boxA.top + boxA.height, boxB.top + boxB.height);
    float isectWidth = std::max(0.0f, isectRight - isectLeft);
    float isectHeight = std::max(0.0f, isectBottom - isectTop);
    float isectArea = isectWidth * isectHeight;

    // Compute intersection over union
    float iou = isectArea / ((areaA + areaB) - isectArea);

    return 1.0f - iou;
}

static inline void
initializeObject(ClusteredObject &object)
{
    object.box = {0.0f, 0.0f, 0.0f, 0.0f};
    object.boxVariance = {0.0f, 0.0f, 0.0f, 0.0f};
    object.maximumConfidence = 0.0f;
    object.numMembers = 1U;
    object.totalConfidence = 0.0f;
}

void
NvDsInferDBScan::clusterObjects(NvDsInferObjectDetectionInfo *detections,
        size_t &numDetections, NvDsInferDBScanClusteringParams *params, bool hybridClustering)
{
    if (numDetections > m_maxProposals)
    {
        m_maxProposals = numDetections;
        re_allocate (numDetections);
    }

    int32_t inputSizeInt = static_cast<int32_t>(numDetections);

    // Reset everything
    m_neighborshipMatrix->setZero(inputSizeInt, inputSizeInt);
    std::fill(m_visited.begin(), m_visited.end(), 0);
    std::fill(m_neighborCounts.begin(), m_neighborCounts.end(), 0);
    std::fill(m_labels.begin(), m_labels.end(), -1);

    buildDistanceMatrix(detections, inputSizeInt);

    int32_t clusterId = 0;

    // Get references to the pointers to avoid dereferencing it everytime
    RowMatrixXi &neighborMatrix = *m_neighborListMatrix;

    for (int32_t inpIdx = 0; inpIdx < inputSizeInt; ++inpIdx)
    {
        if (!m_visited[inpIdx])
        {
            m_visited[inpIdx] = true;

            findNeighbors(inpIdx, inputSizeInt, params->eps);

            uint32_t neighborCount = m_neighborCounts[inpIdx];
            // Check if it has enough neighbors
            if ((neighborCount + 1) >= params->minBoxes)
            {
                m_labels[inpIdx] = clusterId;

                // Traverse through the neighbors
                for (int32_t nbIdx = 0;
                        nbIdx < static_cast<int32_t>(neighborCount); ++nbIdx)
                {
                    int32_t neighborIdx = neighborMatrix(inpIdx, nbIdx);

                    if (!m_visited[neighborIdx])
                    {
                        m_visited[neighborIdx] = true;

                        findNeighbors(neighborIdx, inputSizeInt, params->eps);

                        if ((m_neighborCounts[neighborIdx] + 1) >=
                                params->minBoxes)
                        {
                            // Merge neighbors of the neighbor box with
                            // neighbors of the current box
                            joinNeighbors(inpIdx, neighborIdx);
                            // All neighbors of the box should be the same label
                            m_labels[neighborIdx] = clusterId;
                        }
                    }
                }

                // Create new label
                clusterId++;
            }
        }
    }

    //Return partially clustered outputs to be proccessed by NMS
    if(hybridClustering)
    {
        std::vector<NvDsInferObjectDetectionInfo> unclusterdProposals;
        for(uint32_t inpIdx = 0U; inpIdx < numDetections; ++inpIdx)
        {
            if(m_labels[inpIdx] != -1)
                unclusterdProposals.push_back(detections[inpIdx]);
        }

        numDetections = unclusterdProposals.size();
        for(uint32_t i = 0; i < unclusterdProposals.size(); ++i)
        {
            detections[i].left = unclusterdProposals[i].left;
            detections[i].top = unclusterdProposals[i].top;
            detections[i].width = unclusterdProposals[i].width;
            detections[i].height = unclusterdProposals[i].height;
            detections[i].detectionConfidence =
                        unclusterdProposals[i].detectionConfidence;
            detections[i].classId = unclusterdProposals[i].classId;
        }

        return;
    }

    // Traverse through the bounding boxes to get the final clusters
    uint32_t outIdx = 0U;
    for (int32_t cIdx = 0; cIdx < clusterId; ++cIdx)
    {
        ClusteredObject &cluster = m_clusteredObjects[outIdx];
        BBox &clusterBox = cluster.box;
        BBox &clusterBoxVariance = cluster.boxVariance;
        uint32_t &nElements = cluster.numMembers;

        initializeObject(cluster);

        // Compute first order statistics
        for (uint32_t inpIdx = 0U; inpIdx < numDetections; ++inpIdx)
        {
            if (m_labels[inpIdx] == cIdx)
            {
                const NvDsInferObjectDetectionInfo &inputBox = detections[inpIdx];

                cluster.maximumConfidence = std::max(cluster.maximumConfidence,
                        inputBox.detectionConfidence);
                cluster.totalConfidence += inputBox.detectionConfidence;

                clusterBox.x += inputBox.detectionConfidence * inputBox.left;
                clusterBox.y += inputBox.detectionConfidence * inputBox.top;
                clusterBox.width +=
                    inputBox.detectionConfidence * inputBox.width;
                clusterBox.height +=
                    inputBox.detectionConfidence * inputBox.height;

                ++nElements;
            }
        }

        if (cluster.totalConfidence < params->minScore)
            continue;

        // Compute weighted average bounding box
        clusterBox.x = clusterBox.x / cluster.totalConfidence;
        clusterBox.y = clusterBox.y / cluster.totalConfidence;
        clusterBox.width = clusterBox.width / cluster.totalConfidence;
        clusterBox.height = clusterBox.height / cluster.totalConfidence;

        // Run ATHR filter if desired
        bool clusterValid = true;
        if (params->enableATHRFilter)
        {
            float areaSqrt = std::sqrt(cluster.box.width * cluster.box.height);
            float areaHitRatio = areaSqrt / static_cast<float>(nElements);
            if (areaHitRatio > params->thresholdATHR)
            {
                clusterValid = false;
            }
        }

        if (clusterValid)
        {
            // Compute bounding box variance
            for (uint32_t inpIdx = 0U; inpIdx < numDetections; ++inpIdx)
            {
                if (m_labels[inpIdx] == cIdx)
                {
                    const NvDsInferObjectDetectionInfo &inputBox = detections[inpIdx];
                    // Changing coordinates to left, top, right, bottom
                    clusterBoxVariance.x += inputBox.detectionConfidence *
                        (inputBox.left - clusterBox.x) *
                        (inputBox.left - clusterBox.x);
                    clusterBoxVariance.y += inputBox.detectionConfidence *
                        (inputBox.top - clusterBox.y) *
                        (inputBox.top - clusterBox.y);
                    clusterBoxVariance.width += inputBox.detectionConfidence *
                        (inputBox.left + inputBox.width - clusterBox.x -
                                clusterBox.width) *
                        (inputBox.left + inputBox.width - clusterBox.x -
                                clusterBox.width);
                    clusterBoxVariance.height += inputBox.detectionConfidence *
                        (inputBox.top + inputBox.height - clusterBox.y -
                                clusterBox.height) *
                        (inputBox.top + inputBox.height - clusterBox.y -
                                clusterBox.height);
                }
            }

            // Compute variance
            clusterBoxVariance.x /= cluster.totalConfidence;
            clusterBoxVariance.y /= cluster.totalConfidence;
            clusterBoxVariance.width /= cluster.totalConfidence;
            clusterBoxVariance.height /= cluster.totalConfidence;
            // This cluster is valid and will be output. Proceed to the next one.
            outIdx++;
        }
    }

    // Update the output size
    numDetections = std::min(numDetections, static_cast<size_t>(outIdx));
    for (size_t i = 0; i < numDetections; i++)
    {
        detections[i].left = m_clusteredObjects[i].box.x;
        detections[i].top = m_clusteredObjects[i].box.y;
        detections[i].width = m_clusteredObjects[i].box.width;
        detections[i].height = m_clusteredObjects[i].box.height;
        detections[i].detectionConfidence =
            m_clusteredObjects[i].maximumConfidence;
    }
}

void
NvDsInferDBScan::buildDistanceMatrix(const NvDsInferObjectDetectionInfo input[],
        int32_t inputSize)
{
    MatrixXf &distanceMatrix = *m_distanceMatrix;
    for (int32_t idx1 = 0; idx1 < inputSize; ++idx1)
    {
        for (int32_t idx2 = idx1 + 1; idx2 < inputSize; ++idx2)
        {
            float distance = intersectionOverUnion(input[idx1], input[idx2]);
            distanceMatrix(idx1, idx2) = distance;
            distanceMatrix(idx2, idx1) = distance;
        }
    }
}

void
NvDsInferDBScan::findNeighbors(int32_t boxIdx, int32_t inputSize, float eps)
{
    uint32_t neighborIdx = m_neighborCounts[boxIdx];
    MatrixXf &distanceMatrix = *m_distanceMatrix;
    RowMatrixXi &neighborMatrix = *m_neighborListMatrix;
    MatrixXb &neighborshipMatrix = *m_neighborshipMatrix;

    for (int32_t inpIdx = 0; inpIdx < inputSize; ++inpIdx)
    {
        // A box is a neighbor of another box if the distance is smaller than
        // the epsilon
        float distance = distanceMatrix(boxIdx, inpIdx);
        if ((inpIdx != boxIdx) && (distance < eps))
        {
            neighborMatrix(boxIdx, neighborIdx) = inpIdx;
            neighborshipMatrix(boxIdx, inpIdx) = 1;
            neighborIdx++;
        }
    }

    // Update neighbor count
    m_neighborCounts[boxIdx] = neighborIdx;
}

void
NvDsInferDBScan::joinNeighbors(int32_t boxDstIdx, int32_t boxSrcIdx)
{
    uint32_t neighborIdx = m_neighborCounts[boxDstIdx];
    uint32_t boxSrcCount = m_neighborCounts[boxSrcIdx];
    RowMatrixXi &neighborMatrix = *m_neighborListMatrix;
    MatrixXb &neighborshipMatrix = *m_neighborshipMatrix;

    // Merge neighbors from source to destination
    for (uint32_t boxSrcNbIdx = 0U; boxSrcNbIdx < boxSrcCount; ++boxSrcNbIdx)
    {
        int32_t newNbIdx = neighborMatrix(boxSrcIdx, boxSrcNbIdx);
        if (neighborshipMatrix(boxDstIdx, newNbIdx) == 0)
        {
            neighborMatrix(boxDstIdx, neighborIdx) = newNbIdx;
            neighborshipMatrix(boxDstIdx, newNbIdx) = 1;
            neighborIdx++;
        }
    }

    // Update neighbor count
    m_neighborCounts[boxDstIdx] = neighborIdx;
}

__attribute__ ((visibility ("default")))
NvDsInferDBScanHandle
NvDsInferDBScanCreate()
{
    return new NvDsInferDBScan();
}

__attribute__ ((visibility ("default")))
void
NvDsInferDBScanDestroy(NvDsInferDBScanHandle handle)
{
    delete handle;
}

__attribute__ ((visibility ("default")))
void
NvDsInferDBScanCluster(NvDsInferDBScanHandle handle,
        NvDsInferDBScanClusteringParams *params, NvDsInferObjectDetectionInfo *objects,
        size_t *numObjects)
{
    handle->clusterObjects(objects, *numObjects, params, false);
}

__attribute__ ((visibility ("default")))
void
NvDsInferDBScanClusterHybrid(NvDsInferDBScanHandle handle,
        NvDsInferDBScanClusteringParams *params, NvDsInferObjectDetectionInfo *objects,
        size_t *numObjects)
{
    handle->clusterObjects(objects, *numObjects, params, true);
}