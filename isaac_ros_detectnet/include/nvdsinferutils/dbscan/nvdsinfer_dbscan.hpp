/////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////


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
#include <eigen3/Eigen/Dense>
#pragma GCC diagnostic pop
#include "EigenDefs.hpp"

#include "nvdsinferutils/include/nvdsinfer_dbscan.h"

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
