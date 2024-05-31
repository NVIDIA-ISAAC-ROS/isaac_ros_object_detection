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

#ifndef DW_CORE_EIGENDEFS_HPP__
#define DW_CORE_EIGENDEFS_HPP__

#ifdef Success
    #undef Success
#endif

#include <cstdint>

#include <Eigen/Core>



/////////////////////////////////////////////////////////////////////////////////////////////////
// Alignment issues:
// There is an outstanding bug with Eigen's alignment requirement. The classes of the sfm
// module contain Eigen matrices as members. They seem to be aligned but the compiler sometimes
// returns the wrong address for the member. The ReconstructorTests unit tests fail when alignment
// is enabled in linux and PX. The main problem was observed with Matrix34f but it probably
// affects all Eigen types. The bug was observed in .cu files suggesting it is a problem
// between Eigen and nvcc.
//
// All alignment is now disabled pending further investigation.
/////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T, int Rows, int Cols>
using Matrix = Eigen::Matrix<T, Rows, Cols, Eigen::DontAlign>;

template<typename T, int Rows>
using Vector = Eigen::Matrix<T, Rows, 1, Eigen::DontAlign>;

template<typename T, int Cols>
using RowVector = Eigen::Matrix<T, 1, Cols, Eigen::RowMajor | Eigen::DontAlign>;

template<typename T, int Rows, int Cols>
using UnalignedMatrix = Eigen::Matrix<T, Rows, Cols, Eigen::DontAlign>;

template<typename T, int Rows>
using UnalignedVector = Eigen::Matrix<T, Rows, 1, Eigen::DontAlign>;

template<typename T, int Cols>
using UnalignedRowVector = Eigen::Matrix<T, 1, Cols, Eigen::RowMajor | Eigen::DontAlign>;

// clang-format off
#define EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, Size, SizeSuffix)                        \
    /** \ingroup matrixtypedefs */                                                     \
                                                                                       \
    typedef UnalignedMatrix<Type, Size, Size> UnalignedMatrix##SizeSuffix##TypeSuffix; \
    /** \ingroup matrixtypedefs */                                                     \
                                                                                       \
    typedef Matrix<Type, Size, Size> Matrix##SizeSuffix##TypeSuffix;                   \
    /** \ingroup matrixtypedefs */                                                     \
                                                                                       \
    typedef UnalignedVector<Type, Size> UnalignedVector##SizeSuffix##TypeSuffix;       \
    /** \ingroup matrixtypedefs */                                                     \
                                                                                       \
    typedef Vector<Type, Size> Vector##SizeSuffix##TypeSuffix;                         \
    /** \ingroup matrixtypedefs */                                                     \
                                                                                       \
    typedef UnalignedRowVector<Type, Size> UnalignedRowVector##SizeSuffix##TypeSuffix; \
    /** \ingroup matrixtypedefs */                                                     \
                                                                                       \
    typedef RowVector<Type, Size> RowVector##SizeSuffix##TypeSuffix;

#define EIGEN_MAKE_DYNAMIC_TYPEDEFS(Type, TypeSuffix, Size)                        \
    /** \ingroup matrixtypedefs */                                                 \
                                                                                   \
    typedef Eigen::Matrix<Type, Size, Eigen::Dynamic> Matrix##Size##X##TypeSuffix; \
    /** \ingroup matrixtypedefs */                                                 \
                                                                                   \
    typedef Eigen::Matrix<Type, Eigen::Dynamic, Size> Matrix##X##Size##TypeSuffix;

#define EIGEN_MAKE_TYPEDEFS_ALL_SIZES(Type, TypeSuffix)      \
                                                             \
    EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 2, 2)              \
                                                             \
    EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 3, 3)              \
                                                             \
    EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 4, 4)              \
                                                             \
    EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 5, 5)              \
                                                             \
    EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 6, 6)              \
                                                             \
    EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, Eigen::Dynamic, X) \
                                                             \
    EIGEN_MAKE_DYNAMIC_TYPEDEFS(Type, TypeSuffix, 2)           \
                                                             \
    EIGEN_MAKE_DYNAMIC_TYPEDEFS(Type, TypeSuffix, 3)           \
                                                             \
    EIGEN_MAKE_DYNAMIC_TYPEDEFS(Type, TypeSuffix, 4)
// clang-format on

EIGEN_MAKE_TYPEDEFS_ALL_SIZES(uint8_t, ub)
EIGEN_MAKE_TYPEDEFS_ALL_SIZES(int8_t, b)
EIGEN_MAKE_TYPEDEFS_ALL_SIZES(uint16_t, us)
EIGEN_MAKE_TYPEDEFS_ALL_SIZES(int16_t, s)
EIGEN_MAKE_TYPEDEFS_ALL_SIZES(uint32_t, ui)
EIGEN_MAKE_TYPEDEFS_ALL_SIZES(int32_t, i)
EIGEN_MAKE_TYPEDEFS_ALL_SIZES(float, f)
EIGEN_MAKE_TYPEDEFS_ALL_SIZES(double, d)

#undef EIGEN_MAKE_TYPEDEFS_ALL_SIZES
#undef EIGEN_MAKE_FIXED_TYPEDEFS
#undef EIGEN_MAKE_DYNAMIC_TYPEDEFS

#undef ALIGN

//////////////////////////////////////////////
// Array used for images

template<typename T>
struct EigenImage
{
    typedef Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> ArrayX;
    typedef Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ArrayXr;
};

template<>
struct EigenImage<float>
{
    typedef Eigen::Array<uint16_t, Eigen::Dynamic, Eigen::Dynamic> ArrayX;
    typedef Eigen::Array<uint16_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ArrayXr;
};


typedef EigenImage<uint8_t>::ArrayX ArrayXb;
typedef EigenImage<uint8_t>::ArrayXr ArrayXbr;


//////////////////////////////////////////////
// saturated_cast

template <typename Tout, typename Tin>
Tout saturated_cast(Tin value)
{
    if (value < Tin(std::numeric_limits<Tout>::min()))
        return std::numeric_limits<Tout>::min();
    else if (value >= Tin(std::numeric_limits<Tout>::max()))
        return std::numeric_limits<Tout>::max();
    else
        return static_cast<Tout>(value);
}

//////////////////////////////////////////////
// Quaternion
typedef Eigen::Quaternion<float, Eigen::DontAlign> Quaternionf;

//////////////////////////////////////////////
// Geometry
template<typename T, uint32_t SpaceDim>
using Hyperplane = Eigen::Hyperplane<T, SpaceDim>;

typedef Hyperplane<float, 2> Hyperplane2f;
typedef Hyperplane<float, 3> Hyperplane3f;

template<typename T, uint32_t SpaceDim>
using ParametrizedLine = Eigen::ParametrizedLine<T, SpaceDim>;

typedef ParametrizedLine<float, 2> ParametrizedLine2f;
typedef ParametrizedLine<float, 3> ParametrizedLine3f;

#endif // DW_CORE_EIGENDEFS_HPP__
