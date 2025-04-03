/**
 * @file spline2.inl
 * @author Jinyang Liu, Shixun Wu, Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-05-15
 *
 * (copyright to be updated)
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

 #ifndef CUSZ_KERNEL_SPLINE3_CUH
 #define CUSZ_KERNEL_SPLINE3_CUH
 
 #include <stdint.h>
 #include <stdio.h>
 
 #include <type_traits>
 
 #include "cusz/type.h"
 #include "utils/err.hh"
 #include "utils/timer.hh"
 
 #define SPLINE3_COMPR true
 #define SPLINE3_DECOMPR false
 
 #if __cplusplus >= 201703L
 #define CONSTEXPR constexpr
 #else
 #define CONSTEXPR
 #endif
 
 #define TIX threadIdx.x
 #define TIY threadIdx.y
 #define TIZ threadIdx.z
 #define BIX blockIdx.x
 #define BIY blockIdx.y
 #define BIZ blockIdx.z
 #define BDX blockDim.x
 #define BDY blockDim.y
 #define BDZ blockDim.z
 #define GDX gridDim.x
 #define GDY gridDim.y
 #define GDZ gridDim.z
 
 using DIM = u4;
 using STRIDE = u4;
 using DIM3 = dim3;
 using STRIDE3 = dim3;
 
 constexpr int DEFAULT_LINEAR_BLOCK_SIZE = 384;
 
 #define SHM_ERROR s_ectrl
 
 namespace cusz {
 
 /********************************************************************************
  * host API
  ********************************************************************************/
 
 template <
     typename TITER, typename EITER, typename FP = float,
     int SPLINE_DIM = 2,
     int AnchorBlockSizeX = 8, int AnchorBlockSizeY = 8,
     int AnchorBlockSizeZ = 1,
     int numAnchorBlockX = 4,  // Number of Anchor blocks along X
     int numAnchorBlockY = 1,  // Number of Anchor blocks along Y
     int numAnchorBlockZ = 1,  // Number of Anchor blocks along Z
     int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE,
     typename CompactVal = TITER, typename CompactIdx = uint32_t*,
     typename CompactNum = uint32_t*>
 __global__ void c_spline_infprecis_data(
     TITER data, DIM3 data_size, STRIDE3 data_leap, EITER ectrl,
     DIM3 ectrl_size, STRIDE3 ectrl_leap, TITER anchor, STRIDE3 anchor_leap,
     CompactVal cval, CompactIdx cidx, CompactNum cn, FP eb_r, FP ebx2,
     int radius, INTERPOLATION_PARAMS intp_param, TITER errors);
 
 template <
     typename EITER, typename TITER, typename FP = float,
     int SPLINE_DIM = 2, 
     int AnchorBlockSizeX = 8, int AnchorBlockSizeY = 8,
     int AnchorBlockSizeZ = 1,
     int numAnchorBlockX = 4,  // Number of Anchor blocks along X
     int numAnchorBlockY = 1,  // Number of Anchor blocks along Y
     int numAnchorBlockZ = 1,  // Number of Anchor blocks along Z
     int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
 __global__ void x_spline_infprecis_data(
     EITER ectrl,          // input 1
     DIM3 ectrl_size,      //
     STRIDE3 ectrl_leap,   //
     TITER anchor,         // input 2
     DIM3 anchor_size,     //
     STRIDE3 anchor_leap,  //
     TITER data,           // output
     DIM3 data_size,       //
     STRIDE3 data_leap,    //
     FP eb_r, FP ebx2, int radius, INTERPOLATION_PARAMS intp_param);
 
 namespace device_api {
 /********************************************************************************
  * device API
  ********************************************************************************/
 
 template <
     typename T1, typename T2, typename FP, int SPLINE_DIM = 2, int AnchorBlockSizeX = 8,
     int AnchorBlockSizeY = 8, int AnchorBlockSizeZ = 8,
     int numAnchorBlockX = 4,  // Number of Anchor blocks along X
     int numAnchorBlockY = 1,  // Number of Anchor blocks along Y
     int numAnchorBlockZ = 1,  // Number of Anchor blocks along Z
     int LINEAR_BLOCK_SIZE, bool WORKFLOW = SPLINE3_COMPR,
     bool PROBE_PRED_ERROR = false>
 __device__ void spline_layout_interpolate(
     volatile T1 s_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
                       [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
                       [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
     volatile T2 s_ectrl[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
                        [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
                        [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
     DIM3 data_size, FP eb_r, FP ebx2, int radius,
     INTERPOLATION_PARAMS intp_param);
 }  // namespace device_api
 
 }  // namespace cusz
 
 /********************************************************************************
  * helper function
  ********************************************************************************/
 
 namespace {
 
 template <int SPLINE_DIM,
     int AnchorBlockSizeX, int AnchorBlockSizeY, int AnchorBlockSizeZ,
     int numAnchorBlockX,  // Number of Anchor blocks along X
     int numAnchorBlockY,  // Number of Anchor blocks along Y
     int numAnchorBlockZ,  // Number of Anchor blocks along Z
     bool INCLUSIVE = true>
 __forceinline__ __device__ bool xyz_predicate(
     unsigned int x, unsigned int y, unsigned int z, const DIM3& data_size)
 {
   if CONSTEXPR (INCLUSIVE) {
     return (x <= (AnchorBlockSizeX * numAnchorBlockX) and
             y <= (AnchorBlockSizeY * numAnchorBlockY) and
             z <= (AnchorBlockSizeZ * numAnchorBlockZ)) and
            BIX * (AnchorBlockSizeX * numAnchorBlockX) + x < data_size.x and
            BIY * (AnchorBlockSizeY * numAnchorBlockY) + y < data_size.y and
            BIZ * (AnchorBlockSizeZ * numAnchorBlockZ) + z < data_size.z;
   }
   else {
     return x < (AnchorBlockSizeX * numAnchorBlockX) + (BIX == GDX - 1) * (SPLINE_DIM <= 1) and
            y < (AnchorBlockSizeY * numAnchorBlockY) + (BIY == GDY - 1) * (SPLINE_DIM <= 2) and
            z < (AnchorBlockSizeZ * numAnchorBlockZ) + (BIZ == GDZ - 1) * (SPLINE_DIM <= 3) and
            BIX * (AnchorBlockSizeX * numAnchorBlockX) + x < data_size.x and
            BIY * (AnchorBlockSizeY * numAnchorBlockY) + y < data_size.y and
            BIZ * (AnchorBlockSizeZ * numAnchorBlockZ) + z < data_size.z;
   }
 }
 
 template <
     typename T1, typename T2, int SPLINE_DIM, int AnchorBlockSizeX, int AnchorBlockSizeY,
     int AnchorBlockSizeZ,
     int numAnchorBlockX,  // Number of Anchor blocks along X
     int numAnchorBlockY,  // Number of Anchor blocks along Y
     int numAnchorBlockZ,  // Number of Anchor blocks along Z
     int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
 __device__ void c_reset_scratch_data(
     volatile T1 s_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
                       [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
                       [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
     volatile T2 s_ectrl[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
                        [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
                        [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
     int radius)
 {
   // alternatively, reinterprete cast volatile T?[][][] to 1D
   for (auto _tix = TIX; _tix < (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)) *
                                    (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)) * (AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3));
        _tix += LINEAR_BLOCK_SIZE) {
     auto x = (_tix % (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)));
     auto y = (_tix / (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1))) %
              (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2));
     auto z = (_tix / (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1))) /
              (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2));
 
     s_data[z][y][x] = 0;
     /*****************************************************************************
      okay to use
      ******************************************************************************/
     if (x % AnchorBlockSizeX == 0 and y % AnchorBlockSizeY == 0 and
         z % AnchorBlockSizeZ == 0)
       s_ectrl[z][y][x] = radius;
     /*****************************************************************************
      alternatively
      ******************************************************************************/
     // s_ectrl[z][y][x] = radius;
   }
   __syncthreads();
 }
 
 template <
     typename T1, int AnchorBlockSizeX, int AnchorBlockSizeY,
     int AnchorBlockSizeZ,
     int numAnchorBlockX,  // Number of Anchor blocks along X
     int numAnchorBlockY,  // Number of Anchor blocks along Y
     int numAnchorBlockZ,  // Number of Anchor blocks along Z
     int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
 __device__ void c_gather_anchor(
     T1* data, DIM3 data_size, STRIDE3 data_leap, T1* anchor,
     STRIDE3 anchor_leap)
 {
   auto x = (TIX % (AnchorBlockSizeX * numAnchorBlockX)) +
            BIX * (AnchorBlockSizeX * numAnchorBlockX);
   auto y = (TIX / (AnchorBlockSizeX * numAnchorBlockX)) %
                (AnchorBlockSizeY * numAnchorBlockY) +
            BIY * (AnchorBlockSizeY * numAnchorBlockY);
   auto z = (TIX / (AnchorBlockSizeX * numAnchorBlockX)) /
                (AnchorBlockSizeY * numAnchorBlockY) +
            BIZ * (AnchorBlockSizeZ * numAnchorBlockZ);
 
   bool pred1 = x % AnchorBlockSizeX == 0 and y % AnchorBlockSizeY == 0 and
                z % AnchorBlockSizeZ == 0;
   bool pred2 = x < data_size.x and y < data_size.y and z < data_size.z;
 
   if (pred1 and pred2) {
     auto data_id = x + y * data_leap.y + z * data_leap.z;
     auto anchor_id = (x / AnchorBlockSizeX) +
                      (y / AnchorBlockSizeY) * anchor_leap.y +
                      (z / AnchorBlockSizeZ) * anchor_leap.z;
     anchor[anchor_id] = data[data_id];
   }
   __syncthreads();
 }
 
 template <
     typename T1, typename T2 = T1, int SPLINE_DIM = 2, int AnchorBlockSizeX = 8,
     int AnchorBlockSizeY = 8, int AnchorBlockSizeZ = 8,
     int numAnchorBlockX = 4,  // Number of Anchor blocks along X
     int numAnchorBlockY = 1,  // Number of Anchor blocks along Y
     int numAnchorBlockZ = 1,  // Number of Anchor blocks along Z
     int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
 __device__ void x_reset_scratch_data(
     volatile T1 s_xdata[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
                        [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
                        [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
     volatile T2 s_ectrl[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
                        [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
                        [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
     T1* anchor, DIM3 anchor_size, STRIDE3 anchor_leap)
 {
   for (auto _tix = TIX; _tix < (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)) * (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)) * (AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3));
        _tix += LINEAR_BLOCK_SIZE) {
     auto x = (_tix % (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)));
     auto y = (_tix / (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1))) %
              (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2));
     auto z = (_tix / (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1))) /
              (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2));
 
     s_ectrl[z][y][x] = 0;  // TODO explicitly handle zero-padding
     /*****************************************************************************
      okay to use
      ******************************************************************************/
     // Todo 2d
     // Here 8 is the interpolation block size, not the entire compression
     // manipulated by a threadblock, need to dinstiguish with CompressionBlock.
     if (x % AnchorBlockSizeX == 0 and y % AnchorBlockSizeY == 0 and
         z % AnchorBlockSizeZ == 0) {
       s_xdata[z][y][x] = 0;
 
       auto ax = ((x / AnchorBlockSizeX) + BIX * numAnchorBlockX);
       auto ay = ((y / AnchorBlockSizeY) + BIY * numAnchorBlockY);
       auto az = ((z / AnchorBlockSizeZ) + BIZ * numAnchorBlockZ);
 
       if (ax < anchor_size.x and ay < anchor_size.y and az < anchor_size.z)
         s_xdata[z][y][x] =
             anchor[ax + ay * anchor_leap.y + az * anchor_leap.z];
     }
     /*****************************************************************************
      alternatively
      ******************************************************************************/
   }
 
   __syncthreads();
 }
 
 template <
     typename T1, typename T2, int SPLINE_DIM = 2, int AnchorBlockSizeX = 8,
     int AnchorBlockSizeY = 8, int AnchorBlockSizeZ = 8,
     int numAnchorBlockX = 4,  // Number of Anchor blocks along X
     int numAnchorBlockY = 1,  // Number of Anchor blocks along Y
     int numAnchorBlockZ = 1,  // Number of Anchor blocks along Z
     int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
 __device__ void global2shmem_data(
     T1* data, DIM3 data_size, STRIDE3 data_leap,
     volatile T2 s_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
                       [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
                       [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)])
 {
   constexpr auto TOTAL = (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)) *
                          (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)) * 
                          (AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3));
   for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
     auto x = (_tix % (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)));
     auto y = (_tix / (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1))) %
              (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2));
     auto z = (_tix / (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1))) /
              (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2));
     auto gx = (x + BIX * (AnchorBlockSizeX * numAnchorBlockX));
     auto gy = (y + BIY * (AnchorBlockSizeY * numAnchorBlockY));
     auto gz = (z + BIZ * (AnchorBlockSizeZ * numAnchorBlockZ));
     auto gid = gx + gy * data_leap.y + gz * data_leap.z;
    
     if (gx < data_size.x and gy < data_size.y and gz < data_size.z)
       s_data[z][y][x] = data[gid];
   }
   __syncthreads();
 }
 
 
 template <
     typename T = float, typename E = u4, int SPLINE_DIM = 2, int AnchorBlockSizeX = 8,
     int AnchorBlockSizeY = 8, int AnchorBlockSizeZ = 8,
     int numAnchorBlockX = 4,  // Number of Anchor blocks along X
     int numAnchorBlockY = 1,  // Number of Anchor blocks along Y
     int numAnchorBlockZ = 1,  // Number of Anchor blocks along Z
     int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
 __device__ void global2shmem_fuse(
     E* ectrl, dim3 ectrl_size, dim3 ectrl_leap, T* scattered_outlier,
     volatile T s_ectrl[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
                       [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
                       [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)])
 {
   constexpr auto TOTAL = (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)) *
                          (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)) *
                          (AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3));
    unsigned int level_offsets[5] = { 0, 
          (ectrl_size.x * ectrl_size.y * ectrl_size.z) / (16 * 16 * 16),
          (ectrl_size.x * ectrl_size.y * ectrl_size.z) / (8 * 8 * 8),
          (ectrl_size.x * ectrl_size.y * ectrl_size.z) / (4 * 4 * 4),
          (ectrl_size.x * ectrl_size.y * ectrl_size.z) / (2 * 2 * 2),
      };
 
   for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
     auto x = (_tix % (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)));
     auto y = (_tix / (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1))) %
              (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2));
     auto z = (_tix / (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1))) /
              (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2));
     auto gx = (x + BIX * (AnchorBlockSizeX * numAnchorBlockX));
     auto gy = (y + BIY * (AnchorBlockSizeY * numAnchorBlockY));
     auto gz = (z + BIZ * (AnchorBlockSizeZ * numAnchorBlockZ));
     
     auto gid = gx + gy * ectrl_leap.y + gz * ectrl_leap.z;
    
    //  unsigned int level = 0;
    //  int level_size = AnchorBlockSizeX;
    //  while(level_size > 1){
    //   if ((gx % level_size == 0) && (gy % level_size == 0) && (gz % level_size == 0)) break;
    //   level += 1;
    //   level_size /= 2;
    //  }
    unsigned int level = 0;
    // if ((gx % 16 == 0) && (gy % 16 == 0) && (gz % 16 == 0)) level = 0;
    // else if ((gx % 8 == 0) && (gy % 8 == 0) && (gz % 8 == 0)) level = 1;
    // else if ((gx % 4 == 0) && (gy % 4 == 0) && (gz % 4 == 0)) level = 2;
    // else if ((gx % 2 == 0) && (gy % 2 == 0) && (gz % 2 == 0)) level = 3;
    // else level = 4;
    level = ((gx % 16 == 0) && (gy % 16 == 0) && (gz % 16 == 0)) ? 0 :
    ((gx % 8 == 0) && (gy % 8 == 0) && (gz % 8 == 0)) ? 1 :
    ((gx % 4 == 0) && (gy % 4 == 0) && (gz % 4 == 0)) ? 2 :
    ((gx % 2 == 0) && (gy % 2 == 0) && (gz % 2 == 0)) ? 3 : 4;
     unsigned int shift = 4 - level;

     // Compute level index using integer division
     unsigned int level_index_x = gx >> shift;
     unsigned int level_index_y = gy >> shift;
     unsigned int level_index_z = gz >> shift;

     // Compute level dimensions
     unsigned int level_dim_x = ectrl_size.x >> shift;
     unsigned int level_dim_y = ectrl_size.y >> shift;
     unsigned int level_dim_z = ectrl_size.z >> shift;

     // Compute upper-level dimensions
     unsigned int upper_level_dim_x = ectrl_size.x >> (shift + 1);
     unsigned int upper_level_dim_y = ectrl_size.y >> (shift + 1);
     unsigned int upper_level_dim_z = ectrl_size.z >> (shift + 1);

     // Compute offset
     unsigned int offset = (level == 0) ? 0 : (
     ((level_index_x + 1) / 2) * upper_level_dim_y * upper_level_dim_z +
     ((level_index_y + 1) / 2) * upper_level_dim_z * (1 - (level_index_x % 2)) +
     ((level_index_z + 1) / 2) * (1 - (level_index_x % 2)) * (1 - (level_index_y % 2)));

     // Compute index
     unsigned int index = level_offsets[level] +
         level_index_x * level_dim_y * level_dim_z +
         level_index_y * level_dim_z +
         level_index_z - offset;
      
     if (gx < ectrl_size.x and gy < ectrl_size.y and gz < ectrl_size.z)
       s_ectrl[z][y][x] = static_cast<T>(ectrl[index]) + scattered_outlier[gid];
   }
   __syncthreads();
 }
 
 // dram_outlier should be the same in type with shared memory buf
 template <
     typename T1, typename T2, int SPLINE_DIM = 2, int AnchorBlockSizeX = 8,
     int AnchorBlockSizeY = 8, int AnchorBlockSizeZ = 8,
     int numAnchorBlockX = 4,  // Number of Anchor blocks along X
     int numAnchorBlockY = 1,  // Number of Anchor blocks along Y
     int numAnchorBlockZ = 1,  // Number of Anchor blocks along Z
     int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
 __device__ void shmem2global_data(
     volatile T1 s_buf[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
                      [AnchorBlockSizeY * numAnchorBlockY +  + (SPLINE_DIM >= 2)]
                      [AnchorBlockSizeX * numAnchorBlockX +  + (SPLINE_DIM >= 1)],
     T2* dram_buf, DIM3 buf_size, STRIDE3 buf_leap)
 {
   auto x_size = AnchorBlockSizeX * numAnchorBlockX + (BIX == GDX - 1) * (SPLINE_DIM >= 1);
   auto y_size = AnchorBlockSizeY * numAnchorBlockY + (BIY == GDY - 1) * (SPLINE_DIM >= 2);
   auto z_size = AnchorBlockSizeZ * numAnchorBlockZ + (BIZ == GDZ - 1) * (SPLINE_DIM >= 3);
   auto TOTAL = x_size * y_size * z_size;
 
   for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
     auto x = (_tix % x_size);
     auto y = (_tix / x_size) % y_size;
     auto z = (_tix / x_size) / y_size;
     auto gx = (x + BIX * AnchorBlockSizeX * numAnchorBlockX);
     auto gy = (y + BIY * AnchorBlockSizeY * numAnchorBlockY);
     auto gz = (z + BIZ * AnchorBlockSizeZ * numAnchorBlockZ);
     auto gid = gx + gy * buf_leap.y + gz * buf_leap.z;
 
     if (gx < buf_size.x and gy < buf_size.y and gz < buf_size.z)
       dram_buf[gid] = s_buf[z][y][x];
   }
   __syncthreads();
 }
 
 // dram_outlier should be the same in type with shared memory buf
 template <
     typename T1, typename T2, int SPLINE_DIM = 2, int AnchorBlockSizeX = 8,
     int AnchorBlockSizeY = 8, int AnchorBlockSizeZ = 8,
     int numAnchorBlockX = 4,  // Number of Anchor blocks along X
     int numAnchorBlockY = 1,  // Number of Anchor blocks along Y
     int numAnchorBlockZ = 1,  // Number of Anchor blocks along Z
     int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
 __device__ void shmem2global_data_with_compaction(
     volatile T1 s_buf[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
                      [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
                      [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
     T2* dram_buf, DIM3 buf_size, STRIDE3 buf_leap, int radius,
     T1* dram_compactval = nullptr, uint32_t* dram_compactidx = nullptr,
     uint32_t* dram_compactnum = nullptr)
 {
   auto x_size = AnchorBlockSizeX * numAnchorBlockX + (BIX == GDX - 1) * (SPLINE_DIM >= 1);
   auto y_size = AnchorBlockSizeY * numAnchorBlockY + (BIY == GDY - 1) * (SPLINE_DIM >= 2);
   auto z_size = AnchorBlockSizeZ * numAnchorBlockZ + (BIZ == GDZ - 1) * (SPLINE_DIM >= 3);
   auto TOTAL = x_size * y_size * z_size;
   unsigned int level_offsets[5] = { 0, 
    (buf_size.x * buf_size.y * buf_size.z) / (16 * 16 * 16),
    (buf_size.x * buf_size.y * buf_size.z) / (8 * 8 * 8),
    (buf_size.x * buf_size.y * buf_size.z) / (4 * 4 * 4),
    (buf_size.x * buf_size.y * buf_size.z) / (2 * 2 * 2),
};
 
   for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
     auto x = (_tix % x_size);
     auto y = (_tix / x_size) % y_size;
     auto z = (_tix / x_size) / y_size;
     auto gx = (x + BIX * AnchorBlockSizeX * numAnchorBlockX);
     auto gy = (y + BIY * AnchorBlockSizeY * numAnchorBlockY);
     auto gz = (z + BIZ * AnchorBlockSizeZ * numAnchorBlockZ);
     auto gid = gx + gy * buf_leap.y + gz * buf_leap.z;

     unsigned int level  =0;
    //  if ((gx % 16 == 0) && (gy % 16 == 0) && (gz % 16 == 0)) level = 0;
    //  else if ((gx % 8 == 0) && (gy % 8 == 0) && (gz % 8 == 0)) level = 1;
    //  else if ((gx % 4 == 0) && (gy % 4 == 0) && (gz % 4 == 0)) level = 2;
    //  else if ((gx % 2 == 0) && (gy % 2 == 0) && (gz % 2 == 0)) level = 3;
    //  else level = 4;
    level = ((gx % 16 == 0) && (gy % 16 == 0) && (gz % 16 == 0)) ? 0 :
        ((gx % 8 == 0) && (gy % 8 == 0) && (gz % 8 == 0)) ? 1 :
        ((gx % 4 == 0) && (gy % 4 == 0) && (gz % 4 == 0)) ? 2 :
        ((gx % 2 == 0) && (gy % 2 == 0) && (gz % 2 == 0)) ? 3 : 4;
 
     unsigned int shift = 4 - level;
 
     // Compute level index using integer division
     unsigned int level_index_x = gx >> shift;
     unsigned int level_index_y = gy >> shift;
     unsigned int level_index_z = gz >> shift;
 
     // Compute level dimensions
     unsigned int level_dim_x = buf_size.x >> shift;
     unsigned int level_dim_y = buf_size.y >> shift;
     unsigned int level_dim_z = buf_size.z >> shift;
 
     // Compute upper-level dimensions
     unsigned int upper_level_dim_x = buf_size.x >> (shift + 1);
     unsigned int upper_level_dim_y = buf_size.y >> (shift + 1);
     unsigned int upper_level_dim_z = buf_size.z >> (shift + 1);
 
     // Compute offset
     unsigned int offset = (level == 0) ? 0 : (
     ((level_index_x + 1) / 2) * upper_level_dim_y * upper_level_dim_z +
     ((level_index_y + 1) / 2) * upper_level_dim_z * (1 - (level_index_x % 2)) +
     ((level_index_z + 1) / 2) * (1 - (level_index_x % 2)) * (1 -  (level_index_y % 2)));
 
     // Compute index
     unsigned int index = level_offsets[level] +
         level_index_x * level_dim_y * level_dim_z +
         level_index_y * level_dim_z +
         level_index_z - offset;
 

     auto candidate = s_buf[z][y][x];
     bool quantizable = (candidate >= 0) and (candidate < 2 * radius);
    
     if (gx < buf_size.x and gy < buf_size.y and gz < buf_size.z) {
       // TODO this is for algorithmic demo by reading from shmem
       // For performance purpose, it can be inlined in quantization
       dram_buf[index] = quantizable * static_cast<T2>(candidate);
      //  dram_buf[gid] = quantizable * static_cast<T2>(candidate);
 
       if (not quantizable) {
         auto cur_idx = atomicAdd(dram_compactnum, 1);
         dram_compactidx[cur_idx] = gid;
         dram_compactval[cur_idx] = candidate;
       }
     }
   }
   __syncthreads();
 }
 

 template <
     typename T1,
     typename T2,
     typename FP, int SPLINE_DIM, int AnchorBlockSizeX,
     int AnchorBlockSizeY, int AnchorBlockSizeZ,
     int numAnchorBlockX,  // Number of Anchor blocks along X
     int numAnchorBlockY,  // Number of Anchor blocks along Y
     int numAnchorBlockZ,  // Number of Anchor blocks along Z
     typename LAMBDA,
     bool LINE,
     bool FACE,
     bool CUBE,
     int  LINEAR_BLOCK_SIZE,
     int NUM_ELE,
     bool COARSEN,
     bool BORDER_INCLUSIVE,
     bool WORKFLOW,
     typename INTERP>
 __forceinline__ __device__ void interpolate_stage_md(
     volatile T1 s_data[17][17][17],
     volatile T2 s_ectrl[17][17][17],
     DIM3    data_size,
     LAMBDA xyzmap,
     int         unit,
     FP          eb_r,
     FP          ebx2,
     int         radius,
     INTERP cubic_interpolator)
 {
     static_assert(COARSEN or (NUM_ELE <= 384), "block oversized");
     static_assert((LINE or FACE or CUBE) == true, "must be one hot");
     static_assert((LINE and FACE) == false, "must be only one hot (1)");
     static_assert((LINE and CUBE) == false, "must be only one hot (2)");
     static_assert((FACE and CUBE) == false, "must be only one hot (3)");
 
     auto run = [&](auto x, auto y, auto z) {
 
         
 
         if (xyz_predicate<SPLINE_DIM,
          AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
          numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ,
          BORDER_INCLUSIVE>(x, y, z, data_size)) {
             T1 pred = 0;
             auto global_x=BIX*AnchorBlockSizeX + x, global_y=BIY*AnchorBlockSizeY+y, global_z=BIZ*AnchorBlockSizeZ+z;
            
             if CONSTEXPR (LINE) {  //
                 //bool I_X = x&1; 
                 bool I_Y = (y % (2*unit) )> 0; 
                 bool I_Z = (z % (2*unit) )> 0; 
                 if (I_Z){
                     //assert(x&1==0 and y&1==0);
 
                     if(BIZ!=GDZ-1){
 
                         if(z>=3*unit and z+3*unit<=AnchorBlockSizeZ  )
                             pred = cubic_interpolator(s_data[z - 3*unit][y][x],s_data[z - unit][y][x],s_data[z + unit][y][x],s_data[z + 3*unit][y][x]);
                         else if (z+3*unit<=AnchorBlockSizeZ)
                             pred = (3*s_data[z - unit][y][x] + 6*s_data[z + unit][y][x]-s_data[z + 3*unit][y][x]) / 8;
                         else if (z>=3*unit)
                             pred = (-s_data[z - 3*unit][y][x]+6*s_data[z - unit][y][x] + 3*s_data[z + unit][y][x]) / 8;
 
                         else
                             pred = (s_data[z - unit][y][x] + s_data[z + unit][y][x]) / 2;
                     }
                     else{
                         if(z>=3*unit){
                             if(z+3*unit<=AnchorBlockSizeZ and global_z+3*unit<data_size.z)
                                 pred = cubic_interpolator(s_data[z - 3*unit][y][x],s_data[z - unit][y][x] ,s_data[z + unit][y][x],s_data[z + 3*unit][y][x]);
                             else if (global_z+unit<data_size.z)
                                 pred = (-s_data[z - 3*unit][y][x]+6*s_data[z - unit][y][x] + 3*s_data[z + unit][y][x]) / 8;
                             else
                                 pred=s_data[z - unit][y][x];
 
                         }
                         else{
                             if(z+3*unit<=AnchorBlockSizeZ and global_z+3*unit<data_size.z)
                                 pred = (3*s_data[z - unit][y][x] + 6*s_data[z + unit][y][x]-s_data[z + 3*unit][y][x]) / 8;
                             else if (global_z+unit<data_size.z)
                                 pred = (s_data[z - unit][y][x] + s_data[z + unit][y][x]) / 2;
                             else
                                 pred=s_data[z - unit][y][x];
                         } 
                     }
 
                 }
                 else if (I_Y){
                     //assert(x&1==0 and z&1==0);
                     if(BIY!=GDY-1){
                         if(y>=3*unit and y+3*unit<=AnchorBlockSizeY )
                             pred = cubic_interpolator(s_data[z ][y- 3*unit][x],s_data[z ][y- unit][x] ,s_data[z ][y+ unit][x],s_data[z][y + 3*unit][x]) ;
                         else if (y+3*unit<=AnchorBlockSizeY)
                             pred = (3*s_data[z ][y - unit][x] + 6*s_data[z][y + unit][x]-s_data[z][y + 3*unit][x]) / 8;
                         else if (y>=3*unit)
                             pred = (-s_data[z ][y- 3*unit][x]+6*s_data[z][y - unit][x] + 3*s_data[z][y + unit][x]) / 8;
                         else
                             pred = (s_data[z][y - unit][x] + s_data[z][y + unit][x]) / 2;
                     }
                     else{
                         if(y>=3*unit){
                             if(y+3*unit<=AnchorBlockSizeY and global_y+3*unit<data_size.y)
                                 pred = cubic_interpolator(s_data[z ][y- 3*unit][x],s_data[z][y - unit][x],s_data[z ][y+ unit][x],s_data[z ][y+ 3*unit][x]);
                             else if (global_y+unit<data_size.y)
                                 pred = (-s_data[z ][y- 3*unit][x]+6*s_data[z ][y- unit][x] + 3*s_data[z ][y+ unit][x]) / 8;
                             else
                                 pred=s_data[z ][y- unit][x];
 
                         }
                         else{
                             if(y+3*unit<=AnchorBlockSizeY and global_y+3*unit<data_size.y)
                                 pred = (3*s_data[z][y - unit][x] + 6*s_data[z ][y+ unit][x]-s_data[z][y + 3*unit][x]) / 8;
                             else if (global_y+unit<data_size.y)
                                 pred = (s_data[z ][y- unit][x] + s_data[z][y + unit][x]) / 2;
                             else
                                 pred=s_data[z ][y- unit][x];
                         } 
                     }
                 }
                 else{//I_X
                     //assert(y&1==0 and z&1==0);
                     if(BIX!=GDX-1){
                         if(x>=3*unit and x+3*unit<=AnchorBlockSizeX )
                             pred = cubic_interpolator(s_data[z ][y][x- 3*unit],s_data[z ][y][x- unit],s_data[z ][y][x+ unit],s_data[z ][y][x + 3*unit]);
                         else if (x+3*unit<=AnchorBlockSizeX)
                             pred = (3*s_data[z ][y][x- unit] + 6*s_data[z ][y][x + unit]-s_data[z][y][x + 3*unit]) / 8;
                         else if (x>=3*unit)
                             pred = (-s_data[z][y][x - 3*unit]+6*s_data[z][y][x - unit] + 3*s_data[z ][y][x + unit]) / 8;
                         else
                             pred = (s_data[z][y][x - unit] + s_data[z][y][x + unit]) / 2;
                     }
                     else{
                         if(x>=3*unit){
                             if(x+3*unit<=AnchorBlockSizeX and global_x+3*unit<data_size.x)
                                 pred = cubic_interpolator(s_data[z ][y][x- 3*unit],s_data[z][y ][x- unit],s_data[z ][y][x+ unit],s_data[z ][y][x+ 3*unit]);
                             else if (global_x+unit<data_size.x)
                                 pred = (-s_data[z ][y][x- 3*unit]+6*s_data[z ][y][x- unit] + 3*s_data[z ][y][x+ unit]) / 8;
                             else
                                 pred=s_data[z ][y][x- unit];
 
                         }
                         else{
                             if(x+3*unit<=AnchorBlockSizeX and global_x+3*unit<data_size.x)
                                 pred = (3*s_data[z][y ][x- unit] + 6*s_data[z ][y][x+ unit]-s_data[z][y ][x+ 3*unit]) / 8;
                             else if (global_x+unit<data_size.x)
                                 pred = (s_data[z ][y][x- unit] + s_data[z][y ][x+ unit]) / 2;
                             else
                                 pred=s_data[z ][y][x- unit];
                         } 
                     }
 
                 }
             }
             auto get_interp_order = [&](auto x, auto BI, auto GD, auto gx, auto gs, auto AnchorBlockSize){
                 int b = x >= 3*unit ? 3 : 1;
                 int f = 0;
                 if(x+3*unit<=AnchorBlockSize and (BI != GD-1 or gx+3*unit < gs) )
                     f = 3;
                 else if (BI != GD-1 or gx+unit < gs)
                     f = 1;
                 if (b==3){
                     if(f==3)
                         return 4;
                     else if (f==1)
                         return 3;
                     else
                         return 0;
                 }
                 else{//b==1
                     if(f==3)
                         return 2;
                     else if (f==1)
                         return 1;
                     else
                         return 0;
                 }
             };
             if CONSTEXPR (FACE) {  //
 
                 bool I_YZ = (x % (2*unit) ) == 0;
                 bool I_XZ = (y % (2*unit ) )== 0;

                  
                 if (I_YZ){
 
 
                     auto interp_z = get_interp_order(z,BIZ,GDZ,global_z,data_size.z, AnchorBlockSizeZ);
                     auto interp_y = get_interp_order(y,BIY,GDY,global_y,data_size.y, AnchorBlockSizeY);
 
                     if(interp_z==4){
                         if(interp_y==4){
                             pred = (cubic_interpolator(s_data[z - 3*unit][y][x],s_data[z - unit][y][x],s_data[z + unit][y][x],s_data[z + 3*unit][y][x])+
                                     cubic_interpolator(s_data[z ][y- 3*unit][x],s_data[z ][y- unit][x] ,s_data[z ][y+ unit][x],s_data[z][y + 3*unit][x]) ) / 2;
                         }
                         else
                             pred = cubic_interpolator(s_data[z - 3*unit][y][x],s_data[z - unit][y][x],s_data[z + unit][y][x],s_data[z + 3*unit][y][x]);
 
                     }
                     else if (interp_z == 3){
                         if(interp_y==4)
                             pred = cubic_interpolator(s_data[z ][y- 3*unit][x],s_data[z ][y- unit][x] ,s_data[z ][y+ unit][x],s_data[z][y + 3*unit][x]);
                         else if (interp_y == 3)
                             pred = (-s_data[z - 3*unit][y][x]+6*s_data[z - unit][y][x] + 3*s_data[z + unit][y][x] - s_data[z ][y- 3*unit][x]+6*s_data[z ][y- unit][x] + 3*s_data[z ][y+ unit][x]) / 16;
                         else if (interp_y == 2)
                             pred = (-s_data[z - 3*unit][y][x]+6*s_data[z - unit][y][x] + 3*s_data[z + unit][y][x] + 3*s_data[z ][y - unit][x] + 6*s_data[z][y + unit][x]-s_data[z][y + 3*unit][x]) / 16;
                         else
                             pred = (-s_data[z - 3*unit][y][x]+6*s_data[z - unit][y][x] + 3*s_data[z + unit][y][x]) / 8;
 
                     }
 
                     else if (interp_z == 2){
                         if(interp_y==4)
                             pred = cubic_interpolator(s_data[z ][y- 3*unit][x],s_data[z ][y- unit][x] ,s_data[z ][y+ unit][x],s_data[z][y + 3*unit][x]);
                         else if (interp_y == 3)
                             pred = (3*s_data[z - unit][y][x] + 6*s_data[z + unit][y][x]-s_data[z + 3*unit][y][x] - s_data[z ][y- 3*unit][x]+6*s_data[z ][y- unit][x] + 3*s_data[z ][y+ unit][x]) / 16;
                         else if (interp_y == 2)
                             pred = (3*s_data[z - unit][y][x] + 6*s_data[z + unit][y][x]-s_data[z + 3*unit][y][x] + 3*s_data[z ][y - unit][x] + 6*s_data[z][y + unit][x]-s_data[z][y + 3*unit][x]) / 16;
                         else
                             pred = (3*s_data[z - unit][y][x] + 6*s_data[z + unit][y][x]-s_data[z + 3*unit][y][x]) / 8;
 
                     }
                     else if (interp_z == 1){
                         if(interp_y == 4)
                             pred = cubic_interpolator(s_data[z ][y- 3*unit][x],s_data[z ][y- unit][x] ,s_data[z ][y+ unit][x],s_data[z][y + 3*unit][x]);
                         else if (interp_y == 3)
                             pred = (-s_data[z ][y- 3*unit][x] + 6*s_data[z][y - unit][x] + 3*s_data[z][y + unit][x]) / 8;
                         else if (interp_y == 2)
                             pred = (3*s_data[z][y - unit][x] + 6*s_data[z ][y+ unit][x]-s_data[z][y + 3*unit][x]) / 8;
                         else if (interp_y == 1)
                             pred = (s_data[z - unit][y][x] + s_data[z + unit][y][x] + s_data[z ][y - unit][x] + s_data[z][y + unit][x]) / 4;
                         else 
                             pred = (s_data[z - unit][y][x] + s_data[z + unit][y][x]) / 2;
                     }
                     else{
                         if(interp_y == 4)
                             pred = cubic_interpolator(s_data[z ][y- 3*unit][x],s_data[z ][y- unit][x] ,s_data[z ][y+ unit][x],s_data[z][y + 3*unit][x]);
                         else if (interp_y == 3)
                             pred = (-s_data[z ][y- 3*unit][x] + 6*s_data[z][y - unit][x] + 3*s_data[z][y + unit][x]) / 8;
                         else if (interp_y == 2)
                             pred = (3*s_data[z][y - unit][x] + 6*s_data[z ][y+ unit][x]-s_data[z][y + 3*unit][x]) / 8;
                         else if (interp_y == 1)
                             pred = (s_data[z ][y - unit][x] + s_data[z][y + unit][x]) / 2;
                         else 
                             pred = (s_data[z - unit][y][x] + s_data[z ][y - unit][x] - s_data[z - unit][y - unit][x]);
 
                     }
 
                 }
                 else if (I_XZ){
                     auto interp_z = get_interp_order(z,BIZ,GDZ,global_z,data_size.z, AnchorBlockSizeZ);
                     auto interp_x = get_interp_order(x,BIX,GDX,global_x,data_size.x, AnchorBlockSizeX);
 
                     //if(BIX == 10 and BIY == 12 and BIZ == 0 and x==13 and y==6 and z==9)
                     //printf("ixz %d %d\n", interp_x,interp_z);
 
                     if(interp_z==4){
                         if(interp_x==4){
                             pred = (cubic_interpolator(s_data[z - 3*unit][y][x],
                                                          s_data[z - unit][y][x],
                                                          s_data[z + unit][y][x],
                                                          s_data[z + 3*unit][y][x]) +
                                     cubic_interpolator(s_data[z][y][x - 3*unit],
                                                          s_data[z][y][x - unit],
                                                          s_data[z][y][x + unit],
                                                          s_data[z][y][x + 3*unit])
                                    ) / 2;
                         }
                         else
                             pred = cubic_interpolator(s_data[z - 3*unit][y][x],
                                                       s_data[z - unit][y][x],
                                                       s_data[z + unit][y][x],
                                                       s_data[z + 3*unit][y][x]);
 
                     }
                     else if (interp_z == 3){
                         if(interp_x==4)
                             pred = cubic_interpolator(s_data[z][y][x - 3*unit],
                                                       s_data[z][y][x - unit],
                                                       s_data[z][y][x + unit],
                                                       s_data[z][y][x + 3*unit]);
                         else if (interp_x == 3)
                             pred = (-s_data[z - 3*unit][y][x] + 6*s_data[z - unit][y][x] + 3*s_data[z + unit][y][x]
                                     - s_data[z][y][x - 3*unit] + 6*s_data[z][y][x - unit] + 3*s_data[z][y][x + unit]) / 16;
                         else if (interp_x == 2)
                             pred = (-s_data[z - 3*unit][y][x] + 6*s_data[z - unit][y][x] + 3*s_data[z + unit][y][x]
                                     + 3*s_data[z][y][x - unit] + 6*s_data[z][y][x + unit] - s_data[z][y][x + 3*unit]) / 16;
                         else
                             pred = (-s_data[z - 3*unit][y][x] + 6*s_data[z - unit][y][x] + 3*s_data[z + unit][y][x]) / 8;
 
                     }
                     else if (interp_z == 2){
                         if(interp_x==4)
                             pred = cubic_interpolator(s_data[z][y][x - 3*unit],
                                                       s_data[z][y][x - unit],
                                                       s_data[z][y][x + unit],
                                                       s_data[z][y][x + 3*unit]);
                         else if (interp_x == 3)
                             pred = (3*s_data[z - unit][y][x] + 6*s_data[z + unit][y][x] - s_data[z + 3*unit][y][x]
                                     - s_data[z][y][x - 3*unit] + 6*s_data[z][y][x - unit] + 3*s_data[z][y][x + unit]) / 16;
                         else if (interp_x == 2)
                             pred = (3*s_data[z - unit][y][x] + 6*s_data[z + unit][y][x] - s_data[z + 3*unit][y][x]
                                     + 3*s_data[z][y][x - unit] + 6*s_data[z][y][x + unit] - s_data[z][y][x + 3*unit]) / 16;
                         else
                             pred = (3*s_data[z - unit][y][x] + 6*s_data[z + unit][y][x] - s_data[z + 3*unit][y][x]) / 8;
 
                     }
                     else if (interp_z == 1){
                         if(interp_x == 4)
                             pred = cubic_interpolator(s_data[z][y][x - 3*unit],
                                                       s_data[z][y][x - unit],
                                                       s_data[z][y][x + unit],
                                                       s_data[z][y][x + 3*unit]);
                         else if (interp_x == 3)
                             pred = (-s_data[z][y][x - 3*unit] + 6*s_data[z][y][x - unit] + 3*s_data[z][y][x + unit]) / 8;
                         else if (interp_x == 2)
                             pred = (3*s_data[z][y][x - unit] + 6*s_data[z][y][x + unit] - s_data[z][y][x + 3*unit]) / 8;
                         else if (interp_x == 1)
                             pred = (s_data[z - unit][y][x] + s_data[z + unit][y][x]
                                     + s_data[z][y][x - unit] + s_data[z][y][x + unit]) / 4;
                         else 
                             pred = (s_data[z - unit][y][x] + s_data[z + unit][y][x]) / 2;
                     }
                     else{
                         if(interp_x == 4)
                             pred = cubic_interpolator(s_data[z][y][x - 3*unit],
                                                       s_data[z][y][x - unit],
                                                       s_data[z][y][x + unit],
                                                       s_data[z][y][x + 3*unit]);
                         else if (interp_x == 3)
                             pred = (-s_data[z][y][x - 3*unit] + 6*s_data[z][y][x - unit] + 3*s_data[z][y][x + unit]) / 8;
                         else if (interp_x == 2)
                             pred = (3*s_data[z][y][x - unit] + 6*s_data[z][y][x + unit] - s_data[z][y][x + 3*unit]) / 8;
                         else if (interp_x == 1)
                             pred = (s_data[z][y][x - unit] + s_data[z][y][x + unit]) / 2;
                         else 
                             pred = (s_data[z - unit][y][x] + s_data[z][y][x - unit] - s_data[z - unit][y][x - unit]);
                     }
 
                 }
                 else{//I_XY
                     //assert(z&1==0);
 
                     auto interp_y = get_interp_order(y,BIY,GDY,global_y,data_size.y,  AnchorBlockSizeY);
                     auto interp_x = get_interp_order(x,BIX,GDX,global_x,data_size.x,  AnchorBlockSizeX);
 
                     if(interp_y==4){
                         if(interp_x==4){
                             pred = (cubic_interpolator(s_data[z][y - 3*unit][x],
                                                          s_data[z][y - unit][x],
                                                          s_data[z][y + unit][x],
                                                          s_data[z][y + 3*unit][x]) +
                                     cubic_interpolator(s_data[z][y][x - 3*unit],
                                                          s_data[z][y][x - unit],
                                                          s_data[z][y][x + unit],
                                                          s_data[z][y][x + 3*unit])
                                    ) / 2;
                         }
                         else
                             pred = cubic_interpolator(s_data[z][y - 3*unit][x],
                                                       s_data[z][y - unit][x],
                                                       s_data[z][y + unit][x],
                                                       s_data[z][y + 3*unit][x]);
                     }
                     else if (interp_y == 3){
                         if(interp_x==4)
                             pred = cubic_interpolator(s_data[z][y][x - 3*unit],
                                                       s_data[z][y][x - unit],
                                                       s_data[z][y][x + unit],
                                                       s_data[z][y][x + 3*unit]);
                         else if (interp_x == 3)
                             pred = (-s_data[z][y - 3*unit][x] + 6*s_data[z][y - unit][x] + 3*s_data[z][y + unit][x]
                                     - s_data[z][y][x - 3*unit] + 6*s_data[z][y][x - unit] + 3*s_data[z][y][x + unit]) / 16;
                         else if (interp_x == 2)
                             pred = (-s_data[z][y - 3*unit][x] + 6*s_data[z][y - unit][x] + 3*s_data[z][y + unit][x]
                                     + 3*s_data[z][y][x - unit] + 6*s_data[z][y][x + unit] - s_data[z][y][x + 3*unit]) / 16;
                         else
                             pred = (-s_data[z][y - 3*unit][x] + 6*s_data[z][y - unit][x] + 3*s_data[z][y + unit][x]) / 8;
                     }
                     else if (interp_y == 2){
                         if(interp_x==4)
                             pred = cubic_interpolator(s_data[z][y][x - 3*unit],
                                                       s_data[z][y][x - unit],
                                                       s_data[z][y][x + unit],
                                                       s_data[z][y][x + 3*unit]);
                         else if (interp_x == 3)
                             pred = (3*s_data[z][y - unit][x] + 6*s_data[z][y + unit][x] - s_data[z][y + 3*unit][x]
                                     - s_data[z][y][x - 3*unit] + 6*s_data[z][y][x - unit] + 3*s_data[z][y][x + unit]) / 16;
                         else if (interp_x == 2)
                             pred = (3*s_data[z][y - unit][x] + 6*s_data[z][y + unit][x] - s_data[z][y + 3*unit][x]
                                     + 3*s_data[z][y][x - unit] + 6*s_data[z][y][x + unit] - s_data[z][y][x + 3*unit]) / 16;
                         else
                             pred = (3*s_data[z][y - unit][x] + 6*s_data[z][y + unit][x] - s_data[z][y + 3*unit][x]) / 8;
                     }
                     else if (interp_y == 1){
                         if(interp_x == 4)
                             pred = cubic_interpolator(s_data[z][y][x - 3*unit],
                                                       s_data[z][y][x - unit],
                                                       s_data[z][y][x + unit],
                                                       s_data[z][y][x + 3*unit]);
                         else if (interp_x == 3)
                             pred = (-s_data[z][y][x - 3*unit] + 6*s_data[z][y][x - unit] + 3*s_data[z][y][x + unit]) / 8;
                         else if (interp_x == 2)
                             pred = (3*s_data[z][y][x - unit] + 6*s_data[z][y][x + unit] - s_data[z][y][x + 3*unit]) / 8;
                         else if (interp_x == 1)
                             pred = (s_data[z][y - unit][x] + s_data[z][y + unit][x]
                                     + s_data[z][y][x - unit] + s_data[z][y][x + unit]) / 4;
                         else 
                             pred = (s_data[z][y - unit][x] + s_data[z][y + unit][x]) / 2;
                     }
                     else{
                         if(interp_x == 4)
                             pred = cubic_interpolator(s_data[z][y][x - 3*unit],
                                                       s_data[z][y][x - unit],
                                                       s_data[z][y][x + unit],
                                                       s_data[z][y][x + 3*unit]);
                         else if (interp_x == 3)
                             pred = (-s_data[z][y][x - 3*unit] + 6*s_data[z][y][x - unit] + 3*s_data[z][y][x + unit]) / 8;
                         else if (interp_x == 2)
                             pred = (3*s_data[z][y][x - unit] + 6*s_data[z][y][x + unit] - s_data[z][y][x + 3*unit]) / 8;
                         else if (interp_x == 1)
                             pred = (s_data[z][y][x - unit] + s_data[z][y][x + unit]) / 2;
                         else 
                             pred = (s_data[z][y - unit][x] + s_data[z][y][x - unit] - s_data[z][y - unit][x - unit]);
                     }
                 }
             }
 
             if CONSTEXPR (CUBE) {  //
                 auto interp_z = get_interp_order(z,BIZ,GDZ,global_z,data_size.z, AnchorBlockSizeZ);
                 auto interp_y = get_interp_order(y,BIY,GDY,global_y,data_size.y, AnchorBlockSizeY);
                 auto interp_x = get_interp_order(x,BIX,GDX,global_x,data_size.x, AnchorBlockSizeX);
 
                 if(interp_z == 4){
                     if(interp_y == 4){
                         if(interp_x == 4){
                             pred = (cubic_interpolator(s_data[z - 3*unit][y][x],
                                                          s_data[z - unit][y][x],
                                                          s_data[z + unit][y][x],
                                                          s_data[z+ 3*unit][y][x]) +
                                     cubic_interpolator(s_data[z][y - 3*unit][x],
                                                          s_data[z][y - unit][x],
                                                          s_data[z][y + unit][x],
                                                          s_data[z][y + 3*unit][x]) +
                                     cubic_interpolator(s_data[z][y][x - 3*unit],
                                                          s_data[z][y][x - unit],
                                                          s_data[z][y][x + unit],
                                                          s_data[z][y][x + 3*unit])
                                     ) / 3;
                         }
                         else{
                             pred = (cubic_interpolator(s_data[z - 3*unit][y][x],
                                                          s_data[z - unit][y][x],
                                                          s_data[z + unit][y][x],
                                                          s_data[z+ 3*unit][y][x]) +
                                     cubic_interpolator(s_data[z][y - 3*unit][x],
                                                          s_data[z][y - unit][x],
                                                          s_data[z][y + unit][x],
                                                          s_data[z][y + 3*unit][x])
                                     ) / 2;
                         }
                     }
                     else if(interp_x == 4){
                         pred = (cubic_interpolator(s_data[z - 3*unit][y][x],
                                                          s_data[z - unit][y][x],
                                                          s_data[z + unit][y][x],
                                                          s_data[z+ 3*unit][y][x]) +
                                 cubic_interpolator(s_data[z][y][x - 3*unit],
                                                      s_data[z][y][x - unit],
                                                      s_data[z][y][x + unit],
                                                      s_data[z][y][x + 3*unit])
                                 ) / 2;
 
                     }
                     else{
                         pred = cubic_interpolator(s_data[z - 3*unit][y][x],
                                                          s_data[z - unit][y][x],
                                                          s_data[z + unit][y][x],
                                                          s_data[z+ 3*unit][y][x]);
                     }
                 }
 
                 else if(interp_y == 4){
                     
                     if(interp_x == 4){
                         pred = (cubic_interpolator(s_data[z][y - 3*unit][x],
                                                          s_data[z][y - unit][x],
                                                          s_data[z][y + unit][x],
                                                          s_data[z][y+ 3*unit][x]) +
                                 cubic_interpolator(s_data[z][y][x - 3*unit],
                                                      s_data[z][y][x - unit],
                                                      s_data[z][y][x + unit],
                                                      s_data[z][y][x + 3*unit])
                                 ) / 2;
 
                     }
                     else{
                         pred = cubic_interpolator(s_data[z][y - 3*unit][x],
                                                          s_data[z][y - unit][x],
                                                          s_data[z][y + unit][x],
                                                          s_data[z][y+ 3*unit][x]);
                     }
                 }
                 else{
                     if(interp_x == 4)
                         pred = cubic_interpolator(s_data[z][y][x - 3*unit],
                                                   s_data[z][y][x - unit],
                                                   s_data[z][y][x + unit],
                                                   s_data[z][y][x + 3*unit]);
                     else if (interp_x == 3)
                         pred = (-s_data[z][y][x - 3*unit] + 6*s_data[z][y][x - unit] + 3*s_data[z][y][x + unit]) / 8;
                     else if (interp_x == 2)
                         pred = (3*s_data[z][y][x - unit] + 6*s_data[z][y][x + unit] - s_data[z][y][x + 3*unit]) / 8;
                     else if (interp_x == 1)
                         pred = (s_data[z][y][x - unit] + s_data[z][y][x + unit]) / 2;
                     else 
                         pred = s_data[z][y][x - unit];///to revise;
 
 
                 }
 
             }
 
 
                 
             
             
 
             if CONSTEXPR (WORKFLOW == SPLINE3_COMPR) {
                 
                 auto          err = s_data[z][y][x] - pred;
                 decltype(err) code;
                 // TODO unsafe, did not deal with the out-of-cap case
                 {
                     code = fabs(err) * eb_r + 1;
                     code = err < 0 ? -code : code;
                     code = int(code / 2) + radius;
                 }
                 s_ectrl[z][y][x] = code;  // TODO double check if unsigned type works
                 s_data[z][y][x]  = pred + (code - radius) * ebx2;
                 
 
             }
             else {  // TODO == DECOMPRESSS and static_assert
                 auto code       = s_ectrl[z][y][x];
                 s_data[z][y][x] = pred + (code - radius) * ebx2;
             }
         }
     };
     // -------------------------------------------------------------------------------- //
 
     if CONSTEXPR (COARSEN) {
         constexpr auto TOTAL = NUM_ELE;
             for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
                 auto [x,y,z]    = xyzmap(_tix, unit);
                 run(x, y, z);
             }
         
     }
     else {
         auto [x,y,z]    = xyzmap(TIX, unit);
         
 
      //   printf("%d %d %d\n", x,y,z);
         run(x, y, z);
     }
     __syncthreads();
 }
 

 template <
     typename T1, typename T2, typename FP, int SPLINE_DIM, int AnchorBlockSizeX,
     int AnchorBlockSizeY, int AnchorBlockSizeZ,
     int numAnchorBlockX,  // Number of Anchor blocks along X
     int numAnchorBlockY,  // Number of Anchor blocks along Y
     int numAnchorBlockZ,  // Number of Anchor blocks along Z
     typename LAMBDAX, typename LAMBDAY, typename LAMBDAZ, bool BLUE,
     bool YELLOW, bool HOLLOW, int LINEAR_BLOCK_SIZE, bool BORDER_INCLUSIVE,
     bool WORKFLOW>
 __forceinline__ __device__ void interpolate_stage(
     volatile T1 s_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
                       [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
                       [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
     volatile T2 s_ectrl[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
                        [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
                        [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
     DIM3 data_size, LAMBDAX xmap, LAMBDAY ymap, LAMBDAZ zmap, int unit,
     FP eb_r, FP ebx2, int radius, bool interpolator, int BLOCK_DIMX,
     int BLOCK_DIMY, bool COARSEN, int BLOCK_DIMZ)
 {
   // static_assert(
   //     BLOCK_DIMX * BLOCK_DIMY * (COARSEN ? 1 : BLOCK_DIMZ) <= 384,
   //     "block oversized");
   static_assert((BLUE or YELLOW or HOLLOW) == true, "must be one hot");
   static_assert((BLUE and YELLOW) == false, "must be only one hot (1)");
   static_assert((BLUE and YELLOW) == false, "must be only one hot (2)");
   static_assert((YELLOW and HOLLOW) == false, "must be only one hot (3)");
 
   auto run = [&](auto x, auto y, auto z) {
     if (xyz_predicate<SPLINE_DIM,
             AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
             numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ,
             BORDER_INCLUSIVE>(x, y, z, data_size)) {
       T1 pred = 0;
       auto global_x = BIX * (AnchorBlockSizeX * numAnchorBlockX) + x;
       auto global_y = BIY * (AnchorBlockSizeY * numAnchorBlockY) + y;
       auto global_z = BIZ * (AnchorBlockSizeZ * numAnchorBlockZ) + z;
       if (interpolator == 0) {
         if CONSTEXPR (BLUE) {  //
           if (BIZ != GDZ - 1) {
             if (z >= 3 * unit and
                 z + 3 * unit <= AnchorBlockSizeZ * numAnchorBlockZ)
               pred =
                   (-s_data[z - 3 * unit][y][x] + 9 * s_data[z - unit][y][x] +
                    9 * s_data[z + unit][y][x] - s_data[z + 3 * unit][y][x]) /
                   16;
             else if (z + 3 * unit <= AnchorBlockSizeZ * numAnchorBlockZ)
               pred = (3 * s_data[z - unit][y][x] + 6 * s_data[z + unit][y][x] -
                       s_data[z + 3 * unit][y][x]) /
                      8;
             else if (z >= 3 * unit)
               pred =
                   (-s_data[z - 3 * unit][y][x] + 6 * s_data[z - unit][y][x] +
                    3 * s_data[z + unit][y][x]) /
                   8;
 
             else
               pred = (s_data[z - unit][y][x] + s_data[z + unit][y][x]) / 2;
           }
           else {
             if (z >= 3 * unit) {
               if (z + 3 * unit <= AnchorBlockSizeZ * numAnchorBlockZ and
                   global_z + 3 * unit < data_size.z)
                 pred =
                     (-s_data[z - 3 * unit][y][x] + 9 * s_data[z - unit][y][x] +
                      9 * s_data[z + unit][y][x] - s_data[z + 3 * unit][y][x]) /
                     16;
               else if (global_z + unit < data_size.z)
                 pred =
                     (-s_data[z - 3 * unit][y][x] + 6 * s_data[z - unit][y][x] +
                      3 * s_data[z + unit][y][x]) /
                     8;
               else
                 pred = s_data[z - unit][y][x];
             }
             else {
               if (z + 3 * unit <= AnchorBlockSizeZ * numAnchorBlockZ and
                   global_z + 3 * unit < data_size.z)
                 pred =
                     (3 * s_data[z - unit][y][x] + 6 * s_data[z + unit][y][x] -
                      s_data[z + 3 * unit][y][x]) /
                     8;
               else if (global_z + unit < data_size.z)
                 pred = (s_data[z - unit][y][x] + s_data[z + unit][y][x]) / 2;
               else
                 pred = s_data[z - unit][y][x];
             }
           }
         }
         if CONSTEXPR (YELLOW) {  //
           if (BIY != GDY - 1) {
             if (y >= 3 * unit and
                 y + 3 * unit <= AnchorBlockSizeY * numAnchorBlockY)
               pred =
                   (-s_data[z][y - 3 * unit][x] + 9 * s_data[z][y - unit][x] +
                    9 * s_data[z][y + unit][x] - s_data[z][y + 3 * unit][x]) /
                   16;
             else if (y + 3 * unit <= AnchorBlockSizeY * numAnchorBlockY)
               pred = (3 * s_data[z][y - unit][x] + 6 * s_data[z][y + unit][x] -
                       s_data[z][y + 3 * unit][x]) /
                      8;
             else if (y >= 3 * unit)
               pred =
                   (-s_data[z][y - 3 * unit][x] + 6 * s_data[z][y - unit][x] +
                    3 * s_data[z][y + unit][x]) /
                   8;
             else
               pred = (s_data[z][y - unit][x] + s_data[z][y + unit][x]) / 2;
           }
           else {
             if (y >= 3 * unit) {
               if (y + 3 * unit <= AnchorBlockSizeY * numAnchorBlockY and
                   global_y + 3 * unit < data_size.y)
                 pred =
                     (-s_data[z][y - 3 * unit][x] + 9 * s_data[z][y - unit][x] +
                      9 * s_data[z][y + unit][x] - s_data[z][y + 3 * unit][x]) /
                     16;
               else if (global_y + unit < data_size.y)
                 pred =
                     (-s_data[z][y - 3 * unit][x] + 6 * s_data[z][y - unit][x] +
                      3 * s_data[z][y + unit][x]) /
                     8;
               else
                 pred = s_data[z][y - unit][x];
             }
             else {
               if (y + 3 * unit <= AnchorBlockSizeY * numAnchorBlockY and
                   global_y + 3 * unit < data_size.y)
                 pred =
                     (3 * s_data[z][y - unit][x] + 6 * s_data[z][y + unit][x] -
                      s_data[z][y + 3 * unit][x]) /
                     8;
               else if (global_y + unit < data_size.y)
                 pred = (s_data[z][y - unit][x] + s_data[z][y + unit][x]) / 2;
               else
                 pred = s_data[z][y - unit][x];
             }
           }
         }
 
         if CONSTEXPR (HOLLOW) {  //
           if (BIX != GDX - 1) {
             if (x >= 3 * unit and
                 x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX)
               pred =
                   (-s_data[z][y][x - 3 * unit] + 9 * s_data[z][y][x - unit] +
                    9 * s_data[z][y][x + unit] - s_data[z][y][x + 3 * unit]) /
                   16;
             else if (x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX)
               pred = (3 * s_data[z][y][x - unit] + 6 * s_data[z][y][x + unit] -
                       s_data[z][y][x + 3 * unit]) /
                      8;
             else if (x >= 3 * unit)
               pred =
                   (-s_data[z][y][x - 3 * unit] + 6 * s_data[z][y][x - unit] +
                    3 * s_data[z][y][x + unit]) /
                   8;
             else
               pred = (s_data[z][y][x - unit] + s_data[z][y][x + unit]) / 2;
           }
           else {
             if (x >= 3 * unit) {
               if (x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX and
                   global_x + 3 * unit < data_size.x)
                 pred =
                     (-s_data[z][y][x - 3 * unit] + 9 * s_data[z][y][x - unit] +
                      9 * s_data[z][y][x + unit] - s_data[z][y][x + 3 * unit]) /
                     16;
               else if (global_x + unit < data_size.x)
                 pred =
                     (-s_data[z][y][x - 3 * unit] + 6 * s_data[z][y][x - unit] +
                      3 * s_data[z][y][x + unit]) /
                     8;
               else
                 pred = s_data[z][y][x - unit];
             }
             else {
               if (x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX and
                   global_x + 3 * unit < data_size.x)
                 pred =
                     (3 * s_data[z][y][x - unit] + 6 * s_data[z][y][x + unit] -
                      s_data[z][y][x + 3 * unit]) /
                     8;
               else if (global_x + unit < data_size.x)
                 pred = (s_data[z][y][x - unit] + s_data[z][y][x + unit]) / 2;
               else
                 pred = s_data[z][y][x - unit];
             }
           }
         }
       }
       else {
         if CONSTEXPR (BLUE) {  //
 
           if (BIZ != GDZ - 1) {
             if (z >= 3 * unit and
                 z + 3 * unit <= AnchorBlockSizeZ * numAnchorBlockZ)
               pred =
                   (-3 * s_data[z - 3 * unit][y][x] +
                    23 * s_data[z - unit][y][x] + 23 * s_data[z + unit][y][x] -
                    3 * s_data[z + 3 * unit][y][x]) /
                   40;
             else if (z + 3 * unit <= AnchorBlockSizeZ * numAnchorBlockZ)
               pred = (3 * s_data[z - unit][y][x] + 6 * s_data[z + unit][y][x] -
                       s_data[z + 3 * unit][y][x]) /
                      8;
             else if (z >= 3 * unit)
               pred =
                   (-s_data[z - 3 * unit][y][x] + 6 * s_data[z - unit][y][x] +
                    3 * s_data[z + unit][y][x]) /
                   8;
 
             else
               pred = (s_data[z - unit][y][x] + s_data[z + unit][y][x]) / 2;
           }
           else {
             if (z >= 3 * unit) {
               if (z + 3 * unit <= AnchorBlockSizeZ * numAnchorBlockZ and
                   global_z + 3 * unit < data_size.z)
                 pred = (-3 * s_data[z - 3 * unit][y][x] +
                         23 * s_data[z - unit][y][x] +
                         23 * s_data[z + unit][y][x] -
                         3 * s_data[z + 3 * unit][y][x]) /
                        40;
               else if (global_z + unit < data_size.z)
                 pred =
                     (-s_data[z - 3 * unit][y][x] + 6 * s_data[z - unit][y][x] +
                      3 * s_data[z + unit][y][x]) /
                     8;
               else
                 pred = s_data[z - unit][y][x];
             }
             else {
               if (z + 3 * unit <= AnchorBlockSizeZ * numAnchorBlockZ and
                   global_z + 3 * unit < data_size.z)
                 pred =
                     (3 * s_data[z - unit][y][x] + 6 * s_data[z + unit][y][x] -
                      s_data[z + 3 * unit][y][x]) /
                     8;
               else if (global_z + unit < data_size.z)
                 pred = (s_data[z - unit][y][x] + s_data[z + unit][y][x]) / 2;
               else
                 pred = s_data[z - unit][y][x];
             }
           }
         }
         if CONSTEXPR (YELLOW) {  //
           if (BIY != GDY - 1) {
             if (y >= 3 * unit and
                 y + 3 * unit <= AnchorBlockSizeY * numAnchorBlockY)
               pred =
                   (-3 * s_data[z][y - 3 * unit][x] +
                    23 * s_data[z][y - unit][x] + 23 * s_data[z][y + unit][x] -
                    3 * s_data[z][y + 3 * unit][x]) /
                   40;
             else if (y + 3 * unit <= AnchorBlockSizeY * numAnchorBlockY)
               pred = (3 * s_data[z][y - unit][x] + 6 * s_data[z][y + unit][x] -
                       s_data[z][y + 3 * unit][x]) /
                      8;
             else if (y >= 3 * unit)
               pred =
                   (-s_data[z][y - 3 * unit][x] + 6 * s_data[z][y - unit][x] +
                    3 * s_data[z][y + unit][x]) /
                   8;
             else
               pred = (s_data[z][y - unit][x] + s_data[z][y + unit][x]) / 2;
           }
           else {
             if (y >= 3 * unit) {
               if (y + 3 * unit <= AnchorBlockSizeY * numAnchorBlockY and
                   global_y + 3 * unit < data_size.y)
                 pred = (-3 * s_data[z][y - 3 * unit][x] +
                         23 * s_data[z][y - unit][x] +
                         23 * s_data[z][y + unit][x] -
                         3 * s_data[z][y + 3 * unit][x]) /
                        40;
               else if (global_y + unit < data_size.y)
                 pred =
                     (-s_data[z][y - 3 * unit][x] + 6 * s_data[z][y - unit][x] +
                      3 * s_data[z][y + unit][x]) /
                     8;
               else
                 pred = s_data[z][y - unit][x];
             }
             else {
               if (y + 3 * unit <= AnchorBlockSizeY * numAnchorBlockY and
                   global_y + 3 * unit < data_size.y)
                 pred =
                     (3 * s_data[z][y - unit][x] + 6 * s_data[z][y + unit][x] -
                      s_data[z][y + 3 * unit][x]) /
                     8;
               else if (global_y + unit < data_size.y)
                 pred = (s_data[z][y - unit][x] + s_data[z][y + unit][x]) / 2;
               else
                 pred = s_data[z][y - unit][x];
             }
           }
         }
 
         if CONSTEXPR (HOLLOW) {  //
           if (BIX != GDX - 1) {
             if (x >= 3 * unit and
                 x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX)
               pred =
                   (-3 * s_data[z][y][x - 3 * unit] +
                    23 * s_data[z][y][x - unit] + 23 * s_data[z][y][x + unit] -
                    3 * s_data[z][y][x + 3 * unit]) /
                   40;
             else if (x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX)
               pred = (3 * s_data[z][y][x - unit] + 6 * s_data[z][y][x + unit] -
                       s_data[z][y][x + 3 * unit]) /
                      8;
             else if (x >= 3 * unit)
               pred =
                   (-s_data[z][y][x - 3 * unit] + 6 * s_data[z][y][x - unit] +
                    3 * s_data[z][y][x + unit]) /
                   8;
             else
               pred = (s_data[z][y][x - unit] + s_data[z][y][x + unit]) / 2;
           }
           else {
             if (x >= 3 * unit) {
               if (x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX and
                   global_x + 3 * unit < data_size.x)
                 pred = (-3 * s_data[z][y][x - 3 * unit] +
                         23 * s_data[z][y][x - unit] +
                         23 * s_data[z][y][x + unit] -
                         3 * s_data[z][y][x + 3 * unit]) /
                        40;
               else if (global_x + unit < data_size.x)
                 pred =
                     (-s_data[z][y][x - 3 * unit] + 6 * s_data[z][y][x - unit] +
                      3 * s_data[z][y][x + unit]) /
                     8;
               else
                 pred = s_data[z][y][x - unit];
             }
             else {
               if (x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX and
                   global_x + 3 * unit < data_size.x)
                 pred =
                     (3 * s_data[z][y][x - unit] + 6 * s_data[z][y][x + unit] -
                      s_data[z][y][x + 3 * unit]) /
                     8;
               else if (global_x + unit < data_size.x)
                 pred = (s_data[z][y][x - unit] + s_data[z][y][x + unit]) / 2;
               else
                 pred = s_data[z][y][x - unit];
             }
           }
         }
       }
       if CONSTEXPR (WORKFLOW == SPLINE3_COMPR) {
         auto err = s_data[z][y][x] - pred;
         decltype(err) code;
         // TODO unsafe, did not deal with the out-of-cap case
         {
           code = fabs(err) * eb_r + 1;
           code = err < 0 ? -code : code;
           code = int(code / 2) + radius;
         }
         // if()
         s_ectrl[z][y][x] = code;  // TODO double check if unsigned type works
         s_data[z][y][x] = pred + (code - radius) * ebx2;
       }
       else {  // TODO == DECOMPRESSS and static_assert
         auto code = s_ectrl[z][y][x];
         s_data[z][y][x] = pred + (code - radius) * ebx2;
         
       }
     }
   };
   int TOTAL = BLOCK_DIMX * BLOCK_DIMY * BLOCK_DIMZ;
     for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
       auto itix = (_tix % BLOCK_DIMX);
       auto itiy = (_tix / BLOCK_DIMX) % BLOCK_DIMY;
       auto itiz = (_tix / BLOCK_DIMX) / BLOCK_DIMY;
       auto x = xmap(itix, unit);
       auto y = ymap(itiy, unit);
       auto z = zmap(itiz, unit);
       run(x, y, z);
     }
   __syncthreads();
 }
 
 }  // namespace

 template <
     typename T1, typename T2, typename FP, int SPLINE_DIM, int AnchorBlockSizeX,
     int AnchorBlockSizeY, int AnchorBlockSizeZ,
     int numAnchorBlockX,  // Number of Anchor blocks along X
     int numAnchorBlockY,  // Number of Anchor blocks along Y
     int numAnchorBlockZ,  // Number of Anchor blocks along Z
     int LINEAR_BLOCK_SIZE, bool WORKFLOW, bool PROBE_PRED_ERROR>
 __device__ void cusz::device_api::spline_layout_interpolate(
     volatile T1 s_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
                       [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
                       [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
     volatile T2 s_ectrl[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
                        [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
                        [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
     DIM3 data_size, FP eb_r, FP ebx2, int radius,
     INTERPOLATION_PARAMS intp_param)
 {
   auto xblue = [] __device__(int _tix, int unit) -> int {
     return unit * (_tix * 2);
   };
   auto yblue = [] __device__(int _tiy, int unit) -> int {
     return unit * (_tiy * 2);
   };
   auto zblue = [] __device__(int _tiz, int unit) -> int {
     return unit * (_tiz * 2 + 1);
   };
 
   auto xblue_reverse = [] __device__(int _tix, int unit) -> int {
     return unit * (_tix);
   };
   auto yblue_reverse = [] __device__(int _tiy, int unit) -> int {
     return unit * (_tiy);
   };
   auto zblue_reverse = [] __device__(int _tiz, int unit) -> int {
     return unit * (_tiz * 2 + 1);
   };
 
   auto xyellow = [] __device__(int _tix, int unit) -> int {
     return unit * (_tix * 2);
   };
   auto yyellow = [] __device__(int _tiy, int unit) -> int {
     return unit * (_tiy * 2 + 1);
   };
   auto zyellow = [] __device__(int _tiz, int unit) -> int {
     return unit * (_tiz);
   };
 
   auto xyellow_reverse = [] __device__(int _tix, int unit) -> int {
     return unit * (_tix);
   };
   auto yyellow_reverse = [] __device__(int _tiy, int unit) -> int {
     return unit * (_tiy * 2 + 1);
   };
   auto zyellow_reverse = [] __device__(int _tiz, int unit) -> int {
     return unit * (_tiz * 2);
   };
 
   auto xhollow = [] __device__(int _tix, int unit) -> int {
     return unit * (_tix * 2 + 1);
   };
   auto yhollow = [] __device__(int _tiy, int unit) -> int {
     return unit * (_tiy);
   };
   auto zhollow = [] __device__(int _tiz, int unit) -> int {
     return unit * (_tiz);
   };
 
   auto xhollow_reverse = [] __device__(int _tix, int unit) -> int {
     return unit * (_tix * 2 + 1);
   };
   auto yhollow_reverse = [] __device__(int _tiy, int unit) -> int {
     return unit * (_tiy * 2);
   };
   auto zhollow_reverse = [] __device__(int _tiz, int unit) -> int {
     return unit * (_tiz * 2);
   };
   
   auto xyzmap_line_16b_1u = [] __device__(int _tix, int unit) -> std::tuple<int,int,int> {
    constexpr auto N = 8;
    constexpr auto L = N*(N+1)*(N+1); 
    constexpr auto Q = (N+1)*(N+1); 
    auto group = _tix / L ;
    auto m = _tix % L ;
    auto i = m / Q;
    auto j = (m % Q) / (N+1);
    auto k = (m % Q) % (N+1);
    if(group==0)
        return std::make_tuple(2*i+1,2*j,2*k);
    else if (group==1)
        return std::make_tuple(2*k,2*i+1,2*j);
    else
        return std::make_tuple(2*j,2*k,2*i+1);

};

auto xyzmap_face_16b_1u = [] __device__(int _tix, int unit) -> std::tuple<int,int,int> {
    constexpr auto N = 8;
    constexpr auto L = N*N*(N+1);
    constexpr auto Q = N*N; 
    auto group = _tix / L ;
    auto m = _tix % L ;
    auto i = m / Q;
    auto j = (m % Q) / N;
    auto k = (m % Q) % N;
    if(group==0)
        return std::make_tuple(2*i,2*j+1,2*k+1);
    else if (group==1)
        return std::make_tuple(2*k+1,2*i,2*j+1);
    else
        return std::make_tuple(2*j+1,2*k+1,2*i);

};

 auto xyzmap_cube_16b_1u = [] __device__(int _tix, int unit) -> std::tuple<int,int,int> {
    constexpr auto N = 8;
    constexpr auto Q = N * N; 
    auto i = _tix / Q;
    auto j = (_tix % Q) / N;
    auto k = (_tix % Q) % N;
    return std::make_tuple(2*i+1,2*j+1,2*k+1);

};

auto xyzmap_line_16b_2u = [] __device__(int _tix, int unit) -> std::tuple<int,int,int> {
    constexpr auto N = 4;
    constexpr auto L = N*(N+1)*(N+1); 
    constexpr auto Q = (N+1)*(N+1); 
    auto group = _tix / L ;
    auto m = _tix % L ;
    auto i = m / Q;
    auto j = (m % Q) / (N+1);
    auto k = (m % Q) % (N+1);
    if(group==0)
        return std::make_tuple(4*i+2,4*j,4*k);
    else if (group==1)
        return std::make_tuple(4*k,4*i+2,4*j);
    else
        return std::make_tuple(4*j,4*k,4*i+2);

};

auto xyzmap_face_16b_2u = [] __device__(int _tix, int unit) -> std::tuple<int,int,int> {
    constexpr auto N = 4;
    constexpr auto L = N*N*(N+1);
    constexpr auto Q = N*N; 
    auto group = _tix / L ;
    auto m = _tix % L ;
    auto i = m / Q;
    auto j = (m % Q) / N;
    auto k = (m % Q) % N;
    if(group==0)
        return std::make_tuple(4*i,4*j+2,4*k+2);
    else if (group==1)
        return std::make_tuple(4*k+2,4*i,4*j+2);
    else
        return std::make_tuple(4*j+2,4*k+2,4*i);

};

  auto xyzmap_cube_16b_2u = [] __device__(int _tix, int unit) -> std::tuple<int,int,int> {
    constexpr auto N = 4;
    constexpr auto Q = N * N; 
    auto i = _tix / Q;
    auto j = (_tix % Q) / N;
    auto k = (_tix % Q) % N;
    return std::make_tuple(4*i+2,4*j+2,4*k+2);

  };


  auto nan_cubic_interp = [] __device__ (T1 a, T1 b, T1 c, T1 d) -> T1{
      return (-a+9*b+9*c-9*d) / 16;
  };

  auto nat_cubic_interp = [] __device__ (T1 a, T1 b, T1 c, T1 d) -> T1{
      return (-3*a+23*b+23*c-3*d) / 40;
  };




   constexpr auto COARSEN = true;
   constexpr auto NO_COARSEN = false;
   constexpr auto BORDER_INCLUSIVE = true;
   constexpr auto BORDER_EXCLUSIVE = false;
 
   FP cur_ebx2 = ebx2, cur_eb_r = eb_r;
 
   auto calc_eb = [&](auto unit) {
     cur_ebx2 = ebx2, cur_eb_r = eb_r;
     int temp = 1;
     while (temp < unit) {
       temp *= 2;
       cur_eb_r *= intp_param.alpha;
       cur_ebx2 /= intp_param.alpha;
     }
     if (cur_ebx2 < ebx2 / intp_param.beta) {
       cur_ebx2 = ebx2 / intp_param.beta;
       cur_eb_r = eb_r * intp_param.beta;
     }
   };
   

    int max_unit = ((AnchorBlockSizeX >= AnchorBlockSizeY) ? AnchorBlockSizeX : AnchorBlockSizeY);
    max_unit = ((max_unit >= AnchorBlockSizeZ) ? max_unit : AnchorBlockSizeZ);
    max_unit /= 2;
    int unit_x = AnchorBlockSizeX, unit_y = AnchorBlockSizeY, unit_z = AnchorBlockSizeZ;
    
    #pragma unroll
    for(int unit = max_unit; unit > 2; unit /= 2){
      // if(threadIdx.x == 0 && blockIdx.x + blockIdx.y + blockIdx.z == 0) printf("unit=%d\n", unit);
      calc_eb(unit);
     interpolate_stage<
         T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
         numAnchorBlockX,  // Number of Anchor blocks along X
         numAnchorBlockY,  // Number of Anchor blocks along Y
         numAnchorBlockZ,  // Number of Anchor blocks along Z
         decltype(xhollow_reverse), decltype(yhollow_reverse),
         decltype(zhollow_reverse),  //
         false, false, true, LINEAR_BLOCK_SIZE,
         BORDER_INCLUSIVE, WORKFLOW>(
         s_data, s_ectrl, data_size, xhollow_reverse, yhollow_reverse,
         zhollow_reverse, unit, cur_eb_r, cur_ebx2, radius,
         intp_param.interpolators[0], numAnchorBlockX * AnchorBlockSizeX / unit_x, numAnchorBlockY * AnchorBlockSizeY / unit_y + (SPLINE_DIM >= 2), NO_COARSEN, numAnchorBlockZ * AnchorBlockSizeZ / unit_z + (SPLINE_DIM >= 3));
         unit_x /= 2;
     interpolate_stage<
         T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
         numAnchorBlockX,  // Number of Anchor blocks along X
         numAnchorBlockY,  // Number of Anchor blocks along Y
         numAnchorBlockZ,  // Number of Anchor blocks along Z
         decltype(xyellow_reverse), decltype(yyellow_reverse),
         decltype(zyellow_reverse),  //
         false, true, false, LINEAR_BLOCK_SIZE,
         BORDER_INCLUSIVE, WORKFLOW>(
         s_data, s_ectrl, data_size, xyellow_reverse, yyellow_reverse,
         zyellow_reverse, unit, cur_eb_r, cur_ebx2, radius,
         intp_param.interpolators[1], numAnchorBlockX * AnchorBlockSizeX / unit_x + (SPLINE_DIM >= 1), numAnchorBlockY * AnchorBlockSizeY / unit_y, NO_COARSEN, numAnchorBlockZ * AnchorBlockSizeZ / unit_z + (SPLINE_DIM >= 3));
          unit_y /= 2;
    interpolate_stage<
        T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xblue_reverse), decltype(yblue_reverse),
        decltype(zblue_reverse),  //
        true, false, false, LINEAR_BLOCK_SIZE,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xblue_reverse, yblue_reverse,
        zblue_reverse, unit, cur_eb_r, cur_ebx2, radius,
        intp_param.interpolators[2], numAnchorBlockX * AnchorBlockSizeX / unit_x + (SPLINE_DIM >= 1), numAnchorBlockY * AnchorBlockSizeY / unit_y + (SPLINE_DIM >= 2), NO_COARSEN, numAnchorBlockZ * AnchorBlockSizeZ / unit_z);
      unit_z /= 2;
    }
      int unit = 2;
      calc_eb(unit);
         if(intp_param.interpolators[1]==0){
 
          interpolate_stage_md<
              T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
              numAnchorBlockX,  // Number of Anchor blocks along X
              numAnchorBlockY,  // Number of Anchor blocks along Y
              numAnchorBlockZ,  // Number of Anchor blocks along Z
              decltype(xyzmap_line_16b_2u), //
              true, false, false, LINEAR_BLOCK_SIZE,300 ,NO_COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
              s_data, s_ectrl,data_size, xyzmap_line_16b_2u, unit, cur_eb_r, cur_ebx2, radius, nan_cubic_interp);
  
          interpolate_stage_md<
              T1, T2, FP,
              SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
              numAnchorBlockX,  // Number of Anchor blocks along X
              numAnchorBlockY,  // Number of Anchor blocks along Y
              numAnchorBlockZ,  // Number of Anchor blocks along Z
              decltype(xyzmap_face_16b_2u), //
              false, true, false, LINEAR_BLOCK_SIZE,240 ,NO_COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
              s_data, s_ectrl,data_size, xyzmap_face_16b_2u, unit, cur_eb_r, cur_ebx2, radius, nan_cubic_interp);
  
          interpolate_stage_md<
              T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
              numAnchorBlockX,  // Number of Anchor blocks along X
              numAnchorBlockY,  // Number of Anchor blocks along Y
              numAnchorBlockZ,  // Number of Anchor blocks along Z
              decltype(xyzmap_cube_16b_2u), //
              false, false, true, LINEAR_BLOCK_SIZE,64 ,COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
              s_data, s_ectrl,data_size, xyzmap_cube_16b_2u, unit, cur_eb_r, cur_ebx2, radius, nan_cubic_interp);
  
      }
      else{
          interpolate_stage_md<
              T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
              numAnchorBlockX,  // Number of Anchor blocks along X
              numAnchorBlockY,  // Number of Anchor blocks along Y
              numAnchorBlockZ,  // Number of Anchor blocks along Z
              decltype(xyzmap_line_16b_2u), //
              true, false, false, LINEAR_BLOCK_SIZE,300 ,NO_COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
              s_data, s_ectrl,data_size, xyzmap_line_16b_2u, unit, cur_eb_r, cur_ebx2, radius, nat_cubic_interp);
  
          interpolate_stage_md<
              T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
              numAnchorBlockX,  // Number of Anchor blocks along X
              numAnchorBlockY,  // Number of Anchor blocks along Y
              numAnchorBlockZ,  // Number of Anchor blocks along Z
              decltype(xyzmap_face_16b_2u), //
              false, true, false, LINEAR_BLOCK_SIZE,240 ,NO_COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
              s_data, s_ectrl,data_size, xyzmap_face_16b_2u, unit, cur_eb_r, cur_ebx2, radius, nat_cubic_interp);
  
          interpolate_stage_md<
              T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
              numAnchorBlockX,  // Number of Anchor blocks along X
              numAnchorBlockY,  // Number of Anchor blocks along Y
              numAnchorBlockZ,  // Number of Anchor blocks along Z
              decltype(xyzmap_cube_16b_2u), //
              false, false, true, LINEAR_BLOCK_SIZE,64 ,NO_COARSEN, BORDER_EXCLUSIVE, WORKFLOW>(
              s_data, s_ectrl,data_size, xyzmap_cube_16b_2u, unit, cur_eb_r, cur_ebx2, radius, nat_cubic_interp);
      }

      unit = 1;
      calc_eb(unit);
      if(intp_param.interpolators[0]==0){
          interpolate_stage_md<
              T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
              numAnchorBlockX,  // Number of Anchor blocks along X
              numAnchorBlockY,  // Number of Anchor blocks along Y
              numAnchorBlockZ,  // Number of Anchor blocks along Z
              decltype(xyzmap_line_16b_1u), //
              true, false, false, LINEAR_BLOCK_SIZE,1944 ,COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
              s_data, s_ectrl,data_size, xyzmap_line_16b_1u, unit, cur_eb_r, cur_ebx2, radius, nan_cubic_interp);
  
          interpolate_stage_md<
              T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
              numAnchorBlockX,  // Number of Anchor blocks along X
              numAnchorBlockY,  // Number of Anchor blocks along Y
              numAnchorBlockZ,  // Number of Anchor blocks along Z
              decltype(xyzmap_face_16b_1u), //
              false, true, false, LINEAR_BLOCK_SIZE,1728 ,COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
              s_data, s_ectrl,data_size, xyzmap_face_16b_1u, unit, cur_eb_r, cur_ebx2, radius, nan_cubic_interp);
  
          interpolate_stage_md<
              T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
              numAnchorBlockX,  // Number of Anchor blocks along X
              numAnchorBlockY,  // Number of Anchor blocks along Y
              numAnchorBlockZ,  // Number of Anchor blocks along Z
              decltype(xyzmap_cube_16b_1u), //
              false, false, true, LINEAR_BLOCK_SIZE,512 ,COARSEN, BORDER_EXCLUSIVE, WORKFLOW>(
              s_data, s_ectrl,data_size, xyzmap_cube_16b_1u, unit, cur_eb_r, cur_ebx2, radius, nan_cubic_interp);
  
      }
      else{
          interpolate_stage_md<
              T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
              numAnchorBlockX,  // Number of Anchor blocks along X
              numAnchorBlockY,  // Number of Anchor blocks along Y
              numAnchorBlockZ,  // Number of Anchor blocks along Z
              decltype(xyzmap_line_16b_1u), //
              true, false, false, LINEAR_BLOCK_SIZE,1944 ,COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
              s_data, s_ectrl,data_size, xyzmap_line_16b_1u, unit, cur_eb_r, cur_ebx2, radius, nat_cubic_interp);
  
          interpolate_stage_md<
              T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
              numAnchorBlockX,  // Number of Anchor blocks along X
              numAnchorBlockY,  // Number of Anchor blocks along Y
              numAnchorBlockZ,  // Number of Anchor blocks along Z
              decltype(xyzmap_face_16b_1u), //
              false, true, false, LINEAR_BLOCK_SIZE,1728 ,COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
              s_data, s_ectrl,data_size, xyzmap_face_16b_1u, unit, cur_eb_r, cur_ebx2, radius, nat_cubic_interp);
  
          interpolate_stage_md<
              T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
              numAnchorBlockX,  // Number of Anchor blocks along X
              numAnchorBlockY,  // Number of Anchor blocks along Y
              numAnchorBlockZ,  // Number of Anchor blocks along Z
              decltype(xyzmap_cube_16b_1u), //
              false, false, true, LINEAR_BLOCK_SIZE,512 ,COARSEN, BORDER_EXCLUSIVE, WORKFLOW>(
              s_data, s_ectrl,data_size, xyzmap_cube_16b_1u, unit, cur_eb_r, cur_ebx2, radius, nat_cubic_interp);
          
  
      }


   
  
 }
 
 template <
     typename TITER, typename EITER, typename FP, int SPLINE_DIM, int AnchorBlockSizeX,
     int AnchorBlockSizeY, int AnchorBlockSizeZ,
     int numAnchorBlockX,  // Number of Anchor blocks along X
     int numAnchorBlockY,  // Number of Anchor blocks along Y
     int numAnchorBlockZ,  // Number of Anchor blocks along Z
     int LINEAR_BLOCK_SIZE, typename CompactVal, typename CompactIdx,
     typename CompactNum>
 __global__ void cusz::c_spline_infprecis_data(
     TITER data, DIM3 data_size, STRIDE3 data_leap, EITER ectrl,
     DIM3 ectrl_size, STRIDE3 ectrl_leap, TITER anchor, STRIDE3 anchor_leap,
     CompactVal compact_val, CompactIdx compact_idx, CompactNum compact_num,
     FP eb_r, FP ebx2, int radius,
     INTERPOLATION_PARAMS intp_param  //,
                                      // TITER errors
 )
 {
   // compile time variables
   using T = typename std::remove_pointer<TITER>::type;
   using E = typename std::remove_pointer<EITER>::type;
 
   {
     __shared__ struct {
       T data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
             [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
             [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)];
       T ectrl[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
              [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
              [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)];
     } shmem;
     c_reset_scratch_data<
         T, T, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
         numAnchorBlockX,  // Number of Anchor blocks along X
         numAnchorBlockY,  // Number of Anchor blocks along Y
         numAnchorBlockZ,  // Number of Anchor blocks along Z
         LINEAR_BLOCK_SIZE>(shmem.data, shmem.ectrl, radius);
 
     global2shmem_data<
         T, T, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
         numAnchorBlockX,  // Number of Anchor blocks along X
         numAnchorBlockY,  // Number of Anchor blocks along Y
         numAnchorBlockZ,  // Number of Anchor blocks along Z
         LINEAR_BLOCK_SIZE>(data, data_size, data_leap, shmem.data);
     c_gather_anchor<
         T, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
         numAnchorBlockX,  // Number of Anchor blocks along X
         numAnchorBlockY,  // Number of Anchor blocks along Y
         numAnchorBlockZ  // Number of Anchor blocks along Z
         >(data, data_size, data_leap, anchor, anchor_leap);
 
     cusz::device_api::spline_layout_interpolate<
         T, T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
         numAnchorBlockX,  // Number of Anchor blocks along X
         numAnchorBlockY,  // Number of Anchor blocks along Y
         numAnchorBlockZ,  // Number of Anchor blocks along Z
         LINEAR_BLOCK_SIZE, SPLINE3_COMPR, false>(
         shmem.data, shmem.ectrl, data_size, eb_r, ebx2, radius, intp_param);
        //  if(threadIdx.x == 0 && blockIdx.x + blockIdx.y + blockIdx.z == 0) printf("Finish spline layout interpolate, start shmem2global_data_with_compaction\n");
     shmem2global_data_with_compaction<
         T, E, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
         numAnchorBlockX,  // Number of Anchor blocks along X
         numAnchorBlockY,  // Number of Anchor blocks along Y
         numAnchorBlockZ,  // Number of Anchor blocks along Z
         LINEAR_BLOCK_SIZE>(
         shmem.ectrl, ectrl, ectrl_size, ectrl_leap, radius, compact_val,
         compact_idx, compact_num);
   }
 }
 
 template <
     typename EITER, typename TITER, typename FP, int SPLINE_DIM, int AnchorBlockSizeX,
     int AnchorBlockSizeY, int AnchorBlockSizeZ,
     int numAnchorBlockX,  // Number of Anchor blocks along X
     int numAnchorBlockY,  // Number of Anchor blocks along Y
     int numAnchorBlockZ,  // Number of Anchor blocks along Z
     int LINEAR_BLOCK_SIZE>
 __global__ void cusz::x_spline_infprecis_data(
     EITER ectrl,          // input 1
     DIM3 ectrl_size,      //
     STRIDE3 ectrl_leap,   //
     TITER anchor,         // input 2
     DIM3 anchor_size,     //
     STRIDE3 anchor_leap,  //
     TITER data,           // output
     DIM3 data_size,       //
     STRIDE3 data_leap,    //
     TITER outlier_tmp,
     FP eb_r, FP ebx2, int radius, INTERPOLATION_PARAMS intp_param)
 {
   // compile time variables
   using E = typename std::remove_pointer<EITER>::type;
   using T = typename std::remove_pointer<TITER>::type;
 
   __shared__ struct {
     T data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
           [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
           [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)];
     T ectrl[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
            [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
            [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)];
   } shmem;
   x_reset_scratch_data<
       T, T, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
       numAnchorBlockX,  // Number of Anchor blocks along X
       numAnchorBlockY,  // Number of Anchor blocks along Y
       numAnchorBlockZ,  // Number of Anchor blocks along Z
       LINEAR_BLOCK_SIZE>(
       shmem.data, shmem.ectrl, anchor, anchor_size, anchor_leap);
   global2shmem_fuse<
       T, E, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
       numAnchorBlockX,  // Number of Anchor blocks along X
       numAnchorBlockY,  // Number of Anchor blocks along Y
       numAnchorBlockZ,  // Number of Anchor blocks along Z
      //  LINEAR_BLOCK_SIZE>(ectrl, ectrl_size, ectrl_leap, data, shmem.ectrl);
       LINEAR_BLOCK_SIZE>(ectrl, ectrl_size, ectrl_leap, outlier_tmp, shmem.ectrl);
 
   cusz::device_api::spline_layout_interpolate<
       T, T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
       numAnchorBlockX,  // Number of Anchor blocks along X
       numAnchorBlockY,  // Number of Anchor blocks along Y
       numAnchorBlockZ,  // Number of Anchor blocks along Z
       LINEAR_BLOCK_SIZE, SPLINE3_DECOMPR, false>(
       shmem.data, shmem.ectrl, data_size, eb_r, ebx2, radius, intp_param);
   shmem2global_data<
       T, T, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
       numAnchorBlockX,  // Number of Anchor blocks along X
       numAnchorBlockY,  // Number of Anchor blocks along Y
       numAnchorBlockZ,  // Number of Anchor blocks along Z
       LINEAR_BLOCK_SIZE>(shmem.data, data, data_size, data_leap);
 }
 
 #undef TIX
 #undef TIY
 #undef TIZ
 #undef BIX
 #undef BIY
 #undef BIZ
 #undef BDX
 #undef BDY
 #undef BDZ
 #undef GDX
 #undef GDY
 #undef GDZ
 
 #endif