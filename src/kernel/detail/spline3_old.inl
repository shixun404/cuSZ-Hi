 /**
 * @file spline3.inl
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
 
 #define SPLINE3_PRED_ATT true
 #define SPLINE3_AB_ATT false
 
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
 
 using DIM     = u4;
 using STRIDE  = u4;
 using DIM3    = dim3;
 using STRIDE3 = dim3;
 
 //constexpr int BLOCK8  = 8;
 //constexpr int BLOCK32 = 32;
 constexpr int BLOCK16 = 16;
 constexpr int BLOCK17 = 17;
 constexpr int DEFAULT_LINEAR_BLOCK_SIZE = 384;
 
 #define SHM_ERROR s_ectrl
 
 namespace cusz {
 
 /********************************************************************************
  * host API
  ********************************************************************************/
 template <typename TITER, int LINEAR_BLOCK_SIZE>
 __global__ void c_spline3d_profiling_16x16x16data(
     TITER   data,
     DIM3    data_size,
     STRIDE3 data_leap,
     TITER errors);
 
 template <typename TITER, int LINEAR_BLOCK_SIZE>
 __global__ void c_spline3d_profiling_data_2(
     TITER   data,
     DIM3    data_size,
     STRIDE3 data_leap,
     TITER errors);
 
 template <
     typename TITER,
     typename EITER,
     typename FP            = float,
     int  LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE, 
     typename CompactVal = TITER,
     typename CompactIdx = uint32_t*,
     typename CompactNum = uint32_t*>
 __global__ void c_spline3d_infprecis_16x16x16data(
     TITER   data,
     DIM3    data_size,
     STRIDE3 data_leap,
     EITER   ectrl,
     DIM3    ectrl_size,
     STRIDE3 ectrl_leap,
     TITER   anchor,
     STRIDE3 anchor_leap,
     CompactVal cval,
     CompactIdx cidx,
     CompactNum cn,
     FP      eb_r,
     FP      ebx2,
     int     radius,
     INTERPOLATION_PARAMS intp_param,
     TITER errors);
 
 template <
     typename EITER,
     typename TITER,
     typename FP           = float,
     int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
 __global__ void x_spline3d_infprecis_16x16x16data(
     EITER   ectrl,        // input 1
     DIM3    ectrl_size,   //
     STRIDE3 ectrl_leap,   //
     TITER   anchor,       // input 2
     DIM3    anchor_size,  //
     STRIDE3 anchor_leap,  //
     TITER   data,         // output
     DIM3    data_size,    //
     STRIDE3 data_leap,    //
     FP      eb_r,
     FP      ebx2,
     int     radius,
     INTERPOLATION_PARAMS intp_param);
 
 template <typename TITER>
 __global__ void reset_errors(TITER errors);
 
 template <typename TITER, typename FP, int LINEAR_BLOCK_SIZE>
 __global__ void pa_spline3d_infprecis_16x16x16data(
     TITER   data,
     DIM3    data_size,
     STRIDE3 data_leap,
     DIM3 sample_starts,
     DIM3 sample_block_grid_sizes,
     DIM3 sample_strides,
     FP      eb_r,
     FP      ebx2,
     INTERPOLATION_PARAMS intp_param,
     TITER errors,
     bool workflow = SPLINE3_PRED_ATT
     );
 
 
 namespace device_api {
 /********************************************************************************
  * device API
  ********************************************************************************/
 
     template <typename T,int  LINEAR_BLOCK_SIZE>
 __device__ void auto_tuning(volatile T s_data[17][17][17],  volatile T local_errs[6], DIM3  data_size, volatile T* count);
 
  template <typename T,int  LINEAR_BLOCK_SIZE>
 __device__ void auto_tuning_2(volatile T s_data[17][17][17],  volatile T local_errs[6], DIM3  data_size, volatile T* count);
 
 template <
     typename T1,
     typename T2,
     typename FP,
     int  LINEAR_BLOCK_SIZE,
     bool WORKFLOW         = SPLINE3_COMPR,
     bool PROBE_PRED_ERROR = false>
 __device__ void
 spline3d_layout2_interpolate(volatile T1 s_data[17][17][17], volatile T2 s_ectrl[17][17][17],  DIM3  data_size,FP eb_r, FP ebx2, int radius, INTERPOLATION_PARAMS intp_param);
 
 template <typename T, typename FP,int LINEAR_BLOCK_SIZE, bool WORKFLOW>
 __device__ void spline3d_layout2_interpolate_att(
     volatile T s_data[17][17][17],
      DIM3    data_size,
     DIM3 global_starts,FP eb_r, FP ebx2,uint8_t level,INTERPOLATION_PARAMS intp_param,volatile T *error);
 
 
 }  // namespace device_api
 
 
 
 
 }  // namespace cusz
 
 /********************************************************************************
  * helper function
  ********************************************************************************/
 
 namespace {
 
 template <bool INCLUSIVE = true>
 __forceinline__ __device__ bool xyz17x17x17_predicate(unsigned int x, unsigned int y, unsigned int z,const DIM3 &data_size)
 {
     if CONSTEXPR (INCLUSIVE) {  //
 
         
             return (x <= BLOCK16 and y <= BLOCK16 and z <= BLOCK16) and BIX*BLOCK16+x<data_size.x and BIY*BLOCK16+y<data_size.y and BIZ*BLOCK16+z<data_size.z;
     }
     else {
         return x < BLOCK16+(BIX==GDX-1) and y < BLOCK16+(BIY==GDY-1) and z < BLOCK16+(BIZ==GDZ-1) and BIX*BLOCK16+x<data_size.x and BIY*BLOCK16+y<data_size.y and BIZ*BLOCK16+z<data_size.z;
     }
 }
 
 template <bool INCLUSIVE = true>
 __forceinline__ __device__ bool xyz17x17x17_predicate_att(unsigned int x, unsigned int y, unsigned int z,const DIM3 &data_size, const DIM3 & global_starts)
 {
     if CONSTEXPR (INCLUSIVE) {  //
 
         
             return (x <= BLOCK16 and y <= BLOCK16 and z <= BLOCK16) and global_starts.x+x<data_size.x and global_starts.y+y<data_size.y and global_starts.z+z<data_size.z;
     }
     else {
         return x < BLOCK16+(BIX==GDX-1) and y < BLOCK16+(BIY==GDY-1) and z < BLOCK16+(BIZ==GDZ-1) and global_starts.x+x<data_size.x and global_starts.y+y<data_size.y and global_starts.z+z<data_size.z;
     }
 }
 
 
 
 // control block_id3 in function call
 template <typename T, bool PRINT_FP = true, int XEND = BLOCK17, int YEND = BLOCK17, int ZEND = BLOCK17>
 __device__ void
 spline3d_print_block_from_GPU(T volatile a[17][17][17], int radius = 512, bool compress = true, bool print_ectrl = true)
 {
     for (auto z = 0; z < ZEND; z++) {
         printf("\nprint from GPU, z=%d\n", z);
         printf("    ");
         for (auto i = 0; i < BLOCK17; i++) printf("%3d", i);
         printf("\n");
 
         for (auto y = 0; y < YEND; y++) {
             printf("y=%d ", y);
             for (auto x = 0; x < XEND; x++) {  //
                 if CONSTEXPR (PRINT_FP) { printf("%.2e\t", (float)a[z][y][x]); }
                 else {
                     T c = print_ectrl ? a[z][y][x] - radius : a[z][y][x];
                     if (compress) {
                         if (c == 0) { printf("%3c", '.'); }
                         else {
                             if (abs(c) >= 10) { printf("%3c", '*'); }
                             else {
                                 if (print_ectrl) { printf("%3d", c); }
                                 else {
                                     printf("%4.2f", c);
                                 }
                             }
                         }
                     }
                     else {
                         if (print_ectrl) { printf("%3d", c); }
                         else {
                             printf("%4.2f", c);
                         }
                     }
                 }
             }
             printf("\n");
         }
     }
     printf("\nGPU print end\n\n");
 }
 
 template <typename T1, typename T2, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
 __device__ void c_reset_scratch_17x17x17data(volatile T1 s_data[17][17][17], volatile T2 s_ectrl[17][17][17], int radius)
 {
     // alternatively, reinterprete cast volatile T?[][][] to 1D
     for (auto _tix = TIX; _tix < BLOCK17 * BLOCK17 * BLOCK17; _tix += LINEAR_BLOCK_SIZE) {
         auto x = (_tix % BLOCK17);
         auto y = (_tix / BLOCK17) % BLOCK17;
         auto z = (_tix / BLOCK17) / BLOCK17;
 
         s_data[z][y][x] = 0;
         /*****************************************************************************
          okay to use
          ******************************************************************************/
         if (x % 16 == 0 and y % 16 == 0 and z % 16 == 0) s_ectrl[z][y][x] = radius;
         /*****************************************************************************
          alternatively
          ******************************************************************************/
         // s_ectrl[z][y][x] = radius;
     }
     __syncthreads();
 }
 
 
 template <typename T, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
 __device__ void c_reset_scratch_profiling_16x16x16data(volatile T s_data[16][16][16], T default_value)
 {
     for (auto _tix = TIX; _tix < 16 * 16 * 16; _tix += LINEAR_BLOCK_SIZE) {
         auto x = (_tix % 16);
         auto y = (_tix / 16) % 16;
         auto z = (_tix / 16) / 16;
 
         s_data[z][y][x] = default_value;
 
     }
 }
 
 template <typename T, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
 __device__ void c_reset_scratch_profiling_data_2(volatile T s_data[64], T nx[64][4], T ny[64][4], T nz[64][4],T default_value)
 {
     for (auto _tix = TIX; _tix < 64 * 4; _tix += LINEAR_BLOCK_SIZE) {
         auto x = (_tix % 4);
         auto yz=_tix/4;
 
         
 
         nx[yz][x]=ny[yz][x]=nz[yz][x]=default_value;
         s_data[TIX] = default_value;
 
     }
 }
 
 template <typename T1, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
 __device__ void c_gather_anchor_old(T1* data, DIM3 data_size, STRIDE3 data_leap, T1* anchor, STRIDE3 anchor_leap)
 {
     auto x = (TIX % BLOCK16) + BIX * BLOCK16;
     auto y = (TIX / BLOCK16) % BLOCK16 + BIY * BLOCK16;
     auto z = (TIX / BLOCK16) / BLOCK16 + BIZ * BLOCK16;
 
     bool pred1 = x % 8 == 0 and y % 8 == 0 and z % 8 == 0;
     bool pred2 = x < data_size.x and y < data_size.y and z < data_size.z;
 
     if (pred1 and pred2) {
         auto data_id      = x + y * data_leap.y + z * data_leap.z;
         auto anchor_id    = (x / 8) + (y / 8) * anchor_leap.y + (z / 8) * anchor_leap.z;
         anchor[anchor_id] = data[data_id];
     }
     __syncthreads();
 }
 
 template <typename T1, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
 __device__ void c_gather_anchor(T1* data, DIM3 data_size, STRIDE3 data_leap, T1* anchor, STRIDE3 anchor_leap)
 {
     auto ax =  BIX;//1 is block16 by anchor stride
     auto ay = BIY ;
     auto az = BIZ ;
 
     auto x = 16*ax;
     auto y = 16*ay;
     auto z = 16*az;
 
     bool pred1 = TIX < 1;//1 is num of anchor
     bool pred2 = x < data_size.x and y < data_size.y and z < data_size.z;
 
     if (pred1 and pred2) {
         auto data_id      = x + y * data_leap.y + z * data_leap.z;
         auto anchor_id    = ax + ay * anchor_leap.y + az * anchor_leap.z;
         anchor[anchor_id] = data[data_id];
         /*
         if(TIX == 7 and BIX == 12 and BIY == 12 and BIZ == 8){
             printf("anchor: %d, %d, %.2e, %.2e,%d,%d,%d,%d\n", anchor_id,data_id,anchor[anchor_id],data[data_id],data_leap.y,data_leap.z,anchor_leap.y,anchor_leap.z);
         }
         if(TIX == 0 and BIX == 13 and BIY == 13 and BIZ == 9){
             printf("13139anchor: %d, %d, %.2e, %.2e\n", anchor_id,data_id,anchor[anchor_id],data[data_id]);
         }*/
     }
     __syncthreads();
 }
 
 
 /*
  * use shmem, erroneous
 template <typename T1, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
 __device__ void c_gather_anchor(volatile T1 s_data[9][9][33], T1* anchor, STRIDE3 anchor_leap)
 {
     constexpr auto NUM_ITERS = 33 * 9 * 9 / LINEAR_BLOCK_SIZE + 1;  // 11 iterations
     for (auto i = 0; i < NUM_ITERS; i++) {
         auto _tix = i * LINEAR_BLOCK_SIZE + TIX;
 
         if (_tix < 33 * 9 * 9) {
             auto x = (_tix % 33);
             auto y = (_tix / 33) % 9;
             auto z = (_tix / 33) / 9;
 
             if (x % 8 == 0 and y % 8 == 0 and z % 8 == 0) {
                 auto aid = ((x / 8) + BIX * 4) +             //
                            ((y / 8) + BIY) * anchor_leap.y +  //
                            ((z / 8) + BIZ) * anchor_leap.z;   //
                 anchor[aid] = s_data[z][y][x];
             }
         }
     }
     __syncthreads();
 }
 */
 
 template <typename T1, typename T2 = T1, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
 __device__ void x_reset_scratch_17x17x17data(
     volatile T1 s_xdata[17][17][17],
     volatile T2 s_ectrl[17][17][17],
     T1*         anchor,       //
     DIM3        anchor_size,  //
     STRIDE3     anchor_leap)
 {
     for (auto _tix = TIX; _tix < BLOCK17 * BLOCK17 * BLOCK17; _tix += LINEAR_BLOCK_SIZE) {
         auto x = (_tix % BLOCK17);
         auto y = (_tix / BLOCK17) % BLOCK17;
         auto z = (_tix / BLOCK17) / BLOCK17;
 
         s_ectrl[z][y][x] = 0;  // TODO explicitly handle zero-padding
         /*****************************************************************************
          okay to use
          ******************************************************************************/
         if (x % 16 == 0 and y % 16 == 0 and z % 16 == 0) {
             s_xdata[z][y][x] = 0;
 
             auto ax = ((x / 16) + BIX );
             auto ay = ((y / 16) + BIY );
             auto az = ((z / 16) + BIZ );
 
             if (ax < anchor_size.x and ay < anchor_size.y and az < anchor_size.z)
                 s_xdata[z][y][x] = anchor[ax + ay * anchor_leap.y + az * anchor_leap.z];
             /*
             if(BIX == 11 and BIY == 12 and BIZ == 8){
                 printf("anchor: %d, %d, %d, %.2e\n", x, y,z,s_xdata[z][y][x]);
             if(BIX == 13 and BIY == 13 and BIZ == 9 and x==0 and y==0 and z==0)
                 printf("13139anchor: %d, %d, %d, %.2e\n", x, y,z,s_xdata[z][y][x]);
             */
         }
         /*****************************************************************************
          alternatively
          ******************************************************************************/
         // s_ectrl[z][y][x] = radius;
     }
 
     __syncthreads();
 }
 
 template <typename T1, typename T2, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
 __device__ void global2shmem_17x17x17data(T1* data, DIM3 data_size, STRIDE3 data_leap, volatile T2 s_data[17][17][17])
 {
     constexpr auto TOTAL = BLOCK17 * BLOCK17 * BLOCK17;
 
     for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
         auto x   = (_tix % BLOCK17);
         auto y   = (_tix / BLOCK17) % BLOCK17;
         auto z   = (_tix / BLOCK17) / BLOCK17;
         auto gx  = (x + BIX * BLOCK16);
         auto gy  = (y + BIY * BLOCK16);
         auto gz  = (z + BIZ * BLOCK16);
         auto gid = gx + gy * data_leap.y + gz * data_leap.z;
 
         if (gx < data_size.x and gy < data_size.y and gz < data_size.z) s_data[z][y][x] = data[gid];
 /*
         if(BIX == 7 and BIY == 47 and BIZ == 15 and x==10 and y==8 and z==4){
             printf("g2s1084 %d %d %d %d %.2e %.2e \n",gx,gy,gz,gid,s_data[z][y][x],data[gid]);
         }
 
         if(BIX == 7 and BIY == 47 and BIZ == 15 and x==10 and y==4 and z==8){
             printf("g2s1048 %d %d %d %d %.2e %.2e \n",gx,gy,gz,gid,s_data[z][y][x],data[gid]);
         }*/
     }
     __syncthreads();
 }
 
 template <typename T1, typename T2, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
 __device__ void global2shmem_profiling_16x16x16data(T1* data, DIM3 data_size, STRIDE3 data_leap, volatile T2 s_data[16][16][16])
 {
     constexpr auto TOTAL = 16 * 16 * 16;
 
     for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
         auto x   = (_tix % 16);
         auto y   = (_tix / 16) % 16;
         auto z   = (_tix / 16) / 16;
         auto gx_1=x/4;
         auto gx_2=x%4;
         auto gy_1=y/4;
         auto gy_2=y%4;
         auto gz_1=z/4;
         auto gz_2=z%4;
         auto gx=(data_size.x/4)*gx_1+gx_2;
         auto gy=(data_size.y/4)*gy_1+gy_2;
         auto gz=(data_size.z/4)*gz_1+gz_2;
 
         auto gid = gx + gy * data_leap.y + gz * data_leap.z;
 
         if (gx < data_size.x and gy < data_size.y and gz < data_size.z) s_data[z][y][x] = data[gid];
 /*
         if(BIX == 7 and BIY == 47 and BIZ == 15 and x==10 and y==8 and z==4){
             printf("g2s1084 %d %d %d %d %.2e %.2e \n",gx,gy,gz,gid,s_data[z][y][x],data[gid]);
         }
 
         if(BIX == 7 and BIY == 47 and BIZ == 15 and x==10 and y==4 and z==8){
             printf("g2s1048 %d %d %d %d %.2e %.2e \n",gx,gy,gz,gid,s_data[z][y][x],data[gid]);
         }*/
     }
     __syncthreads();
 }
 
 template <typename T1, typename T2, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
 __device__ void global2shmem_profiling_data_2(T1* data, DIM3 data_size, STRIDE3 data_leap, volatile T2 s_data[64], volatile T2 s_nx[64][4], volatile T2 s_ny[64][4], volatile T2 s_nz[64][4])
 {
     constexpr auto TOTAL = 64 * 4;
     int factors[4]={-3,-1,1,3};
     for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
         auto offset   = (_tix % 4);
         auto idx = _tix /4;
         auto x = idx%4;
         auto y   = (idx / 4) % 4;
         auto z   = (idx / 4) / 4;
         auto gx=(data_size.x/4)*x+data_size.x/8;
         auto gy=(data_size.y/4)*y+data_size.y/8;
         auto gz=(data_size.z/4)*z+data_size.z/8;
 
         auto gid = gx + gy * data_leap.y + gz * data_leap.z;
 
         if (gx>=3 and gy>=3 and gz>=3 and gx+3 < data_size.x and gy+3 < data_size.y and gz+3 < data_size.z) {
             s_data[idx] = data[gid];
             
             auto factor=factors[offset];
             s_nx[idx][offset]=data[gid+factor];
             s_ny[idx][offset]=data[gid+factor*data_leap.y];
             s_nz[idx][offset]=data[gid+factor*data_leap.z];
            
         }
 /*
         if(BIX == 7 and BIY == 47 and BIZ == 15 and x==10 and y==8 and z==4){
             printf("g2s1084 %d %d %d %d %.2e %.2e \n",gx,gy,gz,gid,s_data[z][y][x],data[gid]);
         }
 
         if(BIX == 7 and BIY == 47 and BIZ == 15 and x==10 and y==4 and z==8){
             printf("g2s1048 %d %d %d %d %.2e %.2e \n",gx,gy,gz,gid,s_data[z][y][x],data[gid]);
         }*/
     }
     __syncthreads();
 }
 
 template <typename T = float, typename E = u4, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
 __device__ void global2shmem_fuse(E* ectrl, dim3 ectrl_size, dim3 ectrl_leap, T* scattered_outlier, volatile T s_ectrl[17][17][17],volatile STRIDE3 grid_leaps[5],volatile size_t prefix_nums[5])
 {
     constexpr auto TOTAL = BLOCK17 * BLOCK17 * BLOCK17;
 
     for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
         auto x   = (_tix % BLOCK17);
         auto y   = (_tix / BLOCK17) % BLOCK17;
         auto z   = (_tix / BLOCK17) / BLOCK17;
         auto gx  = (x + BIX * BLOCK16);
         auto gy  = (y + BIY * BLOCK16);
         auto gz  = (z + BIZ * BLOCK16);
         //auto gid = gx + gy * ectrl_leap.y + gz * ectrl_leap.z;
 
         if (gx < ectrl_size.x and gy < ectrl_size.y and gz < ectrl_size.z) {
             //todo: pre-compute the leaps and their halves
 
             int level = 0;
             auto data_gid = gx + gy * ectrl_leap.y + gz * ectrl_leap.z;
             while(gx%2==0 and gy%2==0 and gz%2==0 and level <= 3){
                 
                 gx=gx>>1;
                 gy=gy>>1;
                 gz=gz>>1;
                 level ++;
             }
             auto gid = gx + gy*grid_leaps[level].y+gz*grid_leaps[level].z;
 
             if(level < 4){//non-anchor
                 gid += prefix_nums[level]-((gz+1)>>1)*grid_leaps[level+1].z-(gz%2==0)*((gy+1)>>1)*grid_leaps[level+1].y-(gz%2==0 && gy%2==0)*((gx+1)>>1);
             }
 
             s_ectrl[z][y][x] = static_cast<T>(ectrl[gid]) + scattered_outlier[data_gid];
         }
     }
     __syncthreads();
 }
 
 // dram_outlier should be the same in type with shared memory buf
 template <typename T1, typename T2, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
 __device__ void
 shmem2global_17x17x17data(volatile T1 s_buf[17][17][17], T2* dram_buf, DIM3 buf_size, STRIDE3 buf_leap)
 {
     auto x_size=BLOCK16+(BIX==GDX-1);
     auto y_size=BLOCK16+(BIY==GDY-1);
     auto z_size=BLOCK16+(BIZ==GDZ-1);
     //constexpr auto TOTAL = 32 * 8 * 8;
     auto TOTAL = x_size * y_size * z_size;
 
     for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
         auto x   = (_tix % x_size);
         auto y   = (_tix / x_size) % y_size;
         auto z   = (_tix / x_size) / y_size;
         auto gx  = (x + BIX * BLOCK16);
         auto gy  = (y + BIY * BLOCK16);
         auto gz  = (z + BIZ * BLOCK16);
         auto gid = gx + gy * buf_leap.y + gz * buf_leap.z;
 
         if (gx < buf_size.x and gy < buf_size.y and gz < buf_size.z) dram_buf[gid] = s_buf[z][y][x];
         /*
         if(BIX == 7 and BIY == 47 and BIZ == 15 and x==10 and y==8 and z==4){
             printf("s2g1084 %d %d %d %d %.2e %.2e \n",gx,gy,gz,gid,s_buf[z][y][x],dram_buf[gid]);
         }
 
         if(BIX == 7 and BIY == 47 and BIZ == 15 and x==10 and y==4 and z==8){
             printf("s2g1048 %d %d %d %d %.2e %.2e \n",gx,gy,gz,gid,s_buf[z][y][x],dram_buf[gid]);
         }*/
     }
     __syncthreads();
 }
 
 
 // dram_outlier should be the same in type with shared memory buf
 template <typename T1, typename T2, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
 __device__ void
 shmem2global_17x17x17data_with_compaction(volatile T1 s_buf[17][17][17], T2* dram_buf, DIM3 buf_size, STRIDE3 buf_leap, int radius,volatile STRIDE3 grid_leaps[5],volatile size_t prefix_nums[5], T1* dram_compactval = nullptr, uint32_t* dram_compactidx = nullptr, uint32_t* dram_compactnum = nullptr)
 {
     auto x_size = BLOCK16 + (BIX == GDX-1);
     auto y_size = BLOCK16 + (BIY == GDY-1);
     auto z_size = BLOCK16 + (BIZ == GDZ-1);
     auto TOTAL = x_size * y_size * z_size;
 
     for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
         auto x   = (_tix % x_size);
         auto y   = (_tix / x_size) % y_size;
         auto z   = (_tix / x_size) / y_size;
         auto gx  = (x + BIX * BLOCK16);
         auto gy  = (y + BIY * BLOCK16);
         auto gz  = (z + BIZ * BLOCK16);
         //auto gid = gx + gy * buf_leap.y + gz * buf_leap.z;
 
         auto candidate = s_buf[z][y][x];
         bool quantizable = (candidate >= 0) and (candidate < 2*radius);
 
         if (gx < buf_size.x and gy < buf_size.y and gz < buf_size.z) {
 
             
 
             if (not quantizable) {
                 auto data_gid = gx + gy * buf_leap.y + gz * buf_leap.z;
                 auto cur_idx = atomicAdd(dram_compactnum, 1);
                 dram_compactidx[cur_idx] = data_gid;
                 dram_compactval[cur_idx] = candidate;
             }
             int level = 0;
             //todo: pre-compute the leaps and their halves
             while(gx%2==0 and gy%2==0 and gz%2==0 and level <= 3){
                 
                 gx=gx>>1;
                 gy=gy>>1;
                 gz=gz>>1;
                 level ++;
             }
             auto gid = gx + gy*grid_leaps[level].y+gz*grid_leaps[level].z;
 
             if(level < 4){//non-anchor
                 gid += prefix_nums[level]-((gz+1)>>1)*grid_leaps[level+1].z-(gz%2==0)*((gy+1)>>1)*grid_leaps[level+1].y-(gz%2==0 && gy%2==0)*((gx+1)>>1);
             }
 
             // TODO this is for algorithmic demo by reading from shmem
             // For performance purpose, it can be inlined in quantization
             dram_buf[gid] = quantizable * static_cast<T2>(candidate);
 
 
         }
     }
     __syncthreads();
 }
 
 
 
 template <
     typename T1,
     typename T2,
     typename FP,
     typename LAMBDAX,
     typename LAMBDAY,
     typename LAMBDAZ,
     bool BLUE,
     bool YELLOW,
     bool HOLLOW,
     int  LINEAR_BLOCK_SIZE,
     int  BLOCK_DIMX,
     int  BLOCK_DIMY,
     bool COARSEN,
     int  BLOCK_DIMZ,
     bool BORDER_INCLUSIVE,
     bool WORKFLOW>
 __forceinline__ __device__ void interpolate_stage(
     volatile T1 s_data[17][17][17],
     volatile T2 s_ectrl[17][17][17],
     DIM3    data_size,
     LAMBDAX     xmap,
     LAMBDAY     ymap,
     LAMBDAZ     zmap,
     int         unit,
     FP          eb_r,
     FP          ebx2,
     int         radius,
     bool interpolator)
 {
     static_assert(BLOCK_DIMX * BLOCK_DIMY * (COARSEN ? 1 : BLOCK_DIMZ) <= 384, "block oversized");
     static_assert((BLUE or YELLOW or HOLLOW) == true, "must be one hot");
     static_assert((BLUE and YELLOW) == false, "must be only one hot (1)");
     static_assert((BLUE and YELLOW) == false, "must be only one hot (2)");
     static_assert((YELLOW and HOLLOW) == false, "must be only one hot (3)");
 
     auto run = [&](auto x, auto y, auto z) {
 
         
 
         if (xyz17x17x17_predicate<BORDER_INCLUSIVE>(x, y, z,data_size)) {
             T1 pred = 0;
 
             //if(BIX == 7 and BIY == 47 and BIZ == 15 and unit==4 and (CONSTEXPR (YELLOW)) )
             //    printf("%d %d %d\n",x,y,z);
             /*
              if(BIX == 7 and BIY == 47 and BIZ == 15 and unit==4 and x==4 and y==4 and z==4)
                         printf("444 %.2e %.2e \n",s_data[z - unit][y][x],s_data[z + unit][y][x]);
 
             if(BIX == 7 and BIY == 47 and BIZ == 15 and unit==4 and x==4 and y==4 and z==0)
                         printf("440 %.2e %.2e \n",s_data[z][y - unit][x],s_data[z][y + unit][x]);
             if(BIX == 7 and BIY == 47 and BIZ == 15 and unit==4 and x==4 and y==8 and z==0)
                         printf("480 %.2e %.2e \n",s_data[z][y ][x- unit],s_data[z][y ][x+ unit]);*/
                   //  }
             auto global_x=BIX*BLOCK16+x, global_y=BIY*BLOCK16+y, global_z=BIZ*BLOCK16+z;
             /*
             int interpolation_coeff_set1[2]={-1,-3};
             int interpolation_coeff_set2[2]={9,23};
             int interpolation_coeff_set3[2]={16,40};
             auto a=interpolation_coeff_set1[interpolator];
             auto b=interpolation_coeff_set2[interpolator];
             auto c=interpolation_coeff_set3[interpolator];
             */
             if(interpolator==0){
                 if CONSTEXPR (BLUE) {  //
 
                     if(BIZ!=GDZ-1){
 
                         if(z>=3*unit and z+3*unit<=BLOCK16  )
                             pred = (-s_data[z - 3*unit][y][x]+9*s_data[z - unit][y][x] + 9*s_data[z + unit][y][x]-s_data[z + 3*unit][y][x]) / 16;
                         else if (z+3*unit<=BLOCK16)
                             pred = (3*s_data[z - unit][y][x] + 6*s_data[z + unit][y][x]-s_data[z + 3*unit][y][x]) / 8;
                         else if (z>=3*unit)
                             pred = (-s_data[z - 3*unit][y][x]+6*s_data[z - unit][y][x] + 3*s_data[z + unit][y][x]) / 8;
 
                         else
                             pred = (s_data[z - unit][y][x] + s_data[z + unit][y][x]) / 2;
                     }
                     else{
                         if(z>=3*unit){
                             if(z+3*unit<=BLOCK16 and global_z+3*unit<data_size.z)
                                 pred = (-s_data[z - 3*unit][y][x]+9*s_data[z - unit][y][x] + 9*s_data[z + unit][y][x]-s_data[z + 3*unit][y][x]) / 16;
                             else if (global_z+unit<data_size.z)
                                 pred = (-s_data[z - 3*unit][y][x]+6*s_data[z - unit][y][x] + 3*s_data[z + unit][y][x]) / 8;
                             else
                                 pred=s_data[z - unit][y][x];
 
                         }
                         else{
                             if(z+3*unit<=BLOCK16 and global_z+3*unit<data_size.z)
                                 pred = (3*s_data[z - unit][y][x] + 6*s_data[z + unit][y][x]-s_data[z + 3*unit][y][x]) / 8;
                             else if (global_z+unit<data_size.z)
                                 pred = (s_data[z - unit][y][x] + s_data[z + unit][y][x]) / 2;
                             else
                                 pred=s_data[z - unit][y][x];
                         } 
                     }
                 }
                 if CONSTEXPR (YELLOW) {  //
                    // if(BIX == 5 and BIY == 22 and BIZ == 6 and unit==1 and x==29 and y==7 and z==0){
                    //     printf("%.2e %.2e %.2e %.2e\n",s_data[z ][y- 3*unit][x],s_data[z ][y- unit][x],s_data[z ][y+ unit][x]);
                   //  }
                     if(BIY!=GDY-1){
                         if(y>=3*unit and y+3*unit<=BLOCK16 )
                             pred = (-s_data[z ][y- 3*unit][x]+9*s_data[z ][y- unit][x] + 9*s_data[z ][y+ unit][x]-s_data[z][y + 3*unit][x]) / 16;
                         else if (y+3*unit<=BLOCK16)
                             pred = (3*s_data[z ][y - unit][x] + 6*s_data[z][y + unit][x]-s_data[z][y + 3*unit][x]) / 8;
                         else if (y>=3*unit)
                             pred = (-s_data[z ][y- 3*unit][x]+6*s_data[z][y - unit][x] + 3*s_data[z][y + unit][x]) / 8;
                         else
                             pred = (s_data[z][y - unit][x] + s_data[z][y + unit][x]) / 2;
                     }
                     else{
                         if(y>=3*unit){
                             if(y+3*unit<=BLOCK16 and global_y+3*unit<data_size.y)
                                 pred = (-s_data[z ][y- 3*unit][x]+9*s_data[z][y - unit][x] + 9*s_data[z ][y+ unit][x]-s_data[z ][y+ 3*unit][x]) / 16;
                             else if (global_y+unit<data_size.y)
                                 pred = (-s_data[z ][y- 3*unit][x]+6*s_data[z ][y- unit][x] + 3*s_data[z ][y+ unit][x]) / 8;
                             else
                                 pred=s_data[z ][y- unit][x];
 
                         }
                         else{
                             if(y+3*unit<=BLOCK16 and global_y+3*unit<data_size.y)
                                 pred = (3*s_data[z][y - unit][x] + 6*s_data[z ][y+ unit][x]-s_data[z][y + 3*unit][x]) / 8;
                             else if (global_y+unit<data_size.y)
                                 pred = (s_data[z ][y- unit][x] + s_data[z][y + unit][x]) / 2;
                             else
                                 pred=s_data[z ][y- unit][x];
                         } 
                     }
                 }
 
                 if CONSTEXPR (HOLLOW) {  //
                     //if(BIX == 5 and BIY == 22 and BIZ == 6 and unit==1)
                     //    printf("%d %d %d\n",x,y,z);
                     if(BIX!=GDX-1){
                         if(x>=3*unit and x+3*unit<=BLOCK16 )
                             pred = (-s_data[z ][y][x- 3*unit]+9*s_data[z ][y][x- unit] + 9*s_data[z ][y][x+ unit]-s_data[z ][y][x + 3*unit]) / 16;
                         else if (x+3*unit<=BLOCK16)
                             pred = (3*s_data[z ][y][x- unit] + 6*s_data[z ][y][x + unit]-s_data[z][y][x + 3*unit]) / 8;
                         else if (x>=3*unit)
                             pred = (-s_data[z][y][x - 3*unit]+6*s_data[z][y][x - unit] + 3*s_data[z ][y][x + unit]) / 8;
                         else
                             pred = (s_data[z][y][x - unit] + s_data[z][y][x + unit]) / 2;
                     }
                     else{
                         if(x>=3*unit){
                             if(x+3*unit<=BLOCK16 and global_x+3*unit<data_size.x)
                                 pred = (-s_data[z ][y][x- 3*unit]+9*s_data[z][y ][x- unit] + 9*s_data[z ][y][x+ unit]-s_data[z ][y][x+ 3*unit]) / 16;
                             else if (global_x+unit<data_size.x)
                                 pred = (-s_data[z ][y][x- 3*unit]+6*s_data[z ][y][x- unit] + 3*s_data[z ][y][x+ unit]) / 8;
                             else
                                 pred=s_data[z ][y][x- unit];
 
                         }
                         else{
                             if(x+3*unit<=BLOCK16 and global_x+3*unit<data_size.x)
                                 pred = (3*s_data[z][y ][x- unit] + 6*s_data[z ][y][x+ unit]-s_data[z][y ][x+ 3*unit]) / 8;
                             else if (global_x+unit<data_size.x)
                                 pred = (s_data[z ][y][x- unit] + s_data[z][y ][x+ unit]) / 2;
                             else
                                 pred=s_data[z ][y][x- unit];
                         } 
                     }
                 }
 
             }
             else{
                 if CONSTEXPR (BLUE) {  //
 
                     if(BIZ!=GDZ-1){
 
                         if(z>=3*unit and z+3*unit<=BLOCK16  )
                             pred = (-3*s_data[z - 3*unit][y][x]+23*s_data[z - unit][y][x] + 23*s_data[z + unit][y][x]-3*s_data[z + 3*unit][y][x]) / 40;
                         else if (z+3*unit<=BLOCK16)
                             pred = (3*s_data[z - unit][y][x] + 6*s_data[z + unit][y][x]-s_data[z + 3*unit][y][x]) / 8;
                         else if (z>=3*unit)
                             pred = (-s_data[z - 3*unit][y][x]+6*s_data[z - unit][y][x] + 3*s_data[z + unit][y][x]) / 8;
 
                         else
                             pred = (s_data[z - unit][y][x] + s_data[z + unit][y][x]) / 2;
                     }
                     else{
                         if(z>=3*unit){
                             if(z+3*unit<=BLOCK16 and global_z+3*unit<data_size.z)
                                 pred = (-3*s_data[z - 3*unit][y][x]+23*s_data[z - unit][y][x] + 23*s_data[z + unit][y][x]-3*s_data[z + 3*unit][y][x]) / 40;
                             else if (global_z+unit<data_size.z)
                                 pred = (-s_data[z - 3*unit][y][x]+6*s_data[z - unit][y][x] + 3*s_data[z + unit][y][x]) / 8;
                             else
                                 pred=s_data[z - unit][y][x];
 
                         }
                         else{
                             if(z+3*unit<=BLOCK16 and global_z+3*unit<data_size.z)
                                 pred = (3*s_data[z - unit][y][x] + 6*s_data[z + unit][y][x]-s_data[z + 3*unit][y][x]) / 8;
                             else if (global_z+unit<data_size.z)
                                 pred = (s_data[z - unit][y][x] + s_data[z + unit][y][x]) / 2;
                             else
                                 pred=s_data[z - unit][y][x];
                         } 
                     }
                 }
                 if CONSTEXPR (YELLOW) {  //
                    // if(BIX == 5 and BIY == 22 and BIZ == 6 and unit==1 and x==29 and y==7 and z==0){
                    //     printf("%.2e %.2e %.2e %.2e\n",s_data[z ][y- 3*unit][x],s_data[z ][y- unit][x],s_data[z ][y+ unit][x]);
                   //  }
                     if(BIY!=GDY-1){
                         if(y>=3*unit and y+3*unit<=BLOCK16 )
                             pred = (-3*s_data[z ][y- 3*unit][x]+23*s_data[z ][y- unit][x] + 23*s_data[z ][y+ unit][x]-3*s_data[z][y + 3*unit][x]) / 40;
                         else if (y+3*unit<=BLOCK16)
                             pred = (3*s_data[z ][y - unit][x] + 6*s_data[z][y + unit][x]-s_data[z][y + 3*unit][x]) / 8;
                         else if (y>=3*unit)
                             pred = (-s_data[z ][y- 3*unit][x]+6*s_data[z][y - unit][x] + 3*s_data[z][y + unit][x]) / 8;
                         else
                             pred = (s_data[z][y - unit][x] + s_data[z][y + unit][x]) / 2;
                     }
                     else{
                         if(y>=3*unit){
                             if(y+3*unit<=BLOCK16 and global_y+3*unit<data_size.y)
                                 pred = (-3*s_data[z ][y- 3*unit][x]+23*s_data[z][y - unit][x] + 23*s_data[z ][y+ unit][x]-3*s_data[z ][y+ 3*unit][x]) / 40;
                             else if (global_y+unit<data_size.y)
                                 pred = (-s_data[z ][y- 3*unit][x]+6*s_data[z ][y- unit][x] + 3*s_data[z ][y+ unit][x]) / 8;
                             else
                                 pred=s_data[z ][y- unit][x];
 
                         }
                         else{
                             if(y+3*unit<=BLOCK16 and global_y+3*unit<data_size.y)
                                 pred = (3*s_data[z][y - unit][x] + 6*s_data[z ][y+ unit][x]-s_data[z][y + 3*unit][x]) / 8;
                             else if (global_y+unit<data_size.y)
                                 pred = (s_data[z ][y- unit][x] + s_data[z][y + unit][x]) / 2;
                             else
                                 pred=s_data[z ][y- unit][x];
                         } 
                     }
                 }
 
                 if CONSTEXPR (HOLLOW) {  //
                     //if(BIX == 5 and BIY == 22 and BIZ == 6 and unit==1)
                     //    printf("%d %d %d\n",x,y,z);
                     if(BIX!=GDX-1){
                         if(x>=3*unit and x+3*unit<=BLOCK16 )
                             pred = (-3*s_data[z ][y][x- 3*unit]+23*s_data[z ][y][x- unit] + 23*s_data[z ][y][x+ unit]-3*s_data[z ][y][x + 3*unit]) / 40;
                         else if (x+3*unit<=BLOCK16)
                             pred = (3*s_data[z ][y][x- unit] + 6*s_data[z ][y][x + unit]-s_data[z][y][x + 3*unit]) / 8;
                         else if (x>=3*unit)
                             pred = (-s_data[z][y][x - 3*unit]+6*s_data[z][y][x - unit] + 3*s_data[z ][y][x + unit]) / 8;
                         else
                             pred = (s_data[z][y][x - unit] + s_data[z][y][x + unit]) / 2;
                     }
                     else{
                         if(x>=3*unit){
                             if(x+3*unit<=BLOCK16 and global_x+3*unit<data_size.x)
                                 pred = (-3*s_data[z ][y][x- 3*unit]+23*s_data[z][y ][x- unit] + 23*s_data[z ][y][x+ unit]-3*s_data[z ][y][x+ 3*unit]) / 40;
                             else if (global_x+unit<data_size.x)
                                 pred = (-s_data[z ][y][x- 3*unit]+6*s_data[z ][y][x- unit] + 3*s_data[z ][y][x+ unit]) / 8;
                             else
                                 pred=s_data[z ][y][x- unit];
 
                         }
                         else{
                             if(x+3*unit<=BLOCK16 and global_x+3*unit<data_size.x)
                                 pred = (3*s_data[z][y ][x- unit] + 6*s_data[z ][y][x+ unit]-s_data[z][y ][x+ 3*unit]) / 8;
                             else if (global_x+unit<data_size.x)
                                 pred = (s_data[z ][y][x- unit] + s_data[z][y ][x+ unit]) / 2;
                             else
                                 pred=s_data[z ][y][x- unit];
                         } 
                     }
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
                 /*
                  if(BIX == 12 and BIY == 12 and BIZ == 8 and unit==4 and x==0 and y==0 and z==4)
                         printf("004pred %.2e %.2e %.2e %.2e %.2e %.2e\n",pred,code,s_data[z][y][x],s_data[0][0][0],s_data[0][0][8],s_data[0][0][16]);
                     if(BIX == 12 and BIY == 12 and BIZ == 8 and unit==4 and x==8 and y==8 and z==4)
                         printf("884pred %.2e %.2e %.2e %.2e %.2e %.2e\n",pred,code,s_data[z][y][x],s_data[8][8][0],s_data[8][8][8],s_data[8][8][16]);
                   */      
                // if(fabs(pred)>=3)
                //     printf("%d %d %d %d %d %d %d %d %d %d %.2e %.2e %.2e\n",unit,CONSTEXPR (BLUE),CONSTEXPR (YELLOW),CONSTEXPR (HOLLOW),BIX,BIY,BIZ,x,y,z,pred,code,s_data[z][y][x]);
               
                 s_data[z][y][x]  = pred + (code - radius) * ebx2;
                 
 
             }
             else {  // TODO == DECOMPRESSS and static_assert
                 auto code       = s_ectrl[z][y][x];
                 s_data[z][y][x] = pred + (code - radius) * ebx2;
                 /*
                 if(BIX == 12 and BIY == 12 and BIZ == 8 and unit==4 and x==0 and y==0 and z==4)
                         printf("004pred %.2e %.2e %.2e %.2e %.2e %.2e\n",pred,code,s_data[z][y][x],s_data[0][0][0],s_data[0][0][8],s_data[0][0][16]);
                     if(BIX == 12 and BIY == 12 and BIZ == 8 and unit==4 and x==8 and y==8 and z==4)
                         printf("884pred %.2e %.2e %.2e %.2e %.2e %.2e\n",pred,code,s_data[z][y][x],s_data[8][8][0],s_data[8][8][8],s_data[8][8][16]);
                         
                 */
                 //if(BIX == 4 and BIY == 20 and BIZ == 20 and unit==1 and CONSTEXPR (BLUE)){
                //     if(fabs(s_data[z][y][x])>=3)
 
               //      printf("%d %d %d %d %d %d %d %d %d %d %.2e %.2e %.2e\n",unit,CONSTEXPR (BLUE),CONSTEXPR (YELLOW),CONSTEXPR (HOLLOW),BIX,BIY,BIZ,x,y,z,pred,code,s_data[z][y][x]);
                // }
             }
         }
     };
     // -------------------------------------------------------------------------------- //
 
     if CONSTEXPR (COARSEN) {
         constexpr auto TOTAL = BLOCK_DIMX * BLOCK_DIMY * BLOCK_DIMZ;
         //if( BLOCK_DIMX *BLOCK_DIMY<= LINEAR_BLOCK_SIZE){
             for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
                 auto itix = (_tix % BLOCK_DIMX);
                 auto itiy = (_tix / BLOCK_DIMX) % BLOCK_DIMY;
                 auto itiz = (_tix / BLOCK_DIMX) / BLOCK_DIMY;
                 auto x    = xmap(itix, unit);
                 auto y    = ymap(itiy, unit);
                 auto z    = zmap(itiz, unit);
                 
                 run(x, y, z);
             }
         //}
         //may have bug    
         /*
         else{
             for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
                 auto itix = (_tix % BLOCK_DIMX);
                 auto itiz = (_tix / BLOCK_DIMX) % BLOCK_DIMZ;
                 auto itiy = (_tix / BLOCK_DIMX) / BLOCK_DIMZ;
                 auto x    = xmap(itix, unit);
                 auto y    = ymap(itiy, unit);
                 auto z    = zmap(itiz, unit);
                 run(x, y, z);
             }
         }*/
         //may have bug  end
         
     }
     else {
         auto itix = (TIX % BLOCK_DIMX);
         auto itiy = (TIX / BLOCK_DIMX) % BLOCK_DIMY;
         auto itiz = (TIX / BLOCK_DIMX) / BLOCK_DIMY;
         auto x    = xmap(itix, unit);
         auto y    = ymap(itiy, unit);
         auto z    = zmap(itiz, unit);
 
         
 
      //   printf("%d %d %d\n", x,y,z);
         run(x, y, z);
     }
     __syncthreads();
 }
 
 template <
     typename T1,
     typename T2,
     typename FP,
     //typename LAMBDAX,
     //typename LAMBDAY,
     //typename LAMBDAZ,
     typename LAMBDA,
     bool LINE,
     bool FACE,
     bool CUBE,
     int  LINEAR_BLOCK_SIZE,
     int NUM_ELE,
     //int  BLOCK_DIMX,
     //int  BLOCK_DIMY,
     bool COARSEN,
     //int  BLOCK_DIMZ,
     bool BORDER_INCLUSIVE,
     bool WORKFLOW,
     typename INTERP>
 __forceinline__ __device__ void interpolate_stage_md(
     volatile T1 s_data[17][17][17],
     volatile T2 s_ectrl[17][17][17],
     DIM3    data_size,
     //LAMBDAX     xmap,
     //LAMBDAY     ymap,
     //LAMBDAZ     zmap,
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
 
         
 
         if (xyz17x17x17_predicate<BORDER_INCLUSIVE>(x, y, z,data_size)) {
             T1 pred = 0;
 
             //if(BIX == 23 and BIY == 23 and BIZ == 15 and unit==2)
             //    printf("%d %d %d\n",x,y,z);
             /*
              if(BIX == 7 and BIY == 47 and BIZ == 15 and unit==4 and x==4 and y==4 and z==4)
                         printf("444 %.2e %.2e \n",s_data[z - unit][y][x],s_data[z + unit][y][x]);
 
             if(BIX == 7 and BIY == 47 and BIZ == 15 and unit==4 and x==4 and y==4 and z==0)
                         printf("440 %.2e %.2e \n",s_data[z][y - unit][x],s_data[z][y + unit][x]);
             if(BIX == 7 and BIY == 47 and BIZ == 15 and unit==4 and x==4 and y==8 and z==0)
                         printf("480 %.2e %.2e \n",s_data[z][y ][x- unit],s_data[z][y ][x+ unit]);*/
                   //  }
             auto global_x=BIX*BLOCK16+x, global_y=BIY*BLOCK16+y, global_z=BIZ*BLOCK16+z;
             /*
             int interpolation_coeff_set1[2]={-1,-3};
             int interpolation_coeff_set2[2]={9,23};
             int interpolation_coeff_set3[2]={16,40};
             auto a=interpolation_coeff_set1[interpolator];
             auto b=interpolation_coeff_set2[interpolator];
             auto c=interpolation_coeff_set3[interpolator];
             */
            
             if CONSTEXPR (LINE) {  //
                 //bool I_X = x&1; 
                 bool I_Y = (y % (2*unit) )> 0; 
                 bool I_Z = (z % (2*unit) )> 0; 
                 if (I_Z){
                     //assert(x&1==0 and y&1==0);
 
                     if(BIZ!=GDZ-1){
 
                         if(z>=3*unit and z+3*unit<=BLOCK16  )
                             pred = cubic_interpolator(s_data[z - 3*unit][y][x],s_data[z - unit][y][x],s_data[z + unit][y][x],s_data[z + 3*unit][y][x]);
                         else if (z+3*unit<=BLOCK16)
                             pred = (3*s_data[z - unit][y][x] + 6*s_data[z + unit][y][x]-s_data[z + 3*unit][y][x]) / 8;
                         else if (z>=3*unit)
                             pred = (-s_data[z - 3*unit][y][x]+6*s_data[z - unit][y][x] + 3*s_data[z + unit][y][x]) / 8;
 
                         else
                             pred = (s_data[z - unit][y][x] + s_data[z + unit][y][x]) / 2;
                     }
                     else{
                         if(z>=3*unit){
                             if(z+3*unit<=BLOCK16 and global_z+3*unit<data_size.z)
                                 pred = cubic_interpolator(s_data[z - 3*unit][y][x],s_data[z - unit][y][x] ,s_data[z + unit][y][x],s_data[z + 3*unit][y][x]);
                             else if (global_z+unit<data_size.z)
                                 pred = (-s_data[z - 3*unit][y][x]+6*s_data[z - unit][y][x] + 3*s_data[z + unit][y][x]) / 8;
                             else
                                 pred=s_data[z - unit][y][x];
 
                         }
                         else{
                             if(z+3*unit<=BLOCK16 and global_z+3*unit<data_size.z)
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
                         if(y>=3*unit and y+3*unit<=BLOCK16 )
                             pred = cubic_interpolator(s_data[z ][y- 3*unit][x],s_data[z ][y- unit][x] ,s_data[z ][y+ unit][x],s_data[z][y + 3*unit][x]) ;
                         else if (y+3*unit<=BLOCK16)
                             pred = (3*s_data[z ][y - unit][x] + 6*s_data[z][y + unit][x]-s_data[z][y + 3*unit][x]) / 8;
                         else if (y>=3*unit)
                             pred = (-s_data[z ][y- 3*unit][x]+6*s_data[z][y - unit][x] + 3*s_data[z][y + unit][x]) / 8;
                         else
                             pred = (s_data[z][y - unit][x] + s_data[z][y + unit][x]) / 2;
                     }
                     else{
                         if(y>=3*unit){
                             if(y+3*unit<=BLOCK16 and global_y+3*unit<data_size.y)
                                 pred = cubic_interpolator(s_data[z ][y- 3*unit][x],s_data[z][y - unit][x],s_data[z ][y+ unit][x],s_data[z ][y+ 3*unit][x]);
                             else if (global_y+unit<data_size.y)
                                 pred = (-s_data[z ][y- 3*unit][x]+6*s_data[z ][y- unit][x] + 3*s_data[z ][y+ unit][x]) / 8;
                             else
                                 pred=s_data[z ][y- unit][x];
 
                         }
                         else{
                             if(y+3*unit<=BLOCK16 and global_y+3*unit<data_size.y)
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
                         if(x>=3*unit and x+3*unit<=BLOCK16 )
                             pred = cubic_interpolator(s_data[z ][y][x- 3*unit],s_data[z ][y][x- unit],s_data[z ][y][x+ unit],s_data[z ][y][x + 3*unit]);
                         else if (x+3*unit<=BLOCK16)
                             pred = (3*s_data[z ][y][x- unit] + 6*s_data[z ][y][x + unit]-s_data[z][y][x + 3*unit]) / 8;
                         else if (x>=3*unit)
                             pred = (-s_data[z][y][x - 3*unit]+6*s_data[z][y][x - unit] + 3*s_data[z ][y][x + unit]) / 8;
                         else
                             pred = (s_data[z][y][x - unit] + s_data[z][y][x + unit]) / 2;
                     }
                     else{
                         if(x>=3*unit){
                             if(x+3*unit<=BLOCK16 and global_x+3*unit<data_size.x)
                                 pred = cubic_interpolator(s_data[z ][y][x- 3*unit],s_data[z][y ][x- unit],s_data[z ][y][x+ unit],s_data[z ][y][x+ 3*unit]);
                             else if (global_x+unit<data_size.x)
                                 pred = (-s_data[z ][y][x- 3*unit]+6*s_data[z ][y][x- unit] + 3*s_data[z ][y][x+ unit]) / 8;
                             else
                                 pred=s_data[z ][y][x- unit];
 
                         }
                         else{
                             if(x+3*unit<=BLOCK16 and global_x+3*unit<data_size.x)
                                 pred = (3*s_data[z][y ][x- unit] + 6*s_data[z ][y][x+ unit]-s_data[z][y ][x+ 3*unit]) / 8;
                             else if (global_x+unit<data_size.x)
                                 pred = (s_data[z ][y][x- unit] + s_data[z][y ][x+ unit]) / 2;
                             else
                                 pred=s_data[z ][y][x- unit];
                         } 
                     }
 
                 }
             }
             auto get_interp_order = [&](auto x, auto BI, auto GD, auto gx, auto gs){
                 int b = x >= 3*unit ? 3 : 1;
                 int f = 0;
                 if(x+3*unit<=BLOCK16 and (BI != GD-1 or gx+3*unit < gs) )
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
                // if(BIX == 5 and BIY == 22 and BIZ == 6 and unit==1 and x==29 and y==7 and z==0){
                //     printf("%.2e %.2e %.2e %.2e\n",s_data[z ][y- 3*unit][x],s_data[z ][y- unit][x],s_data[z ][y+ unit][x]);
               //  }
 
                 bool I_YZ = (x % (2*unit) ) == 0;
                 bool I_XZ = (y % (2*unit ) )== 0;
 
                 //if(BIX == 10 and BIY == 12 and BIZ == 0 and x==13 and y==6 and z==9)
                //     printf("face %d %d\n", I_YZ,I_XZ);
 
                  
                 if (I_YZ){
 
 
                     auto interp_z = get_interp_order(z,BIZ,GDZ,global_z,data_size.z);
                     auto interp_y = get_interp_order(y,BIY,GDY,global_y,data_size.y);
 
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
                     auto interp_z = get_interp_order(z,BIZ,GDZ,global_z,data_size.z);
                     auto interp_x = get_interp_order(x,BIX,GDX,global_x,data_size.x);
 
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
 
                     auto interp_y = get_interp_order(y,BIY,GDY,global_y,data_size.y);
                     auto interp_x = get_interp_order(x,BIX,GDX,global_x,data_size.x);
 
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
                 auto interp_z = get_interp_order(z,BIZ,GDZ,global_z,data_size.z);
                 auto interp_y = get_interp_order(y,BIY,GDY,global_y,data_size.y);
                 auto interp_x = get_interp_order(x,BIX,GDX,global_x,data_size.x);
 
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
                 /*
                  if(BIX == 12 and BIY == 12 and BIZ == 8 and unit==4 and x==0 and y==0 and z==4)
                         printf("004pred %.2e %.2e %.2e %.2e %.2e %.2e\n",pred,code,s_data[z][y][x],s_data[0][0][0],s_data[0][0][8],s_data[0][0][16]);
                     if(BIX == 12 and BIY == 12 and BIZ == 8 and unit==4 and x==8 and y==8 and z==4)
                         printf("884pred %.2e %.2e %.2e %.2e %.2e %.2e\n",pred,code,s_data[z][y][x],s_data[8][8][0],s_data[8][8][8],s_data[8][8][16]);
                   */      
                // if(fabs(pred)>=3)
                //     printf("%d %d %d %d %d %d %d %d %d %d %.2e %.2e %.2e\n",unit,CONSTEXPR (BLUE),CONSTEXPR (YELLOW),CONSTEXPR (HOLLOW),BIX,BIY,BIZ,x,y,z,pred,code,s_data[z][y][x]);
               
                 s_data[z][y][x]  = pred + (code - radius) * ebx2;
 
                 //if(BIX == 10 and BIY == 12 and BIZ == 0 and x==13 and y==6 and z==9)
                 //    printf("cmp\n");
                 
 
             }
             else {  // TODO == DECOMPRESSS and static_assert
 
                 //if(BIX == 10 and BIY == 12 and BIZ == 0 and x==13 and y==6 and z==9)
                 //    printf("dcmp\n");
                 auto code       = s_ectrl[z][y][x];
                 s_data[z][y][x] = pred + (code - radius) * ebx2;
                // if(isnan(s_data[z][y][x])){
                 //    printf("nan %d %d %d %d %d %d %d %d %d %.6e %.2e\n",BIX,BIY,BIZ,x,y,z,LINE,FACE,CUBE,pred,code);
                 //}
                 //  
                 //    printf("NAN: %.6e %.2e %.6e %.6e %.6e %.6e %.6e %.6e\n",pred,code,s_data[z][y][x-3*unit],s_data[z][y][x+3*unit],s_data[z][y-3*unit][x],s_data[z][y+3*unit][x],s_data[z-3*unit][y][x],s_data[z+3*unit][y][x] );
                 //if(BIX == 23 and BIY == 15 and BIZ == 0 and x==13 and y==7 and z==11)
                 //    printf("NAN: %.6e %.2e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n",pred,code,s_data[z][y][x-3*unit],s_data[z][y][x-unit],s_data[z][y][x+unit],s_data[z][y][x+3*unit],s_data[z-3*unit][y][x],s_data[z-unit][y][x],s_data[z+unit][y][x],s_data[z+3*unit][y][x] );
                 /*
 
 
                 if(BIX == 12 and BIY == 12 and BIZ == 8 and unit==4 and x==0 and y==0 and z==4)
                         printf("004pred %.2e %.2e %.2e %.2e %.2e %.2e\n",pred,code,s_data[z][y][x],s_data[0][0][0],s_data[0][0][8],s_data[0][0][16]);
                     if(BIX == 12 and BIY == 12 and BIZ == 8 and unit==4 and x==8 and y==8 and z==4)
                         printf("884pred %.2e %.2e %.2e %.2e %.2e %.2e\n",pred,code,s_data[z][y][x],s_data[8][8][0],s_data[8][8][8],s_data[8][8][16]);
                         
                 */
                 //if(BIX == 4 and BIY == 20 and BIZ == 20 and unit==1 and CONSTEXPR (BLUE)){
                //     if(fabs(s_data[z][y][x])>=3)
 
               //      printf("%d %d %d %d %d %d %d %d %d %d %.2e %.2e %.2e\n",unit,CONSTEXPR (BLUE),CONSTEXPR (YELLOW),CONSTEXPR (HOLLOW),BIX,BIY,BIZ,x,y,z,pred,code,s_data[z][y][x]);
                // }
             }
         }
     };
     // -------------------------------------------------------------------------------- //
 
     if CONSTEXPR (COARSEN) {
         constexpr auto TOTAL = NUM_ELE;
         //if( BLOCK_DIMX *BLOCK_DIMY<= LINEAR_BLOCK_SIZE){
             for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
                 auto [x,y,z]    = xyzmap(_tix, unit);
                 run(x, y, z);
             }
         //}
         //may have bug    
         /*
         else{
             for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
                 auto itix = (_tix % BLOCK_DIMX);
                 auto itiz = (_tix / BLOCK_DIMX) % BLOCK_DIMZ;
                 auto itiy = (_tix / BLOCK_DIMX) / BLOCK_DIMZ;
                 auto x    = xmap(itix, unit);
                 auto y    = ymap(itiy, unit);
                 auto z    = zmap(itiz, unit);
                 run(x, y, z);
             }
         }*/
         //may have bug  end
         
     }
     else {
         if(TIX<NUM_ELE){
             auto [x,y,z]    = xyzmap(TIX, unit);
             
 
          //   printf("%d %d %d\n", x,y,z);
             run(x, y, z);
         }
     }
     __syncthreads();
 }
 
 
 
 }  // namespace
 
 /********************************************************************************/
 /*
 template <typename T,typename FP,int  LINEAR_BLOCK_SIZE>
 __device__ void cusz::device_api::auto_tuning(volatile T s_data[9][9][33],  DIM3  data_size, FP eb_r, FP ebx2){
     //current design: 4 points: (4,4,4), (12,4,4), (20,4,4), (28,4,4). 6 configs (3 directions, lin/cubic)
     auto itix=TIX % 4;//follow the warp
     auto c=TIX/4;//follow the warp
     bool predicate=(c<6);
     if(predicate){
         auto x=4+8*itix;
         auto y=4;
         auto z=4;
         T pred=0;
         auto unit = 1;
 
         bool cubic=c%2;
         bool axis=c/2;
         bool flag_cub[2]={0,1};
         bool flag_axis_z[3]={1,0,0};
         bool flag_axis_y[3]={0,1,0};
         bool flag_axis_x[3]={0,0,1};
         T adjust_rate[2]={8.0/9.0,1};
 
         pred=(-flag_cub[cubic]*s_data[z - 3*unit*flag_axis_z[axis]][y- 3*unit*flag_axis_y[axis]][x- 3*unit*flag_axis_x[axis]]
             +9*s_data[z -unit*flag_axis_z[axis]][y-unit*flag_axis_y[axis]][x- unit*flag_axis_x[axis]]
             +9*s_data[z +unit*flag_axis_z[axis]][y+unit*flag_axis_y[axis]][x+ unit*flag_axis_x[axis]]
             -flag_cub[cubic]*s_data[z + 3*unit*flag_axis_z[axis]][y+ 3*unit*flag_axis_y[axis]][x+ 3*unit*flag_axis_x[axis]])/16;
 
         pred*=adjust_rate[cubic];
         T abs_error=fabs(pred-s_data[z][y][x]);
 
     } 
     
 }
 */
 template <typename T,int  LINEAR_BLOCK_SIZE>
 __device__ void cusz::device_api::auto_tuning(volatile T s_data[16][16][16],  volatile T local_errs[2], DIM3  data_size,  T * errs){
  
     if(TIX<2)
         local_errs[TIX]=0;
     __syncthreads(); 
 
     auto local_idx=TIX%2;
     auto temp=TIX/2;
     
 
     auto block_idx_x =  temp % 4;
     auto block_idx_y = ( temp / 4) % 4;
     auto block_idx_z = ( ( temp  / 4) / 4) % 4;
     auto dir = ( ( temp  / 4) / 4) / 4;
 
 
 
     bool predicate=  dir<2;
     // __shared__ T local_errs[6];
 
 
 
     
     if(predicate){
 
        
         
         auto x=4*block_idx_x+1+local_idx;
         //auto x =16;
         auto y=4*block_idx_y+1+local_idx;
         auto z=4*block_idx_z+1+local_idx;
 
         
         T pred=0;
 
         //auto unit = 1;
         switch(dir){
             case 0:
                 pred = (s_data[z - 1][y][x] + s_data[z + 1][y][x]) / 2;
                 break;
 
             
             case 1:
                  pred = (s_data[z][y][x - 1] + s_data[z][y][x + 1]) / 2;
                 break;
 
             default:
             break;
         }
         
         T abs_error=fabs(pred-s_data[z][y][x]);
         atomicAdd(const_cast<T*>(local_errs) + dir, abs_error);
         
 
     } 
     __syncthreads(); 
     if(TIX<2)
         errs[TIX]=local_errs[TIX];
     __syncthreads(); 
        
 }
 
 
 template <typename T,int  LINEAR_BLOCK_SIZE>
 __device__ void cusz::device_api::auto_tuning_2(volatile T s_data[64], volatile T s_nx[64][4], volatile T s_ny[64][4], volatile T s_nz[64][4],  volatile T local_errs[6], DIM3  data_size,  T * errs){
  
     if(TIX<6)
         local_errs[TIX]=0;
     __syncthreads(); 
 
     auto point_idx=TIX%64;
     auto c=TIX/64;
 
 
     bool predicate=  c<6;
     // __shared__ T local_errs[6];
 
     
     if(predicate){
 
        
     
         
         T pred=0;
 
         //auto unit = 1;
         switch(c){
                 case 0:
                     pred = (-s_nz[point_idx][0]+9*s_nz[point_idx][1] + 9*s_nz[point_idx][2]-s_nz[point_idx][3]) / 16;
                     break;
 
                 case 1:
                     pred = (-3*s_nz[point_idx][0]+23*s_nz[point_idx][1] + 23*s_nz[point_idx][2]-3*s_nz[point_idx][3]) / 40;
                     break;
                 case 2:
                     pred = (-s_ny[point_idx][0]+9*s_ny[point_idx][1] + 9*s_ny[point_idx][2]-s_ny[point_idx][3]) / 16;
                     break;
                 case 3:
                     pred = (-3*s_ny[point_idx][0]+23*s_ny[point_idx][1] + 23*s_ny[point_idx][2]-3*s_ny[point_idx][3]) / 40;
                     break;
 
                 case 4:
                     pred = (-s_nx[point_idx][0]+9*s_nx[point_idx][1] + 9*s_nx[point_idx][2]-s_nx[point_idx][3]) / 16;
                     break;
                 case 5:
                     pred = (-3*s_nx[point_idx][0]+23*s_nx[point_idx][1] + 23*s_nx[point_idx][2]-3*s_nx[point_idx][3]) / 40;
                     break;
 
 
 
                 default:
                 break;
             }
         
         T abs_error=fabs(pred-s_data[point_idx]);
         atomicAdd(const_cast<T*>(local_errs) + c, abs_error);
         
 
     } 
     __syncthreads(); 
     if(TIX<6)
         errs[TIX]=local_errs[TIX];
     __syncthreads(); 
        
 }
 
 
 
 template <typename T1, typename T2, typename FP,int LINEAR_BLOCK_SIZE, bool WORKFLOW, bool PROBE_PRED_ERROR>
 __device__ void cusz::device_api::spline3d_layout2_interpolate(
     volatile T1 s_data[17][17][17],
     volatile T2 s_ectrl[17][17][17],
      DIM3    data_size,
     FP          eb_r,
     FP          ebx2,
     int         radius,
     INTERPOLATION_PARAMS intp_param
     )
 {
     auto xblue = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2); };
     auto yblue = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2); };
     auto zblue = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz * 2 + 1); };
 
     auto xblue_reverse = [] __device__(int _tix, int unit) -> int { return unit * (_tix ); };
     auto yblue_reverse = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy ); };
     auto zblue_reverse = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz * 2 + 1); };
 
     auto xyellow = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2); };
     auto yyellow = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2+1); };
     auto zyellow = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz); };
 
     auto xyellow_reverse = [] __device__(int _tix, int unit) -> int { return unit * (_tix ); };
     auto yyellow_reverse = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2+1); };
     auto zyellow_reverse = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz * 2); };
 
 
     auto xhollow = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2 +1); };
     auto yhollow = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy); };
     auto zhollow = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz); };
 
     auto xhollow_reverse = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2 +1); };
     auto yhollow_reverse = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2); };
     auto zhollow_reverse = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz *2); };
 
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
 
     auto xyzmap_line_16b_4u = [] __device__(int _tix, int unit) -> std::tuple<int,int,int> {
         constexpr auto N = 2;
         constexpr auto L = N*(N+1)*(N+1); 
         constexpr auto Q = (N+1)*(N+1); 
         auto group = _tix / L ;
         auto m = _tix % L ;
         auto i = m / Q;
         auto j = (m % Q) / (N+1);
         auto k = (m % Q) % (N+1);
         if(group==0)
             return std::make_tuple(8*i+4,8*j,8*k);
         else if (group==1)
             return std::make_tuple(8*k,8*i+4,8*j);
         else
             return std::make_tuple(8*j,8*k,8*i+4);
 
     };
 
     auto xyzmap_face_16b_4u = [] __device__(int _tix, int unit) -> std::tuple<int,int,int> {
         constexpr auto N = 2;
         constexpr auto L = N*N*(N+1);
         constexpr auto Q = N*N; 
         auto group = _tix / L ;
         auto m = _tix % L ;
         auto i = m / Q;
         auto j = (m % Q) / N;
         auto k = (m % Q) % N;
         if(group==0)
             return std::make_tuple(8*i,8*j+4,8*k+4);
         else if (group==1)
             return std::make_tuple(8*k+4,8*i,8*j+4);
         else
             return std::make_tuple(8*j+4,8*k+4,8*i);
 
     };
 
      auto xyzmap_cube_16b_4u = [] __device__(int _tix, int unit) -> std::tuple<int,int,int> {
         constexpr auto N = 2;
         constexpr auto Q = N * N; 
         auto i = _tix / Q;
         auto j = (_tix % Q) / N;
         auto k = (_tix % Q) % N;
         return std::make_tuple(8*i+4,8*j+4,8*k+4);
 
     };
 
     auto xyzmap_line_16b_8u = [] __device__(int _tix, int unit) -> std::tuple<int,int,int> {
         constexpr auto N = 1;
         constexpr auto L = N*(N+1)*(N+1); 
         constexpr auto Q = (N+1)*(N+1); 
         auto group = _tix / L ;
         auto m = _tix % L ;
         auto i = m / Q;
         auto j = (m % Q) / (N+1);
         auto k = (m % Q) % (N+1);
         if(group==0)
             return std::make_tuple(16*i+8,16*j,16*k);
         else if (group==1)
             return std::make_tuple(16*k,16*i+8,16*j);
         else
             return std::make_tuple(16*j,16*k,16*i+8);
 
     };
 
     auto xyzmap_face_16b_8u = [] __device__(int _tix, int unit) -> std::tuple<int,int,int> {
         constexpr auto N = 1;
         constexpr auto L = N*N*(N+1);
         constexpr auto Q = N*N; 
         auto group = _tix / L ;
         auto m = _tix % L ;
         auto i = m / Q;
         auto j = (m % Q) / N;
         auto k = (m % Q) % N;
         if(group==0)
             return std::make_tuple(16*i,16*j+8,16*k+8);
         else if (group==1)
             return std::make_tuple(16*k+8,16*i,16*j+8);
         else
             return std::make_tuple(16*j+8,16*k+8,16*i);
 
     };
 
      auto xyzmap_cube_16b_8u = [] __device__(int _tix, int unit) -> std::tuple<int,int,int> {
         constexpr auto N = 1;
         constexpr auto Q = N * N; 
         auto i = _tix / Q;
         auto j = (_tix % Q) / N;
         auto k = (_tix % Q) % N;
         return std::make_tuple(16*i+8,16*j+8,16*k+8);
 
     };
 
 
     auto nan_cubic_interp = [] __device__ (T1 a, T1 b, T1 c, T1 d) -> T1{
         return (-a+9*b+9*c-d) / 16;
     };
 
     auto nat_cubic_interp = [] __device__ (T1 a, T1 b, T1 c, T1 d) -> T1{
         return (-3*a+23*b+23*c-3*d) / 40;
     };
 
 
 
 
     constexpr auto COARSEN          = true;
     constexpr auto NO_COARSEN       = false;
     constexpr auto BORDER_INCLUSIVE = true;
     constexpr auto BORDER_EXCLUSIVE = false;
 
     
     FP cur_ebx2=ebx2,cur_eb_r=eb_r;
 
 
     auto calc_eb = [&](auto unit) {
         cur_ebx2=ebx2,cur_eb_r=eb_r;
         int temp=1;
         while(temp<unit){
             temp*=2;
             cur_eb_r *= intp_param.alpha;
             cur_ebx2 /= intp_param.alpha;
 
         }
         if(cur_ebx2 < ebx2 / intp_param.beta){
             cur_ebx2 = ebx2 / intp_param.beta;
             cur_eb_r = eb_r * intp_param.beta;
 
         }
     };
     
     int unit = 8;
     //int m = 2, n = 2, p = 2;//block8 nums;
     calc_eb(unit);
     //set_orders(reverse[2]);
     if(intp_param.use_md[3]){
 
 
         if(intp_param.use_natural[3]==0){
 
             interpolate_stage_md<
                 T1, T2, FP, decltype(xyzmap_line_16b_8u), //
                 true, false, false, LINEAR_BLOCK_SIZE,12 ,NO_COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyzmap_line_16b_8u, unit, cur_eb_r, cur_ebx2, radius, nan_cubic_interp);
 
             interpolate_stage_md<
                 T1, T2, FP, decltype(xyzmap_face_16b_8u), //
                 false, true, false, LINEAR_BLOCK_SIZE,6 ,NO_COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyzmap_face_16b_8u, unit, cur_eb_r, cur_ebx2, radius, nan_cubic_interp);
 
             interpolate_stage_md<
                 T1, T2, FP, decltype(xyzmap_cube_16b_8u), //
                 false, false, true, LINEAR_BLOCK_SIZE,1 ,NO_COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyzmap_cube_16b_8u, unit, cur_eb_r, cur_ebx2, radius, nan_cubic_interp);
 
         }
         else{
             interpolate_stage_md<
                 T1, T2, FP, decltype(xyzmap_line_16b_8u), //
                 true, false, false, LINEAR_BLOCK_SIZE,12 ,NO_COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyzmap_line_16b_8u, unit, cur_eb_r, cur_ebx2, radius, nat_cubic_interp);
 
             interpolate_stage_md<
                 T1, T2, FP, decltype(xyzmap_face_16b_8u), //
                 false, true, false, LINEAR_BLOCK_SIZE,6 ,NO_COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyzmap_face_16b_8u, unit, cur_eb_r, cur_ebx2, radius, nat_cubic_interp);
 
             interpolate_stage_md<
                 T1, T2, FP, decltype(xyzmap_cube_16b_8u), //
                 false, false, true, LINEAR_BLOCK_SIZE,1 ,NO_COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyzmap_cube_16b_8u, unit, cur_eb_r, cur_ebx2, radius, nat_cubic_interp);
             
 
         }
 
     }
     else{
 
         if(intp_param.reverse[3]){
             interpolate_stage<
                 T1, T2, FP, decltype(xhollow_reverse), decltype(yhollow_reverse), decltype(zhollow_reverse),  //
                 false, false, true, LINEAR_BLOCK_SIZE, 1, 2, NO_COARSEN, 2, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xhollow_reverse, yhollow_reverse, zhollow_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[3]);
 
             interpolate_stage<
                 T1, T2, FP, decltype(xyellow_reverse), decltype(yyellow_reverse), decltype(zyellow_reverse),  //
                 false, true, false, LINEAR_BLOCK_SIZE, 3, 1, NO_COARSEN, 2, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyellow_reverse, yyellow_reverse, zyellow_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[3]);
             interpolate_stage<
                 T1, T2, FP, decltype(xblue_reverse), decltype(yblue_reverse), decltype(zblue_reverse),  //
                 true, false, false, LINEAR_BLOCK_SIZE, 3, 3, NO_COARSEN, 1, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xblue_reverse, yblue_reverse, zblue_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[3]);
 
 
         }
         else{
             //if( BIX==0 and BIY==0 and BIZ==0)
            // printf("lv3s0\n");
             interpolate_stage<
                 T1, T2, FP, decltype(xblue), decltype(yblue), decltype(zblue),  //
                 true, false, false, LINEAR_BLOCK_SIZE, 2, 2, NO_COARSEN, 1, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xblue, yblue, zblue, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[3]);
            // if(BIX==0 and BIY==0 and BIZ==0)
            // printf("lv3s1\n");
             interpolate_stage<
                 T1, T2, FP, decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
                 false, true, false, LINEAR_BLOCK_SIZE, 2, 1, NO_COARSEN, 3, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyellow, yyellow, zyellow, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[3]);
             //if(BIX==0 and BIY==0 and BIZ==0)
           //  printf("lv3s2\n");
             interpolate_stage<
                 T1, T2, FP, decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
                 false, false, true, LINEAR_BLOCK_SIZE, 1, 3, NO_COARSEN, 3, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xhollow, yhollow, zhollow, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[3]);
         }
     }
 
 
     unit = 4;
     //int m = 2, n = 2, p = 2;//block8 nums;
     calc_eb(unit);
     //set_orders(reverse[2]);
 
 
     if(intp_param.use_md[2]){
 
 
         if(intp_param.use_natural[2]==0){
 
             interpolate_stage_md<
                 T1, T2, FP, decltype(xyzmap_line_16b_4u), //
                 true, false, false, LINEAR_BLOCK_SIZE,54 ,NO_COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyzmap_line_16b_4u, unit, cur_eb_r, cur_ebx2, radius, nan_cubic_interp);
 
             interpolate_stage_md<
                 T1, T2, FP, decltype(xyzmap_face_16b_4u), //
                 false, true, false, LINEAR_BLOCK_SIZE,36 ,NO_COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyzmap_face_16b_4u, unit, cur_eb_r, cur_ebx2, radius, nan_cubic_interp);
 
             interpolate_stage_md<
                 T1, T2, FP, decltype(xyzmap_cube_16b_4u), //
                 false, false, true, LINEAR_BLOCK_SIZE,8 ,NO_COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyzmap_cube_16b_4u, unit, cur_eb_r, cur_ebx2, radius, nan_cubic_interp);
 
         }
         else{
             interpolate_stage_md<
                 T1, T2, FP, decltype(xyzmap_line_16b_4u), //
                 true, false, false, LINEAR_BLOCK_SIZE,54 ,NO_COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyzmap_line_16b_4u, unit, cur_eb_r, cur_ebx2, radius, nat_cubic_interp);
 
             interpolate_stage_md<
                 T1, T2, FP, decltype(xyzmap_face_16b_4u), //
                 false, true, false, LINEAR_BLOCK_SIZE,36 ,NO_COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyzmap_face_16b_4u, unit, cur_eb_r, cur_ebx2, radius, nat_cubic_interp);
 
             interpolate_stage_md<
                 T1, T2, FP, decltype(xyzmap_cube_16b_4u), //
                 false, false, true, LINEAR_BLOCK_SIZE,8 ,NO_COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyzmap_cube_16b_4u, unit, cur_eb_r, cur_ebx2, radius, nat_cubic_interp);
             
 
         }
 
     }
 
 
     else{
         if(intp_param.reverse[2]){
             interpolate_stage<
                 T1, T2, FP, decltype(xhollow_reverse), decltype(yhollow_reverse), decltype(zhollow_reverse),  //
                 false, false, true, LINEAR_BLOCK_SIZE, 2, 3, NO_COARSEN, 3, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xhollow_reverse, yhollow_reverse, zhollow_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[2]);
 
             interpolate_stage<
                 T1, T2, FP, decltype(xyellow_reverse), decltype(yyellow_reverse), decltype(zyellow_reverse),  //
                 false, true, false, LINEAR_BLOCK_SIZE, 5, 2, NO_COARSEN, 3, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyellow_reverse, yyellow_reverse, zyellow_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[2]);
             interpolate_stage<
                 T1, T2, FP, decltype(xblue_reverse), decltype(yblue_reverse), decltype(zblue_reverse),  //
                 true, false, false, LINEAR_BLOCK_SIZE, 5, 5, NO_COARSEN, 2, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xblue_reverse, yblue_reverse, zblue_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[2]);
 
 
         }
         else{
             //if( BIX==0 and BIY==0 and BIZ==0)
            // printf("lv3s0\n");
             interpolate_stage<
                 T1, T2, FP, decltype(xblue), decltype(yblue), decltype(zblue),  //
                 true, false, false, LINEAR_BLOCK_SIZE, 3, 3, NO_COARSEN, 2, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xblue, yblue, zblue, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[2]);
            // if(BIX==0 and BIY==0 and BIZ==0)
            // printf("lv3s1\n");
             interpolate_stage<
                 T1, T2, FP, decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
                 false, true, false, LINEAR_BLOCK_SIZE, 3, 2, NO_COARSEN, 5, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyellow, yyellow, zyellow, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[2]);
             //if(BIX==0 and BIY==0 and BIZ==0)
           //  printf("lv3s2\n");
             interpolate_stage<
                 T1, T2, FP, decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
                 false, false, true, LINEAR_BLOCK_SIZE, 2, 5, NO_COARSEN, 5, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xhollow, yhollow, zhollow, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[2]);
         }
     }
    // if(BIX==0 and BIY==0 and BIZ==0)
    // printf("lv3\n");
 
     unit = 2;
     //m*=2;
     //n*=2;
     //p*=2;
     calc_eb(unit);
     //set_orders(reverse[1]);
 
     // iteration 2, TODO switch y-z order
     /*
    */
 
     if(intp_param.use_md[1]){
 
 
         if(intp_param.use_natural[1]==0){
 
             interpolate_stage_md<
                 T1, T2, FP, decltype(xyzmap_line_16b_2u), //
                 true, false, false, LINEAR_BLOCK_SIZE,300 ,NO_COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyzmap_line_16b_2u, unit, cur_eb_r, cur_ebx2, radius, nan_cubic_interp);
 
             interpolate_stage_md<
                 T1, T2, FP, decltype(xyzmap_face_16b_2u), //
                 false, true, false, LINEAR_BLOCK_SIZE,240 ,NO_COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyzmap_face_16b_2u, unit, cur_eb_r, cur_ebx2, radius, nan_cubic_interp);
 
             interpolate_stage_md<
                 T1, T2, FP, decltype(xyzmap_cube_16b_2u), //
                 false, false, true, LINEAR_BLOCK_SIZE,64 ,NO_COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyzmap_cube_16b_2u, unit, cur_eb_r, cur_ebx2, radius, nan_cubic_interp);
 
         }
         else{
             interpolate_stage_md<
                 T1, T2, FP, decltype(xyzmap_line_16b_2u), //
                 true, false, false, LINEAR_BLOCK_SIZE,300 ,NO_COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyzmap_line_16b_2u, unit, cur_eb_r, cur_ebx2, radius, nat_cubic_interp);
 
             interpolate_stage_md<
                 T1, T2, FP, decltype(xyzmap_face_16b_2u), //
                 false, true, false, LINEAR_BLOCK_SIZE,240 ,NO_COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyzmap_face_16b_2u, unit, cur_eb_r, cur_ebx2, radius, nat_cubic_interp);
 
             interpolate_stage_md<
                 T1, T2, FP, decltype(xyzmap_cube_16b_2u), //
                 false, false, true, LINEAR_BLOCK_SIZE,64 ,NO_COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyzmap_cube_16b_2u, unit, cur_eb_r, cur_ebx2, radius, nat_cubic_interp);
             
 
         }
     }
     else{
          if(intp_param.reverse[1]){
             interpolate_stage<
                 T1, T2, FP, decltype(xhollow_reverse), decltype(yhollow_reverse), decltype(zhollow_reverse),  //
                 false, false, true, LINEAR_BLOCK_SIZE, 4, 5, NO_COARSEN, 5, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xhollow_reverse, yhollow_reverse, zhollow_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[1]);
             interpolate_stage<
                 T1, T2, FP, decltype(xyellow_reverse), decltype(yyellow_reverse), decltype(zyellow_reverse),  //
                 false, true, false, LINEAR_BLOCK_SIZE, 9, 4, NO_COARSEN, 5, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyellow_reverse, yyellow_reverse, zyellow_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[1]);
             interpolate_stage<
                 T1, T2, FP, decltype(xblue_reverse), decltype(yblue_reverse), decltype(zblue_reverse),  //
                 true, false, false, LINEAR_BLOCK_SIZE, 9, 9, NO_COARSEN, 4, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xblue_reverse, yblue_reverse, zblue_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[1]);
         }
         else{
             interpolate_stage<
                 T1, T2, FP, decltype(xblue), decltype(yblue), decltype(zblue),  //
                 true, false, false, LINEAR_BLOCK_SIZE, 5, 5, NO_COARSEN, 4, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xblue, yblue, zblue, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[1]);
             interpolate_stage<
                 T1, T2, FP, decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
                 false, true, false, LINEAR_BLOCK_SIZE, 5, 4, NO_COARSEN, 9, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyellow, yyellow, zyellow, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[1]);
             interpolate_stage<
                 T1, T2, FP, decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
                 false, false, true, LINEAR_BLOCK_SIZE, 4, 9, NO_COARSEN, 9, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xhollow, yhollow, zhollow, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[1]);
 
         }
 
     }
 
 
     //if(TIX==0 and TIY==0 and TIZ==0 and BIX==0 and BIY==0 and BIZ==0)
     //printf("lv2\n");
     unit = 1;
     //m*=2;
     //n*=2;
     //p*=2;
     calc_eb(unit);
    // set_orders(reverse[0]);
     if(intp_param.use_md[0]){
         if(intp_param.use_natural[0]==0){
 
             interpolate_stage_md<
                 T1, T2, FP, decltype(xyzmap_line_16b_1u), //
                 true, false, false, LINEAR_BLOCK_SIZE,1944 ,COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyzmap_line_16b_1u, unit, cur_eb_r, cur_ebx2, radius, nan_cubic_interp);
 
             interpolate_stage_md<
                 T1, T2, FP, decltype(xyzmap_face_16b_1u), //
                 false, true, false, LINEAR_BLOCK_SIZE,1728 ,COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyzmap_face_16b_1u, unit, cur_eb_r, cur_ebx2, radius, nan_cubic_interp);
 
             interpolate_stage_md<
                 T1, T2, FP, decltype(xyzmap_cube_16b_1u), //
                 false, false, true, LINEAR_BLOCK_SIZE,512 ,COARSEN, BORDER_EXCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyzmap_cube_16b_1u, unit, cur_eb_r, cur_ebx2, radius, nan_cubic_interp);
 
         }
         else{
             interpolate_stage_md<
                 T1, T2, FP, decltype(xyzmap_line_16b_1u), //
                 true, false, false, LINEAR_BLOCK_SIZE,1944 ,COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyzmap_line_16b_1u, unit, cur_eb_r, cur_ebx2, radius, nat_cubic_interp);
 
             interpolate_stage_md<
                 T1, T2, FP, decltype(xyzmap_face_16b_1u), //
                 false, true, false, LINEAR_BLOCK_SIZE,1728 ,COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyzmap_face_16b_1u, unit, cur_eb_r, cur_ebx2, radius, nat_cubic_interp);
 
             interpolate_stage_md<
                 T1, T2, FP, decltype(xyzmap_cube_16b_1u), //
                 false, false, true, LINEAR_BLOCK_SIZE,512 ,COARSEN, BORDER_EXCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyzmap_cube_16b_1u, unit, cur_eb_r, cur_ebx2, radius, nat_cubic_interp);
             
 
         }
     }
     else{
         if(intp_param.reverse[0]){
             //may have bug 
             interpolate_stage<
                 T1, T2, FP, decltype(xhollow_reverse), decltype(yhollow_reverse), decltype(zhollow_reverse),  //
                 false, false, true, LINEAR_BLOCK_SIZE, 8, 9, COARSEN, 9, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xhollow_reverse, yhollow_reverse, zhollow_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[0]);
             interpolate_stage<
                 T1, T2, FP, decltype(xyellow_reverse), decltype(yyellow_reverse), decltype(zyellow_reverse),  //
                 false, true, false, LINEAR_BLOCK_SIZE, 17, 8, COARSEN, 9, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyellow_reverse, yyellow_reverse, zyellow_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[0]);
             interpolate_stage<
                 T1, T2, FP, decltype(xblue_reverse), decltype(yblue_reverse), decltype(zblue_reverse),  //
                 true, false, false, LINEAR_BLOCK_SIZE, 17, 17, COARSEN, 8, BORDER_EXCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xblue_reverse, yblue_reverse, zblue_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[0]);
 
             //may have bug end
         }
         else{
             interpolate_stage<
                 T1, T2, FP, decltype(xblue), decltype(yblue), decltype(zblue),  //
                 true, false, false, LINEAR_BLOCK_SIZE, 9, 9, COARSEN, 8, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xblue, yblue, zblue, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[0]);
             interpolate_stage<
                 T1, T2, FP, decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
                 false, true, false, LINEAR_BLOCK_SIZE, 9, 8, COARSEN, 17, BORDER_INCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xyellow, yyellow, zyellow, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[0]);
            
             interpolate_stage<
                 T1, T2, FP, decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
                 false, false, true, LINEAR_BLOCK_SIZE, 8, 17, COARSEN, 17, BORDER_EXCLUSIVE, WORKFLOW>(
                 s_data, s_ectrl,data_size, xhollow, yhollow, zhollow, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[0]);
 
         }
     }
 
     // iteration 3
     /*
     */
   //  if(TIX==0 and TIY==0 and TIZ==0 and BIX==0 and BIY==0 and BIZ==0)
    // printf("lv1\n");
     
 
 
      /******************************************************************************
      test only: last step inclusive
      ******************************************************************************/
     // interpolate_stage<
     //     T1, T2, FP, decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
     //     false, false, true, LINEAR_BLOCK_SIZE, 33, 4, COARSEN, 9, BORDER_INCLUSIVE, WORKFLOW>(
     //     s_data, s_ectrl, xhollow, yhollow, zhollow, unit, eb_r, ebx2, radius);
     /******************************************************************************
      production
      ******************************************************************************/
 
     /******************************************************************************
      test only: print a block
      ******************************************************************************/
     // if (TIX == 0 and BIX == 7 and BIY == 47 and BIZ == 15) { spline3d_print_block_from_GPU(s_ectrl); }
    //  if (TIX == 0 and BIX == 4 and BIY == 20 and BIZ == 20) { spline3d_print_block_from_GPU(s_data); }
 }
 
 /********************************************************************************
  * host API/kernel
  ********************************************************************************/
 template <typename TITER, int LINEAR_BLOCK_SIZE>
 __global__ void cusz::c_spline3d_profiling_16x16x16data(
     TITER   data,
     DIM3    data_size,
     STRIDE3 data_leap,
     TITER errors)
 {
     // compile time variables
     using T = typename std::remove_pointer<TITER>::type;
  
 
     {
         __shared__ struct {
             T data[16][16][16];
             T local_errs[2];
            // T global_errs[6];
         } shmem;
 
 
         c_reset_scratch_profiling_16x16x16data<T, LINEAR_BLOCK_SIZE>(shmem.data, 0.0);
         //if(TIX==0 and BIX==0 and BIY==0 and BIZ==0)
         //    printf("reset\n");
         //if(TIX==0 and BIX==0 and BIY==0 and BIZ==0)
         //    printf("dsz: %d %d %d\n",data_size.x,data_size.y,data_size.z);
 
         global2shmem_profiling_16x16x16data<T, T, LINEAR_BLOCK_SIZE>(data, data_size, data_leap, shmem.data);
 
 
 
      
 
         //if (TIX < 6 and BIX==0 and BIY==0 and BIZ==0) errors[TIX] = 0.0;//risky
 
         //__syncthreads();
        
 
         cusz::device_api::auto_tuning<T,LINEAR_BLOCK_SIZE>(
             shmem.data, shmem.local_errs, data_size, errors);
         //if(TIX==0 and BIX==0 and BIY==0 and BIZ==0)
         //    printf("device %.4f %.4f\n",errors[0],errors[1]);
 
         
     }
 }
 
 template <typename TITER, int LINEAR_BLOCK_SIZE>
 __global__ void cusz::c_spline3d_profiling_data_2(
     TITER   data,
     DIM3    data_size,
     STRIDE3 data_leap,
     TITER errors)
 {
     // compile time variables
     using T = typename std::remove_pointer<TITER>::type;
  
 
     {
         __shared__ struct {
             T data[64];
             T neighbor_x [64][4];
             T neighbor_y [64][4];
             T neighbor_z [64][4];
             T local_errs[6];
            // T global_errs[6];
         } shmem;
 
 
         c_reset_scratch_profiling_data_2<T, LINEAR_BLOCK_SIZE>(shmem.data, shmem.neighbor_x, shmem.neighbor_y, shmem.neighbor_z,0.0);
         //if(TIX==0 and BIX==0 and BIY==0 and BIZ==0)
         //    printf("reset\n");
         //if(TIX==0 and BIX==0 and BIY==0 and BIZ==0)
         //    printf("dsz: %d %d %d\n",data_size.x,data_size.y,data_size.z);
 
         global2shmem_profiling_data_2<T, T, LINEAR_BLOCK_SIZE>(data, data_size, data_leap,shmem.data, shmem.neighbor_x, shmem.neighbor_y, shmem.neighbor_z);
 
 
 
      
 
         if (TIX < 6 and BIX==0 and BIY==0 and BIZ==0) errors[TIX] = 0.0;//risky
 
         //__syncthreads();
        
 
         cusz::device_api::auto_tuning_2<T,LINEAR_BLOCK_SIZE>(
             shmem.data, shmem.neighbor_x, shmem.neighbor_y, shmem.neighbor_z, shmem.local_errs, data_size, errors);
 
         
     }
 }
 
 
 __forceinline__ __device__ void pre_compute(DIM3 data_size, volatile STRIDE3 grid_leaps[5], volatile size_t prefix_nums[5]){
     if(TIX==0){
         auto d_size = data_size;
         
         int level = 0;
         while(level<=4){
             //grid_sizes[level]=d_size;
             grid_leaps[level].x=1;
             grid_leaps[level].y=d_size.x;
             grid_leaps[level].z=d_size.x*d_size.y;
             if(level<4){
 
                 d_size.x= (d_size.x+1)>>1;
                 d_size.y = (d_size.y+1)>>1;
                 d_size.z = (d_size.z+1)>>1;
                 prefix_nums[level] = d_size.x*d_size.y*d_size.z;
             }
             level++;
         }   
         prefix_nums[4]=0;
     }
     __syncthreads(); 
     
 }
 
 template <typename TITER, typename EITER, typename FP, int LINEAR_BLOCK_SIZE, typename CompactVal, typename CompactIdx, typename CompactNum>
 __global__ void cusz::c_spline3d_infprecis_16x16x16data(
     TITER   data,
     DIM3    data_size,
     STRIDE3 data_leap,
     EITER   ectrl,
     DIM3    ectrl_size,
     STRIDE3 ectrl_leap,
     TITER   anchor,
     STRIDE3 anchor_leap,
     CompactVal compact_val,
     CompactIdx compact_idx,
     CompactNum compact_num,
     FP      eb_r,
     FP      ebx2,
     int     radius,
     INTERPOLATION_PARAMS intp_param//,
     //TITER errors
     )
 {
     // compile time variables
     using T = typename std::remove_pointer<TITER>::type;
     using E = typename std::remove_pointer<EITER>::type;
 
     {
         __shared__ struct {
             T data[17][17][17];
             T ectrl[17][17][17];
             //DIM3 grid_sizes[5];
             STRIDE3 grid_leaps[5];
             size_t prefix_nums[5];
 
            // T global_errs[6];
         } shmem;
 
        // T cubic_errors=errors[0]+errors[2]+errors[4];
        // T linear_errors=errors[1]+errors[3]+errors[5];
      // bool do_cubic=(cubic_errors<=linear_errors);
       //intp_param.interpolators[0]=(errors[0]>errors[1]);
       //intp_param.interpolators[1]=(errors[2]>errors[3]);
       //intp_param.interpolators[2]=(errors[4]>errors[5]);
       
       //bool do_reverse=(errors[4+intp_param.interpolators[2]]>errors[intp_param.interpolators[0]]);
         /*
         if(intp_param.auto_tuning){
             bool do_reverse=(errors[1]>3*errors[0]);
            intp_param.reverse[0]=intp_param.reverse[1]=intp_param.reverse[2]=do_reverse;
        }
        */
     /*
        if(TIX==0 and BIX==0 and BIY==0 and BIZ==0){
         printf("Errors: %.6f %.6f \n",errors[0],errors[1]);
         printf("Cubic: %d %d %d\n",intp_param.interpolators[0],intp_param.interpolators[1],intp_param.interpolators[2]);
         printf("reverse: %d %d %d\n",intp_param.reverse[0],intp_param.reverse[1],intp_param.reverse[2]);
        }
        */
         /*
          if(TIX==0 and BIX==0 and BIY==0 and BIZ==0){
              printf("NAT: %d %d %d %d\n",intp_param.use_natural[3],intp_param.use_natural[2],intp_param.use_natural[1],intp_param.use_natural[0]);
           printf("MD: %d %d %d %d\n",intp_param.use_md[3],intp_param.use_md[2],intp_param.use_md[1],intp_param.use_md[0]);
           printf("REVERSE: %d %d %d %d\n",intp_param.reverse[3],intp_param.reverse[2],intp_param.reverse[1],intp_param.reverse[0]);
       }*/
        
    
         pre_compute(ectrl_size,shmem.grid_leaps,shmem.prefix_nums);
 
         c_reset_scratch_17x17x17data<T, T, LINEAR_BLOCK_SIZE>(shmem.data, shmem.ectrl, radius);
         //if(TIX==0 and BIX==0 and BIY==0 and BIZ==0)
         //    printf("reset\n");
         //if(TIX==0 and BIX==0 and BIY==0 and BIZ==0)
         //    printf("dsz: %d %d %d\n",data_size.x,data_size.y,data_size.z);
 
         global2shmem_17x17x17data<T, T, LINEAR_BLOCK_SIZE>(data, data_size, data_leap, shmem.data);
 
        // if(TIX==0 and BIX==0 and BIY==0 and BIZ==0)
         //    printf("g2s\n");
         // version 1, use shmem, erroneous
         // c_gather_anchor<T>(shmem.data, anchor, anchor_leap);
         // version 2, use global mem, correct
         c_gather_anchor<T>(data, data_size, data_leap, anchor, anchor_leap);
 
 
        
 
 
         cusz::device_api::spline3d_layout2_interpolate<T, T, FP,LINEAR_BLOCK_SIZE, SPLINE3_COMPR, false>(
             shmem.data, shmem.ectrl, data_size, eb_r, ebx2, radius, intp_param);
         
         
 
         //if(TIX==0 and BIX==0 and BIY==0 and BIZ==0)
         //    printf("interp\n");
         //if(TIX==0 and BIX==0 and BIY==0 and BIZ==0)
          //   printf("esz: %d %d %d\n",ectrl_size.x,ectrl_size.y,ectrl_size.z);
 
 
         //if(TIX==0 and BIX==0 and BIY==0 and BIZ==0)
 
         shmem2global_17x17x17data_with_compaction<T, E, LINEAR_BLOCK_SIZE>(shmem.ectrl, ectrl, ectrl_size, ectrl_leap, radius, shmem.grid_leaps,shmem.prefix_nums, compact_val, compact_idx, compact_num);
 
         // shmem2global_32x8x8data<T, E, LINEAR_BLOCK_SIZE>(shmem.ectrl, ectrl, ectrl_size, ectrl_leap);
 
         //if(TIX==0 and BIX==0 and BIY==0 and BIZ==0)
         //    printf("s2g\n");
     }
 }
 
 
 
 
 
 template <
     typename EITER,
     typename TITER,
     typename FP,
     int LINEAR_BLOCK_SIZE>
 __global__ void cusz::x_spline3d_infprecis_16x16x16data(
     EITER   ectrl,        // input 1
     DIM3    ectrl_size,   //
     STRIDE3 ectrl_leap,   //
     TITER   anchor,       // input 2
     DIM3    anchor_size,  //
     STRIDE3 anchor_leap,  //
     TITER   data,         // output
     DIM3    data_size,    //
     STRIDE3 data_leap,    //
     FP      eb_r,
     FP      ebx2,
     int     radius,
     INTERPOLATION_PARAMS intp_param)
 {
     // compile time variables
     using E = typename std::remove_pointer<EITER>::type;
     using T = typename std::remove_pointer<TITER>::type;
 
     __shared__ struct {
         T data[17][17][17];
         T ectrl[17][17][17];
         STRIDE3 grid_leaps[5];
         size_t prefix_nums[5];
         //uint64_t L16_num;
         //uint64_t L8_num;
         //uint64_t L4_num;
         //uint64_t L2_num;
     } shmem;
 
     pre_compute(ectrl_size,shmem.grid_leaps,shmem.prefix_nums);
 
     x_reset_scratch_17x17x17data<T, T, LINEAR_BLOCK_SIZE>(shmem.data, shmem.ectrl, anchor, anchor_size, anchor_leap);
 
     //if(TIX==0 and BIX==0 and BIY==0 and BIZ==0)
     //        printf("esz: %d %d %d\n",ectrl_size.x,ectrl_size.y,ectrl_size.z);
 
     // global2shmem_33x9x9data<E, T, LINEAR_BLOCK_SIZE>(ectrl, ectrl_size, ectrl_leap, shmem.ectrl);
     global2shmem_fuse<T, E, LINEAR_BLOCK_SIZE>(ectrl, ectrl_size, ectrl_leap, data, shmem.ectrl, shmem.grid_leaps,shmem.prefix_nums);
 
     cusz::device_api::spline3d_layout2_interpolate<T, T, FP, LINEAR_BLOCK_SIZE, SPLINE3_DECOMPR, false>(
         shmem.data, shmem.ectrl, data_size, eb_r, ebx2, radius, intp_param);
     //if(TIX==0 and BIX==0 and BIY==0 and BIZ==0)
     //        printf("dsz: %d %d %d\n",data_size.x,data_size.y,data_size.z);
     shmem2global_17x17x17data<T, T, LINEAR_BLOCK_SIZE>(shmem.data, data, data_size, data_leap);
 }
 
 
 template <typename TITER>
 __global__ void cusz::reset_errors(TITER errors)
 {
     if(TIX<18)
         errors[TIX]=0;
 }
 
 template <typename T>
 __forceinline__ __device__ void pre_compute_att(DIM3 sam_starts, DIM3 sam_bgs, DIM3 sam_strides, DIM3 &global_starts,INTERPOLATION_PARAMS &intp_param,uint8_t &level,uint8_t &unit,volatile T err[6],bool workflow){
 
     if(TIX<6)
         err[TIX] = 0.0;
 
     auto grid_idx_x = BIX % sam_bgs.x;
     auto grid_idx_y = (BIX / sam_bgs.x) % sam_bgs.y;
     auto grid_idx_z = (BIX / sam_bgs.x) / sam_bgs.y;
     global_starts.x = sam_starts.x + grid_idx_x * sam_strides.x;
     global_starts.y = sam_starts.y + grid_idx_y * sam_strides.y;
     global_starts.z = sam_starts.z + grid_idx_z * sam_strides.z;
     if(workflow==SPLINE3_PRED_ATT){
         bool use_natural=false,use_md=false,reverse=false;
         if (BIY==0){
             level = 2;
             //use_natural= false;
             //use_md = BIY == 2;
             //reverse = BIY % 3;
 
         }
         else if (BIY <3){
             level = 1;
             use_natural= (BIY==2);
             //use_md = BIY == 5;
             //reverse = BIY % 3;
 
         }
         else{
             level = 0;
             use_natural = BIY > 5;
             use_md = (BIY == 5 or BIY == 8);
             reverse = BIY % 3;
         }
        
         intp_param.use_natural[level]=use_natural;
         intp_param.use_md[level]=use_md;
         intp_param.reverse[level]=reverse;
     }
     else{
         level = 0;
         if(BIY==0){
             intp_param.alpha = 1.0;
             intp_param.beta = 2.0;
         }
         else if (BIY==1){
             intp_param.alpha = 1.25;
             intp_param.beta = 2.0;
         }
         else{
             intp_param.alpha = 1.5+0.25*((BIY-2)/3);
             intp_param.beta = 2.0+((BIY-2)%3);
         }
     }
     unit = 1<<level;
 
         //printf("%d %d %d %d %d %d %d\n",global_starts.x,global_starts.y,global_starts.z,level,use_natural,use_md,reverse);
     
     __syncthreads(); 
     
 }
 
 template <typename T1, typename T2, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
 __device__ void global2shmem_17x17x17data_att(T1* data, DIM3 data_size, STRIDE3 data_leap, volatile T2 s_data[17][17][17], DIM3 global_starts, uint8_t unit)
 {
     constexpr auto TOTAL = BLOCK17 * BLOCK17 * BLOCK17;
 
     //if(TIX==0){
     //    printf("%d %d %d %d %d %d\n",data_size.x,data_size.y,data_size.z,data_leap.x,data_leap.y,data_leap.z);
     //}
 
     for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
         auto x   = (_tix % BLOCK17);
         auto y   = (_tix / BLOCK17) % BLOCK17;
         auto z   = (_tix / BLOCK17) / BLOCK17;
         auto gx  = (x + global_starts.x);
         auto gy  = (y + global_starts.y);
         auto gz  = (z + global_starts.z);
         auto gid = gx + gy * data_leap.y + gz * data_leap.z;
 
         if (gx < data_size.x and gy < data_size.y and gz < data_size.z  ) s_data[z][y][x] = data[gid];
 /*
         if(BIX == 7 and BIY == 47 and BIZ == 15 and x==10 and y==8 and z==4){
             printf("g2s1084 %d %d %d %d %.2e %.2e \n",gx,gy,gz,gid,s_data[z][y][x],data[gid]);
         }
 
         if(BIX == 7 and BIY == 47 and BIZ == 15 and x==10 and y==4 and z==8){
             printf("g2s1048 %d %d %d %d %.2e %.2e \n",gx,gy,gz,gid,s_data[z][y][x],data[gid]);
         }*/
     }
     //if(TIX==0){
    //     printf("%d %d Done\n",BIX,BIY);
    // }
     __syncthreads();
 }
 
 
 template <
     typename T,
     typename FP,
     typename LAMBDAX,
     typename LAMBDAY,
     typename LAMBDAZ,
     bool BLUE,
     bool YELLOW,
     bool HOLLOW,
     int  LINEAR_BLOCK_SIZE,
     int  BLOCK_DIMX,
     int  BLOCK_DIMY,
     bool COARSEN,
     int  BLOCK_DIMZ,
     bool BORDER_INCLUSIVE,
     bool WORKFLOW
     >
 __forceinline__ __device__ void interpolate_stage_att(
     volatile T s_data[17][17][17],
     DIM3    data_size,
      DIM3    global_starts,
     LAMBDAX     xmap,
     LAMBDAY     ymap,
     LAMBDAZ     zmap,
     int         unit,
     FP eb_r,
     FP ebx2,
     bool interpolator,
     volatile T* error)
 {
     static_assert(BLOCK_DIMX * BLOCK_DIMY * (COARSEN ? 1 : BLOCK_DIMZ) <= 384, "block oversized");
     static_assert((BLUE or YELLOW or HOLLOW) == true, "must be one hot");
     static_assert((BLUE and YELLOW) == false, "must be only one hot (1)");
     static_assert((BLUE and YELLOW) == false, "must be only one hot (2)");
     static_assert((YELLOW and HOLLOW) == false, "must be only one hot (3)");
     //DIM3 global_starts (global_starts_v.x,global_starts_v.y, global_starts_v.z);
     auto run = [&](auto x, auto y, auto z) {
 
         
 
         if (xyz17x17x17_predicate_att<BORDER_INCLUSIVE>(x, y, z,data_size, global_starts)) {
             T pred = 0;
 
             //if(BIX == 7 and BIY == 47 and BIZ == 15 and unit==4 and (CONSTEXPR (YELLOW)) )
             //    printf("%d %d %d\n",x,y,z);
             /*
              if(BIX == 7 and BIY == 47 and BIZ == 15 and unit==4 and x==4 and y==4 and z==4)
                         printf("444 %.2e %.2e \n",s_data[z - unit][y][x],s_data[z + unit][y][x]);
 
             if(BIX == 7 and BIY == 47 and BIZ == 15 and unit==4 and x==4 and y==4 and z==0)
                         printf("440 %.2e %.2e \n",s_data[z][y - unit][x],s_data[z][y + unit][x]);
             if(BIX == 7 and BIY == 47 and BIZ == 15 and unit==4 and x==4 and y==8 and z==0)
                         printf("480 %.2e %.2e \n",s_data[z][y ][x- unit],s_data[z][y ][x+ unit]);*/
                   //  }
             auto global_x=global_starts.x+x, global_y=global_starts.y+y, global_z=global_starts.z+z;
             /*
             int interpolation_coeff_set1[2]={-1,-3};
             int interpolation_coeff_set2[2]={9,23};
             int interpolation_coeff_set3[2]={16,40};
             auto a=interpolation_coeff_set1[interpolator];
             auto b=interpolation_coeff_set2[interpolator];
             auto c=interpolation_coeff_set3[interpolator];
             */
             if(interpolator==0){
                 if CONSTEXPR (BLUE) {  //
 
                     if(global_starts.z+BLOCK16<data_size.z){
 
                         if(z>=3*unit and z+3*unit<=BLOCK16  )
                             pred = (-s_data[z - 3*unit][y][x]+9*s_data[z - unit][y][x] + 9*s_data[z + unit][y][x]-s_data[z + 3*unit][y][x]) / 16;
                         else if (z+3*unit<=BLOCK16)
                             pred = (3*s_data[z - unit][y][x] + 6*s_data[z + unit][y][x]-s_data[z + 3*unit][y][x]) / 8;
                         else if (z>=3*unit)
                             pred = (-s_data[z - 3*unit][y][x]+6*s_data[z - unit][y][x] + 3*s_data[z + unit][y][x]) / 8;
 
                         else
                             pred = (s_data[z - unit][y][x] + s_data[z + unit][y][x]) / 2;
                     }
                     else{
                         if(z>=3*unit){
                             if(z+3*unit<=BLOCK16 and global_z+3*unit<data_size.z)
                                 pred = (-s_data[z - 3*unit][y][x]+9*s_data[z - unit][y][x] + 9*s_data[z + unit][y][x]-s_data[z + 3*unit][y][x]) / 16;
                             else if (global_z+unit<data_size.z)
                                 pred = (-s_data[z - 3*unit][y][x]+6*s_data[z - unit][y][x] + 3*s_data[z + unit][y][x]) / 8;
                             else
                                 pred=s_data[z - unit][y][x];
 
                         }
                         else{
                             if(z+3*unit<=BLOCK16 and global_z+3*unit<data_size.z)
                                 pred = (3*s_data[z - unit][y][x] + 6*s_data[z + unit][y][x]-s_data[z + 3*unit][y][x]) / 8;
                             else if (global_z+unit<data_size.z)
                                 pred = (s_data[z - unit][y][x] + s_data[z + unit][y][x]) / 2;
                             else
                                 pred=s_data[z - unit][y][x];
                         } 
                     }
                 }
                 if CONSTEXPR (YELLOW) {  //
                    // if(BIX == 5 and BIY == 22 and BIZ == 6 and unit==1 and x==29 and y==7 and z==0){
                    //     printf("%.2e %.2e %.2e %.2e\n",s_data[z ][y- 3*unit][x],s_data[z ][y- unit][x],s_data[z ][y+ unit][x]);
                   //  }
                     if(global_starts.y+BLOCK16<data_size.y){
                         if(y>=3*unit and y+3*unit<=BLOCK16 )
                             pred = (-s_data[z ][y- 3*unit][x]+9*s_data[z ][y- unit][x] + 9*s_data[z ][y+ unit][x]-s_data[z][y + 3*unit][x]) / 16;
                         else if (y+3*unit<=BLOCK16)
                             pred = (3*s_data[z ][y - unit][x] + 6*s_data[z][y + unit][x]-s_data[z][y + 3*unit][x]) / 8;
                         else if (y>=3*unit)
                             pred = (-s_data[z ][y- 3*unit][x]+6*s_data[z][y - unit][x] + 3*s_data[z][y + unit][x]) / 8;
                         else
                             pred = (s_data[z][y - unit][x] + s_data[z][y + unit][x]) / 2;
                     }
                     else{
                         if(y>=3*unit){
                             if(y+3*unit<=BLOCK16 and global_y+3*unit<data_size.y)
                                 pred = (-s_data[z ][y- 3*unit][x]+9*s_data[z][y - unit][x] + 9*s_data[z ][y+ unit][x]-s_data[z ][y+ 3*unit][x]) / 16;
                             else if (global_y+unit<data_size.y)
                                 pred = (-s_data[z ][y- 3*unit][x]+6*s_data[z ][y- unit][x] + 3*s_data[z ][y+ unit][x]) / 8;
                             else
                                 pred=s_data[z ][y- unit][x];
 
                         }
                         else{
                             if(y+3*unit<=BLOCK16 and global_y+3*unit<data_size.y)
                                 pred = (3*s_data[z][y - unit][x] + 6*s_data[z ][y+ unit][x]-s_data[z][y + 3*unit][x]) / 8;
                             else if (global_y+unit<data_size.y)
                                 pred = (s_data[z ][y- unit][x] + s_data[z][y + unit][x]) / 2;
                             else
                                 pred=s_data[z ][y- unit][x];
                         } 
                     }
                 }
 
                 if CONSTEXPR (HOLLOW) {  //
                     //if(BIX == 5 and BIY == 22 and BIZ == 6 and unit==1)
                     //    printf("%d %d %d\n",x,y,z);
                     if(global_starts.x+BLOCK16<data_size.x){
                         if(x>=3*unit and x+3*unit<=BLOCK16 )
                             pred = (-s_data[z ][y][x- 3*unit]+9*s_data[z ][y][x- unit] + 9*s_data[z ][y][x+ unit]-s_data[z ][y][x + 3*unit]) / 16;
                         else if (x+3*unit<=BLOCK16)
                             pred = (3*s_data[z ][y][x- unit] + 6*s_data[z ][y][x + unit]-s_data[z][y][x + 3*unit]) / 8;
                         else if (x>=3*unit)
                             pred = (-s_data[z][y][x - 3*unit]+6*s_data[z][y][x - unit] + 3*s_data[z ][y][x + unit]) / 8;
                         else
                             pred = (s_data[z][y][x - unit] + s_data[z][y][x + unit]) / 2;
                     }
                     else{
                         if(x>=3*unit){
                             if(x+3*unit<=BLOCK16 and global_x+3*unit<data_size.x)
                                 pred = (-s_data[z ][y][x- 3*unit]+9*s_data[z][y ][x- unit] + 9*s_data[z ][y][x+ unit]-s_data[z ][y][x+ 3*unit]) / 16;
                             else if (global_x+unit<data_size.x)
                                 pred = (-s_data[z ][y][x- 3*unit]+6*s_data[z ][y][x- unit] + 3*s_data[z ][y][x+ unit]) / 8;
                             else
                                 pred=s_data[z ][y][x- unit];
 
                         }
                         else{
                             if(x+3*unit<=BLOCK16 and global_x+3*unit<data_size.x)
                                 pred = (3*s_data[z][y ][x- unit] + 6*s_data[z ][y][x+ unit]-s_data[z][y ][x+ 3*unit]) / 8;
                             else if (global_x+unit<data_size.x)
                                 pred = (s_data[z ][y][x- unit] + s_data[z][y ][x+ unit]) / 2;
                             else
                                 pred=s_data[z ][y][x- unit];
                         } 
                     }
                 }
 
             }
             else{
                 if CONSTEXPR (BLUE) {  //
 
                     if(global_starts.z+BLOCK16<data_size.z){
 
                         if(z>=3*unit and z+3*unit<=BLOCK16  )
                             pred = (-3*s_data[z - 3*unit][y][x]+23*s_data[z - unit][y][x] + 23*s_data[z + unit][y][x]-3*s_data[z + 3*unit][y][x]) / 40;
                         else if (z+3*unit<=BLOCK16)
                             pred = (3*s_data[z - unit][y][x] + 6*s_data[z + unit][y][x]-s_data[z + 3*unit][y][x]) / 8;
                         else if (z>=3*unit)
                             pred = (-s_data[z - 3*unit][y][x]+6*s_data[z - unit][y][x] + 3*s_data[z + unit][y][x]) / 8;
 
                         else
                             pred = (s_data[z - unit][y][x] + s_data[z + unit][y][x]) / 2;
                     }
                     else{
                         if(z>=3*unit){
                             if(z+3*unit<=BLOCK16 and global_z+3*unit<data_size.z)
                                 pred = (-3*s_data[z - 3*unit][y][x]+23*s_data[z - unit][y][x] + 23*s_data[z + unit][y][x]-3*s_data[z + 3*unit][y][x]) / 40;
                             else if (global_z+unit<data_size.z)
                                 pred = (-s_data[z - 3*unit][y][x]+6*s_data[z - unit][y][x] + 3*s_data[z + unit][y][x]) / 8;
                             else
                                 pred=s_data[z - unit][y][x];
 
                         }
                         else{
                             if(z+3*unit<=BLOCK16 and global_z+3*unit<data_size.z)
                                 pred = (3*s_data[z - unit][y][x] + 6*s_data[z + unit][y][x]-s_data[z + 3*unit][y][x]) / 8;
                             else if (global_z+unit<data_size.z)
                                 pred = (s_data[z - unit][y][x] + s_data[z + unit][y][x]) / 2;
                             else
                                 pred=s_data[z - unit][y][x];
                         } 
                     }
                 }
                 if CONSTEXPR (YELLOW) {  //
                    // if(BIX == 5 and BIY == 22 and BIZ == 6 and unit==1 and x==29 and y==7 and z==0){
                    //     printf("%.2e %.2e %.2e %.2e\n",s_data[z ][y- 3*unit][x],s_data[z ][y- unit][x],s_data[z ][y+ unit][x]);
                   //  }
                     if(global_starts.y+BLOCK16<data_size.y){
                         if(y>=3*unit and y+3*unit<=BLOCK16 )
                             pred = (-3*s_data[z ][y- 3*unit][x]+23*s_data[z ][y- unit][x] + 23*s_data[z ][y+ unit][x]-3*s_data[z][y + 3*unit][x]) / 40;
                         else if (y+3*unit<=BLOCK16)
                             pred = (3*s_data[z ][y - unit][x] + 6*s_data[z][y + unit][x]-s_data[z][y + 3*unit][x]) / 8;
                         else if (y>=3*unit)
                             pred = (-s_data[z ][y- 3*unit][x]+6*s_data[z][y - unit][x] + 3*s_data[z][y + unit][x]) / 8;
                         else
                             pred = (s_data[z][y - unit][x] + s_data[z][y + unit][x]) / 2;
                     }
                     else{
                         if(y>=3*unit){
                             if(y+3*unit<=BLOCK16 and global_y+3*unit<data_size.y)
                                 pred = (-3*s_data[z ][y- 3*unit][x]+23*s_data[z][y - unit][x] + 23*s_data[z ][y+ unit][x]-3*s_data[z ][y+ 3*unit][x]) / 40;
                             else if (global_y+unit<data_size.y)
                                 pred = (-s_data[z ][y- 3*unit][x]+6*s_data[z ][y- unit][x] + 3*s_data[z ][y+ unit][x]) / 8;
                             else
                                 pred=s_data[z ][y- unit][x];
 
                         }
                         else{
                             if(y+3*unit<=BLOCK16 and global_y+3*unit<data_size.y)
                                 pred = (3*s_data[z][y - unit][x] + 6*s_data[z ][y+ unit][x]-s_data[z][y + 3*unit][x]) / 8;
                             else if (global_y+unit<data_size.y)
                                 pred = (s_data[z ][y- unit][x] + s_data[z][y + unit][x]) / 2;
                             else
                                 pred=s_data[z ][y- unit][x];
                         } 
                     }
                 }
 
                 if CONSTEXPR (HOLLOW) {  //
                     //if(BIX == 5 and BIY == 22 and BIZ == 6 and unit==1)
                     //    printf("%d %d %d\n",x,y,z);
                     if(global_starts.x+BLOCK16<data_size.x){
                         if(x>=3*unit and x+3*unit<=BLOCK16 )
                             pred = (-3*s_data[z ][y][x- 3*unit]+23*s_data[z ][y][x- unit] + 23*s_data[z ][y][x+ unit]-3*s_data[z ][y][x + 3*unit]) / 40;
                         else if (x+3*unit<=BLOCK16)
                             pred = (3*s_data[z ][y][x- unit] + 6*s_data[z ][y][x + unit]-s_data[z][y][x + 3*unit]) / 8;
                         else if (x>=3*unit)
                             pred = (-s_data[z][y][x - 3*unit]+6*s_data[z][y][x - unit] + 3*s_data[z ][y][x + unit]) / 8;
                         else
                             pred = (s_data[z][y][x - unit] + s_data[z][y][x + unit]) / 2;
                     }
                     else{
                         if(x>=3*unit){
                             if(x+3*unit<=BLOCK16 and global_x+3*unit<data_size.x)
                                 pred = (-3*s_data[z ][y][x- 3*unit]+23*s_data[z][y ][x- unit] + 23*s_data[z ][y][x+ unit]-3*s_data[z ][y][x+ 3*unit]) / 40;
                             else if (global_x+unit<data_size.x)
                                 pred = (-s_data[z ][y][x- 3*unit]+6*s_data[z ][y][x- unit] + 3*s_data[z ][y][x+ unit]) / 8;
                             else
                                 pred=s_data[z ][y][x- unit];
 
                         }
                         else{
                             if(x+3*unit<=BLOCK16 and global_x+3*unit<data_size.x)
                                 pred = (3*s_data[z][y ][x- unit] + 6*s_data[z ][y][x+ unit]-s_data[z][y ][x+ 3*unit]) / 8;
                             else if (global_x+unit<data_size.x)
                                 pred = (s_data[z ][y][x- unit] + s_data[z][y ][x+ unit]) / 2;
                             else
                                 pred=s_data[z ][y][x- unit];
                         } 
                     }
                 }
             }
 
              if CONSTEXPR (WORKFLOW == SPLINE3_AB_ATT) {
                 
                 auto          err = s_data[z][y][x] - pred;
                 decltype(err) code;
                 // TODO unsafe, did not deal with the out-of-cap case
                 {
                     code = fabs(err) * eb_r + 1;
                     code = err < 0 ? -code : code;
                     code = int(code / 2) ;
                 }
                 //s_ectrl[z][y][x] = code;  // TODO double check if unsigned type works
                 /*
                  if(BIX == 12 and BIY == 12 and BIZ == 8 and unit==4 and x==0 and y==0 and z==4)
                         printf("004pred %.2e %.2e %.2e %.2e %.2e %.2e\n",pred,code,s_data[z][y][x],s_data[0][0][0],s_data[0][0][8],s_data[0][0][16]);
                     if(BIX == 12 and BIY == 12 and BIZ == 8 and unit==4 and x==8 and y==8 and z==4)
                         printf("884pred %.2e %.2e %.2e %.2e %.2e %.2e\n",pred,code,s_data[z][y][x],s_data[8][8][0],s_data[8][8][8],s_data[8][8][16]);
                   */      
                // if(fabs(pred)>=3)
                //     printf("%d %d %d %d %d %d %d %d %d %d %.2e %.2e %.2e\n",unit,CONSTEXPR (BLUE),CONSTEXPR (YELLOW),CONSTEXPR (HOLLOW),BIX,BIY,BIZ,x,y,z,pred,code,s_data[z][y][x]);
               
                 s_data[z][y][x]  = pred + code * ebx2;
                 atomicAdd(const_cast<T*>(error),code!=0);
                 
 
             }
 
 
             else{
                 atomicAdd(const_cast<T*>(error),fabs(s_data[z][y][x]-pred));
                 /*
                 auto          err = s_data[z][y][x] - pred;
                 atomicAdd(const_cast<T*>(error),fabs(err));
                 auto delta = ebx2 * int((fabs(err) * eb_r + 1) / 2) - err;
                 s_data[z][y][x] += delta;
                 */
             }
            // if(BIX ==30 and BIY>=0 and BIY < 3 and x == 8 and y ==8 and z ==8){
            //     printf("888 %d %.4e %.4e\n",BIY,s_data[z][y][x],pred);
            // }
             //atomicAdd(const_cast<T*>(error),1.0);
         }
     };
     // -------------------------------------------------------------------------------- //
 
     if CONSTEXPR (COARSEN) {
         constexpr auto TOTAL = BLOCK_DIMX * BLOCK_DIMY * BLOCK_DIMZ;
         //if( BLOCK_DIMX *BLOCK_DIMY<= LINEAR_BLOCK_SIZE){
             for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
                 auto itix = (_tix % BLOCK_DIMX);
                 auto itiy = (_tix / BLOCK_DIMX) % BLOCK_DIMY;
                 auto itiz = (_tix / BLOCK_DIMX) / BLOCK_DIMY;
                 auto x    = xmap(itix, unit);
                 auto y    = ymap(itiy, unit);
                 auto z    = zmap(itiz, unit);
                 
                 run(x, y, z);
             }
         //}
         //may have bug    
         /*
         else{
             for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
                 auto itix = (_tix % BLOCK_DIMX);
                 auto itiz = (_tix / BLOCK_DIMX) % BLOCK_DIMZ;
                 auto itiy = (_tix / BLOCK_DIMX) / BLOCK_DIMZ;
                 auto x    = xmap(itix, unit);
                 auto y    = ymap(itiy, unit);
                 auto z    = zmap(itiz, unit);
                 run(x, y, z);
             }
         }*/
         //may have bug  end
         
     }
     else {
         auto itix = (TIX % BLOCK_DIMX);
         auto itiy = (TIX / BLOCK_DIMX) % BLOCK_DIMY;
         auto itiz = (TIX / BLOCK_DIMX) / BLOCK_DIMY;
         auto x    = xmap(itix, unit);
         auto y    = ymap(itiy, unit);
         auto z    = zmap(itiz, unit);
 
         
 
      //   printf("%d %d %d\n", x,y,z);
         run(x, y, z);
     }
     __syncthreads();
 }
 
 template <
     typename T,
     typename FP,
     //typename LAMBDAX,
     //typename LAMBDAY,
     //typename LAMBDAZ,
     typename LAMBDA,
     bool LINE,
     bool FACE,
     bool CUBE,
     int  LINEAR_BLOCK_SIZE,
     int NUM_ELE,
     //int  BLOCK_DIMX,
     //int  BLOCK_DIMY,
     bool COARSEN,
     //int  BLOCK_DIMZ,
     bool BORDER_INCLUSIVE,
     bool WORKFLOW,
     typename INTERP>
 __forceinline__ __device__ void interpolate_stage_md_att(
     volatile T s_data[17][17][17],
     DIM3    data_size,
     DIM3    global_starts,
     //LAMBDAX     xmap,
     //LAMBDAY     ymap,
     //LAMBDAZ     zmap,
     LAMBDA xyzmap,
     int         unit,
     FP eb_r,
     FP ebx2,
     INTERP cubic_interpolator,
     volatile T* error)
 {
     static_assert(COARSEN or (NUM_ELE <= 384), "block oversized");
     static_assert((LINE or FACE or CUBE) == true, "must be one hot");
     static_assert((LINE and FACE) == false, "must be only one hot (1)");
     static_assert((LINE and CUBE) == false, "must be only one hot (2)");
     static_assert((FACE and CUBE) == false, "must be only one hot (3)");
     //DIM3 global_starts (global_starts_v.x,global_starts_v.y, global_starts_v.z);
     auto run = [&](auto x, auto y, auto z) {
 
         
 
         if (xyz17x17x17_predicate_att<BORDER_INCLUSIVE>(x, y, z,data_size, global_starts)) {
             T pred = 0;
 
             //if(BIX == 23 and BIY == 23 and BIZ == 15 and unit==2)
             //    printf("%d %d %d\n",x,y,z);
             /*
              if(BIX == 7 and BIY == 47 and BIZ == 15 and unit==4 and x==4 and y==4 and z==4)
                         printf("444 %.2e %.2e \n",s_data[z - unit][y][x],s_data[z + unit][y][x]);
 
             if(BIX == 7 and BIY == 47 and BIZ == 15 and unit==4 and x==4 and y==4 and z==0)
                         printf("440 %.2e %.2e \n",s_data[z][y - unit][x],s_data[z][y + unit][x]);
             if(BIX == 7 and BIY == 47 and BIZ == 15 and unit==4 and x==4 and y==8 and z==0)
                         printf("480 %.2e %.2e \n",s_data[z][y ][x- unit],s_data[z][y ][x+ unit]);*/
                   //  }
             auto global_x=global_starts.x+x, global_y=global_starts.y+y, global_z=global_starts.z+z;
             /*
             int interpolation_coeff_set1[2]={-1,-3};
             int interpolation_coeff_set2[2]={9,23};
             int interpolation_coeff_set3[2]={16,40};
             auto a=interpolation_coeff_set1[interpolator];
             auto b=interpolation_coeff_set2[interpolator];
             auto c=interpolation_coeff_set3[interpolator];
             */
            
             if CONSTEXPR (LINE) {  //
                 //bool I_X = x&1; 
                 bool I_Y = (y % (2*unit) )> 0; 
                 bool I_Z = (z % (2*unit) )> 0; 
                 if (I_Z){
                     //assert(x&1==0 and y&1==0);
 
                     if(global_starts.z+BLOCK16 < data_size.z){
 
                         if(z>=3*unit and z+3*unit<=BLOCK16  )
                             pred = cubic_interpolator(s_data[z - 3*unit][y][x],s_data[z - unit][y][x],s_data[z + unit][y][x],s_data[z + 3*unit][y][x]);
                         else if (z+3*unit<=BLOCK16)
                             pred = (3*s_data[z - unit][y][x] + 6*s_data[z + unit][y][x]-s_data[z + 3*unit][y][x]) / 8;
                         else if (z>=3*unit)
                             pred = (-s_data[z - 3*unit][y][x]+6*s_data[z - unit][y][x] + 3*s_data[z + unit][y][x]) / 8;
 
                         else
                             pred = (s_data[z - unit][y][x] + s_data[z + unit][y][x]) / 2;
                     }
                     else{
                         if(z>=3*unit){
                             if(z+3*unit<=BLOCK16 and global_z+3*unit<data_size.z)
                                 pred = cubic_interpolator(s_data[z - 3*unit][y][x],s_data[z - unit][y][x] ,s_data[z + unit][y][x],s_data[z + 3*unit][y][x]);
                             else if (global_z+unit<data_size.z)
                                 pred = (-s_data[z - 3*unit][y][x]+6*s_data[z - unit][y][x] + 3*s_data[z + unit][y][x]) / 8;
                             else
                                 pred=s_data[z - unit][y][x];
 
                         }
                         else{
                             if(z+3*unit<=BLOCK16 and global_z+3*unit<data_size.z)
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
                     if(global_starts.y+BLOCK16 < data_size.y){
                         if(y>=3*unit and y+3*unit<=BLOCK16 )
                             pred = cubic_interpolator(s_data[z ][y- 3*unit][x],s_data[z ][y- unit][x] ,s_data[z ][y+ unit][x],s_data[z][y + 3*unit][x]) ;
                         else if (y+3*unit<=BLOCK16)
                             pred = (3*s_data[z ][y - unit][x] + 6*s_data[z][y + unit][x]-s_data[z][y + 3*unit][x]) / 8;
                         else if (y>=3*unit)
                             pred = (-s_data[z ][y- 3*unit][x]+6*s_data[z][y - unit][x] + 3*s_data[z][y + unit][x]) / 8;
                         else
                             pred = (s_data[z][y - unit][x] + s_data[z][y + unit][x]) / 2;
                     }
                     else{
                         if(y>=3*unit){
                             if(y+3*unit<=BLOCK16 and global_y+3*unit<data_size.y)
                                 pred = cubic_interpolator(s_data[z ][y- 3*unit][x],s_data[z][y - unit][x],s_data[z ][y+ unit][x],s_data[z ][y+ 3*unit][x]);
                             else if (global_y+unit<data_size.y)
                                 pred = (-s_data[z ][y- 3*unit][x]+6*s_data[z ][y- unit][x] + 3*s_data[z ][y+ unit][x]) / 8;
                             else
                                 pred=s_data[z ][y- unit][x];
 
                         }
                         else{
                             if(y+3*unit<=BLOCK16 and global_y+3*unit<data_size.y)
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
                     if(global_starts.x+BLOCK16 < data_size.x){
                         if(x>=3*unit and x+3*unit<=BLOCK16 )
                             pred = cubic_interpolator(s_data[z ][y][x- 3*unit],s_data[z ][y][x- unit],s_data[z ][y][x+ unit],s_data[z ][y][x + 3*unit]);
                         else if (x+3*unit<=BLOCK16)
                             pred = (3*s_data[z ][y][x- unit] + 6*s_data[z ][y][x + unit]-s_data[z][y][x + 3*unit]) / 8;
                         else if (x>=3*unit)
                             pred = (-s_data[z][y][x - 3*unit]+6*s_data[z][y][x - unit] + 3*s_data[z ][y][x + unit]) / 8;
                         else
                             pred = (s_data[z][y][x - unit] + s_data[z][y][x + unit]) / 2;
                     }
                     else{
                         if(x>=3*unit){
                             if(x+3*unit<=BLOCK16 and global_x+3*unit<data_size.x)
                                 pred = cubic_interpolator(s_data[z ][y][x- 3*unit],s_data[z][y ][x- unit],s_data[z ][y][x+ unit],s_data[z ][y][x+ 3*unit]);
                             else if (global_x+unit<data_size.x)
                                 pred = (-s_data[z ][y][x- 3*unit]+6*s_data[z ][y][x- unit] + 3*s_data[z ][y][x+ unit]) / 8;
                             else
                                 pred=s_data[z ][y][x- unit];
 
                         }
                         else{
                             if(x+3*unit<=BLOCK16 and global_x+3*unit<data_size.x)
                                 pred = (3*s_data[z][y ][x- unit] + 6*s_data[z ][y][x+ unit]-s_data[z][y ][x+ 3*unit]) / 8;
                             else if (global_x+unit<data_size.x)
                                 pred = (s_data[z ][y][x- unit] + s_data[z][y ][x+ unit]) / 2;
                             else
                                 pred=s_data[z ][y][x- unit];
                         } 
                     }
 
                 }
             }
             auto get_interp_order = [&](auto x, auto gx, auto gs){
                 int b = x >= 3*unit ? 3 : 1;
                 int f = 0;
                 if(x+3*unit<=BLOCK16 and gx+3*unit < gs )
                     f = 3;
                 else if (gx+unit < gs)
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
                // if(BIX == 5 and BIY == 22 and BIZ == 6 and unit==1 and x==29 and y==7 and z==0){
                //     printf("%.2e %.2e %.2e %.2e\n",s_data[z ][y- 3*unit][x],s_data[z ][y- unit][x],s_data[z ][y+ unit][x]);
               //  }
 
                 bool I_YZ = (x % (2*unit) ) == 0;
                 bool I_XZ = (y % (2*unit ) )== 0;
 
                 //if(BIX == 10 and BIY == 12 and BIZ == 0 and x==13 and y==6 and z==9)
                //     printf("face %d %d\n", I_YZ,I_XZ);
 
                  
                 if (I_YZ){
 
 
                     auto interp_z = get_interp_order(z,global_z,data_size.z);
                     auto interp_y = get_interp_order(y,global_y,data_size.y);
 
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
                     auto interp_z = get_interp_order(z,global_z,data_size.z);
                     auto interp_x = get_interp_order(x,global_x,data_size.x);
 
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
 
                     auto interp_y = get_interp_order(y,global_y,data_size.y);
                     auto interp_x = get_interp_order(x,global_x,data_size.x);
 
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
                 auto interp_z = get_interp_order(z,global_z,data_size.z);
                 auto interp_y = get_interp_order(y,global_y,data_size.y);
                 auto interp_x = get_interp_order(x,global_x,data_size.x);
 
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
             if CONSTEXPR (WORKFLOW == SPLINE3_AB_ATT) {
                 
                 auto          err = s_data[z][y][x] - pred;
                 decltype(err) code;
                 // TODO unsafe, did not deal with the out-of-cap case
                 {
                     code = fabs(err) * eb_r + 1;
                     code = err < 0 ? -code : code;
                     code = int(code / 2) ;
                 }
                 //s_ectrl[z][y][x] = code;  // TODO double check if unsigned type works
                 /*
                  if(BIX == 12 and BIY == 12 and BIZ == 8 and unit==4 and x==0 and y==0 and z==4)
                         printf("004pred %.2e %.2e %.2e %.2e %.2e %.2e\n",pred,code,s_data[z][y][x],s_data[0][0][0],s_data[0][0][8],s_data[0][0][16]);
                     if(BIX == 12 and BIY == 12 and BIZ == 8 and unit==4 and x==8 and y==8 and z==4)
                         printf("884pred %.2e %.2e %.2e %.2e %.2e %.2e\n",pred,code,s_data[z][y][x],s_data[8][8][0],s_data[8][8][8],s_data[8][8][16]);
                   */      
                // if(fabs(pred)>=3)
                //     printf("%d %d %d %d %d %d %d %d %d %d %.2e %.2e %.2e\n",unit,CONSTEXPR (BLUE),CONSTEXPR (YELLOW),CONSTEXPR (HOLLOW),BIX,BIY,BIZ,x,y,z,pred,code,s_data[z][y][x]);
               
                 s_data[z][y][x]  = pred + code * ebx2;
                 atomicAdd(const_cast<T*>(error),code!=0);
                 
 
             }
 
 
             else{
 
                 atomicAdd(const_cast<T*>(error),fabs(s_data[z][y][x]-pred));
                 /*
                 auto          err = s_data[z][y][x] - pred;
                 atomicAdd(const_cast<T*>(error),fabs(err));
                 auto delta = ebx2 * int((fabs(err) * eb_r + 1) / 2) - err;
                 s_data[z][y][x] += delta;
                 */
                 //s_ectrl[z][y][x] = code;  // TODO double check if unsigned type works
                 /*
                  if(BIX == 12 and BIY == 12 and BIZ == 8 and unit==4 and x==0 and y==0 and z==4)
                         printf("004pred %.2e %.2e %.2e %.2e %.2e %.2e\n",pred,code,s_data[z][y][x],s_data[0][0][0],s_data[0][0][8],s_data[0][0][16]);
                     if(BIX == 12 and BIY == 12 and BIZ == 8 and unit==4 and x==8 and y==8 and z==4)
                         printf("884pred %.2e %.2e %.2e %.2e %.2e %.2e\n",pred,code,s_data[z][y][x],s_data[8][8][0],s_data[8][8][8],s_data[8][8][16]);
                   */      
                // if(fabs(pred)>=3)
                //     printf("%d %d %d %d %d %d %d %d %d %d %.2e %.2e %.2e\n",unit,CONSTEXPR (BLUE),CONSTEXPR (YELLOW),CONSTEXPR (HOLLOW),BIX,BIY,BIZ,x,y,z,pred,code,s_data[z][y][x]);
               
                 
             }
            // if(BIX == 30 and BIY>=0 and BIY < 3 and x == 8 and y == 8 and z == 8){
             //    printf("888 %d %.4e %.4e\n",BIY,s_data[z][y][x],pred);
             //}
             //atomicAdd(const_cast<T*>(error),1.0);
         }
     };
     // -------------------------------------------------------------------------------- //
 
     if CONSTEXPR (COARSEN) {
         constexpr auto TOTAL = NUM_ELE;
         //if( BLOCK_DIMX *BLOCK_DIMY<= LINEAR_BLOCK_SIZE){
             for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
                 auto [x,y,z]    = xyzmap(_tix, unit);
                 run(x, y, z);
             }
         //}
         //may have bug    
         /*
         else{
             for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
                 auto itix = (_tix % BLOCK_DIMX);
                 auto itiz = (_tix / BLOCK_DIMX) % BLOCK_DIMZ;
                 auto itiy = (_tix / BLOCK_DIMX) / BLOCK_DIMZ;
                 auto x    = xmap(itix, unit);
                 auto y    = ymap(itiy, unit);
                 auto z    = zmap(itiz, unit);
                 run(x, y, z);
             }
         }*/
         //may have bug  end
         
     }
     else {
         if(TIX<NUM_ELE){
             auto [x,y,z]    = xyzmap(TIX, unit);
             
 
          //   printf("%d %d %d\n", x,y,z);
             run(x, y, z);
         }
     }
     __syncthreads();
 }
 
 
 template <typename T, typename FP,int LINEAR_BLOCK_SIZE, bool WORKFLOW>
 __device__ void cusz::device_api::spline3d_layout2_interpolate_att(
     volatile T s_data[17][17][17],
      DIM3    data_size,
     DIM3 global_starts,FP eb_r, FP ebx2,uint8_t level,INTERPOLATION_PARAMS intp_param,volatile T *error)
 {
     auto xblue = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2); };
     auto yblue = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2); };
     auto zblue = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz * 2 + 1); };
 
     auto xblue_reverse = [] __device__(int _tix, int unit) -> int { return unit * (_tix ); };
     auto yblue_reverse = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy ); };
     auto zblue_reverse = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz * 2 + 1); };
 
     auto xyellow = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2); };
     auto yyellow = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2+1); };
     auto zyellow = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz); };
 
     auto xyellow_reverse = [] __device__(int _tix, int unit) -> int { return unit * (_tix ); };
     auto yyellow_reverse = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2+1); };
     auto zyellow_reverse = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz * 2); };
 
 
     auto xhollow = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2 +1); };
     auto yhollow = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy); };
     auto zhollow = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz); };
 
     auto xhollow_reverse = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2 +1); };
     auto yhollow_reverse = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2); };
     auto zhollow_reverse = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz *2); };
 
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
 
     auto xyzmap_line_16b_4u = [] __device__(int _tix, int unit) -> std::tuple<int,int,int> {
         constexpr auto N = 2;
         constexpr auto L = N*(N+1)*(N+1); 
         constexpr auto Q = (N+1)*(N+1); 
         auto group = _tix / L ;
         auto m = _tix % L ;
         auto i = m / Q;
         auto j = (m % Q) / (N+1);
         auto k = (m % Q) % (N+1);
         if(group==0)
             return std::make_tuple(8*i+4,8*j,8*k);
         else if (group==1)
             return std::make_tuple(8*k,8*i+4,8*j);
         else
             return std::make_tuple(8*j,8*k,8*i+4);
 
     };
 
     auto xyzmap_face_16b_4u = [] __device__(int _tix, int unit) -> std::tuple<int,int,int> {
         constexpr auto N = 2;
         constexpr auto L = N*N*(N+1);
         constexpr auto Q = N*N; 
         auto group = _tix / L ;
         auto m = _tix % L ;
         auto i = m / Q;
         auto j = (m % Q) / N;
         auto k = (m % Q) % N;
         if(group==0)
             return std::make_tuple(8*i,8*j+4,8*k+4);
         else if (group==1)
             return std::make_tuple(8*k+4,8*i,8*j+4);
         else
             return std::make_tuple(8*j+4,8*k+4,8*i);
 
     };
 
      auto xyzmap_cube_16b_4u = [] __device__(int _tix, int unit) -> std::tuple<int,int,int> {
         constexpr auto N = 2;
         constexpr auto Q = N * N; 
         auto i = _tix / Q;
         auto j = (_tix % Q) / N;
         auto k = (_tix % Q) % N;
         return std::make_tuple(8*i+4,8*j+4,8*k+4);
 
     };
 
     auto xyzmap_line_16b_8u = [] __device__(int _tix, int unit) -> std::tuple<int,int,int> {
         constexpr auto N = 1;
         constexpr auto L = N*(N+1)*(N+1); 
         constexpr auto Q = (N+1)*(N+1); 
         auto group = _tix / L ;
         auto m = _tix % L ;
         auto i = m / Q;
         auto j = (m % Q) / (N+1);
         auto k = (m % Q) % (N+1);
         if(group==0)
             return std::make_tuple(16*i+8,16*j,16*k);
         else if (group==1)
             return std::make_tuple(16*k,16*i+8,16*j);
         else
             return std::make_tuple(16*j,16*k,16*i+8);
 
     };
 
     auto xyzmap_face_16b_8u = [] __device__(int _tix, int unit) -> std::tuple<int,int,int> {
         constexpr auto N = 1;
         constexpr auto L = N*N*(N+1);
         constexpr auto Q = N*N; 
         auto group = _tix / L ;
         auto m = _tix % L ;
         auto i = m / Q;
         auto j = (m % Q) / N;
         auto k = (m % Q) % N;
         if(group==0)
             return std::make_tuple(16*i,16*j+8,16*k+8);
         else if (group==1)
             return std::make_tuple(16*k+8,16*i,16*j+8);
         else
             return std::make_tuple(16*j+8,16*k+8,16*i);
 
     };
 
      auto xyzmap_cube_16b_8u = [] __device__(int _tix, int unit) -> std::tuple<int,int,int> {
         constexpr auto N = 1;
         constexpr auto Q = N * N; 
         auto i = _tix / Q;
         auto j = (_tix % Q) / N;
         auto k = (_tix % Q) % N;
         return std::make_tuple(16*i+8,16*j+8,16*k+8);
 
     };
 
 
     auto nan_cubic_interp = [] __device__ (T a, T b, T c, T d) -> T{
         return (-a+9*b+9*c-d) / 16;
     };
 
     auto nat_cubic_interp = [] __device__ (T a, T b, T c, T d) -> T{
         return (-3*a+23*b+23*c-3*d) / 40;
     };
 
 
 
 
     constexpr auto COARSEN          = true;
     constexpr auto NO_COARSEN       = false;
     constexpr auto BORDER_INCLUSIVE = true;
     constexpr auto BORDER_EXCLUSIVE = false;
 
     
 
 
 
     
 
 
     int unit;
 
     FP cur_ebx2=ebx2,cur_eb_r=eb_r;
 
 
     auto calc_eb = [&](auto unit) {
         cur_ebx2=ebx2,cur_eb_r=eb_r;
         int temp=1;
         while(temp<unit){
             temp*=2;
             cur_eb_r *= intp_param.alpha;
             cur_ebx2 /= intp_param.alpha;
 
         }
         if(cur_ebx2 < ebx2 / intp_param.beta){
             cur_ebx2 = ebx2 / intp_param.beta;
             cur_eb_r = eb_r * intp_param.beta;
 
         }
     };
 
 
     if(WORKFLOW==SPLINE3_AB_ATT or level==3){
         unit = 8;
         //int m = 2, n = 2, p = 2;//block8 nums;
         if(WORKFLOW==SPLINE3_AB_ATT)
             calc_eb(unit);
         //set_orders(reverse[2]);
         if(intp_param.use_md[3]){
 
 
             if(intp_param.use_natural[3]==0){
 
                 interpolate_stage_md_att<
                     T, FP, decltype(xyzmap_line_16b_8u), //
                     true, false, false, LINEAR_BLOCK_SIZE,12 ,NO_COARSEN, BORDER_INCLUSIVE,WORKFLOW>(
                     s_data,data_size,global_starts, xyzmap_line_16b_8u, unit,cur_eb_r,cur_ebx2, nan_cubic_interp,error);
 
                 interpolate_stage_md_att<
                     T, FP, decltype(xyzmap_face_16b_8u), //
                     false, true, false, LINEAR_BLOCK_SIZE,6 ,NO_COARSEN, BORDER_INCLUSIVE,WORKFLOW>(
                      s_data,data_size,global_starts, xyzmap_face_16b_8u, unit,cur_eb_r,cur_ebx2, nan_cubic_interp,error);
 
                 interpolate_stage_md_att<
                     T, FP, decltype(xyzmap_cube_16b_8u), //
                     false, false, true, LINEAR_BLOCK_SIZE,1 ,NO_COARSEN, BORDER_INCLUSIVE,WORKFLOW>(
                     s_data,data_size,global_starts, xyzmap_cube_16b_8u, unit,cur_eb_r,cur_ebx2, nan_cubic_interp,error);
 
             }
             else{
                 interpolate_stage_md_att<
                     T, FP, decltype(xyzmap_line_16b_8u), //
                     true, false, false, LINEAR_BLOCK_SIZE,12 ,NO_COARSEN, BORDER_INCLUSIVE,WORKFLOW>(
                     s_data,data_size,global_starts, xyzmap_line_16b_8u, unit,cur_eb_r,cur_ebx2, nat_cubic_interp,error);
 
                 interpolate_stage_md_att<
                     T, FP, decltype(xyzmap_face_16b_8u), //
                     false, true, false, LINEAR_BLOCK_SIZE,6 ,NO_COARSEN, BORDER_INCLUSIVE,WORKFLOW>(
                     s_data,data_size,global_starts, xyzmap_face_16b_8u, unit,cur_eb_r,cur_ebx2, nat_cubic_interp,error);
 
                 interpolate_stage_md_att<
                     T, FP, decltype(xyzmap_cube_16b_8u), //
                     false, false, true, LINEAR_BLOCK_SIZE,1 ,NO_COARSEN, BORDER_INCLUSIVE,WORKFLOW>(
                     s_data,data_size,global_starts, xyzmap_cube_16b_8u, unit,cur_eb_r,cur_ebx2, nat_cubic_interp,error);
                 
 
             }
 
         }
         else{
 
             if(intp_param.reverse[3]){
                 interpolate_stage_att<
                     T, FP, decltype(xhollow_reverse), decltype(yhollow_reverse), decltype(zhollow_reverse),  //
                     false, false, true, LINEAR_BLOCK_SIZE, 1, 2, NO_COARSEN, 2, BORDER_INCLUSIVE,WORKFLOW>(
                     s_data,data_size,global_starts, xhollow_reverse, yhollow_reverse, zhollow_reverse, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[3],error);
 
                 interpolate_stage_att<
                     T, FP, decltype(xyellow_reverse), decltype(yyellow_reverse), decltype(zyellow_reverse),  //
                     false, true, false, LINEAR_BLOCK_SIZE, 3, 1, NO_COARSEN, 2, BORDER_INCLUSIVE,WORKFLOW>(
                     s_data,data_size,global_starts, xyellow_reverse, yyellow_reverse, zyellow_reverse, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[3],error);
                 interpolate_stage_att<
                     T, FP, decltype(xblue_reverse), decltype(yblue_reverse), decltype(zblue_reverse),  //
                     true, false, false, LINEAR_BLOCK_SIZE, 3, 3, NO_COARSEN, 1, BORDER_INCLUSIVE,WORKFLOW>(
                     s_data,data_size,global_starts, xblue_reverse, yblue_reverse, zblue_reverse, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[3],error);
 
 
             }
             else{
                 //if( BIX==0 and BIY==0 and BIZ==0)
                // printf("lv3s0\n");
                 interpolate_stage_att<
                     T, FP, decltype(xblue), decltype(yblue), decltype(zblue),  //
                     true, false, false, LINEAR_BLOCK_SIZE, 2, 2, NO_COARSEN, 1, BORDER_INCLUSIVE,WORKFLOW>(
                     s_data,data_size,global_starts, xblue, yblue, zblue, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[3],error);
                // if(BIX==0 and BIY==0 and BIZ==0)
                // printf("lv3s1\n");
                 interpolate_stage_att<
                     T, FP, decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
                     false, true, false, LINEAR_BLOCK_SIZE, 2, 1, NO_COARSEN, 3, BORDER_INCLUSIVE,WORKFLOW>(
                     s_data,data_size,global_starts, xyellow, yyellow, zyellow, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[3],error);
                 //if(BIX==0 and BIY==0 and BIZ==0)
               //  printf("lv3s2\n");
                 interpolate_stage_att<
                     T, FP, decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
                     false, false, true, LINEAR_BLOCK_SIZE, 1, 3, NO_COARSEN, 3, BORDER_INCLUSIVE,WORKFLOW>(
                     s_data,data_size,global_starts, xhollow, yhollow, zhollow, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[3],error);
             }
         }
     }
 
     
     if(WORKFLOW==SPLINE3_AB_ATT or level == 2){
         unit = 4;
         //int m = 2, n = 2, p = 2;//block8 nums;
         if(WORKFLOW==SPLINE3_AB_ATT)
             calc_eb(unit);
         //set_orders(reverse[2]);
 
 
         if(intp_param.use_md[2]){
 
 
             if(intp_param.use_natural[2]==0){
 
                 interpolate_stage_md_att<
                     T, FP, decltype(xyzmap_line_16b_4u), //
                     true, false, false, LINEAR_BLOCK_SIZE,54 ,NO_COARSEN, BORDER_INCLUSIVE,WORKFLOW>(
                     s_data,data_size,global_starts, xyzmap_line_16b_4u, unit,cur_eb_r,cur_ebx2, nan_cubic_interp,error);
 
                 interpolate_stage_md_att<
                     T, FP, decltype(xyzmap_face_16b_4u), //
                     false, true, false, LINEAR_BLOCK_SIZE,36 ,NO_COARSEN, BORDER_INCLUSIVE,WORKFLOW>(
                     s_data,data_size,global_starts, xyzmap_face_16b_4u, unit,cur_eb_r,cur_ebx2, nan_cubic_interp,error);
 
                 interpolate_stage_md_att<
                     T, FP, decltype(xyzmap_cube_16b_4u), //
                     false, false, true, LINEAR_BLOCK_SIZE,8 ,NO_COARSEN, BORDER_INCLUSIVE,WORKFLOW>(
                     s_data,data_size,global_starts, xyzmap_cube_16b_4u, unit,cur_eb_r,cur_ebx2, nan_cubic_interp,error);
 
             }
             else{
                 interpolate_stage_md_att<
                     T, FP, decltype(xyzmap_line_16b_4u), //
                     true, false, false, LINEAR_BLOCK_SIZE,54 ,NO_COARSEN, BORDER_INCLUSIVE,WORKFLOW>(
                     s_data,data_size,global_starts, xyzmap_line_16b_4u, unit,cur_eb_r,cur_ebx2, nat_cubic_interp,error);
 
                 interpolate_stage_md_att<
                     T, FP, decltype(xyzmap_face_16b_4u), //
                     false, true, false, LINEAR_BLOCK_SIZE,36 ,NO_COARSEN, BORDER_INCLUSIVE,WORKFLOW>(
                     s_data,data_size,global_starts, xyzmap_face_16b_4u, unit,cur_eb_r,cur_ebx2, nat_cubic_interp,error);
 
                 interpolate_stage_md_att<
                     T, FP, decltype(xyzmap_cube_16b_4u), //
                     false, false, true, LINEAR_BLOCK_SIZE,8 ,NO_COARSEN, BORDER_INCLUSIVE,WORKFLOW>(
                     s_data,data_size,global_starts, xyzmap_cube_16b_4u, unit,cur_eb_r,cur_ebx2, nat_cubic_interp,error);
                 
 
             }
 
         }
         else{
             if(intp_param.reverse[2]){
                 interpolate_stage_att<
                     T, FP, decltype(xhollow_reverse), decltype(yhollow_reverse), decltype(zhollow_reverse),  //
                     false, false, true, LINEAR_BLOCK_SIZE, 2, 3, NO_COARSEN, 3, BORDER_INCLUSIVE,WORKFLOW>(
                     s_data,data_size,global_starts, xhollow_reverse, yhollow_reverse, zhollow_reverse, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[2],error);
 
                 interpolate_stage_att<
                     T, FP, decltype(xyellow_reverse), decltype(yyellow_reverse), decltype(zyellow_reverse),  //
                     false, true, false, LINEAR_BLOCK_SIZE, 5, 2, NO_COARSEN, 3, BORDER_INCLUSIVE, WORKFLOW>(
                     s_data,data_size,global_starts, xyellow_reverse, yyellow_reverse, zyellow_reverse, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[2],error);
                 interpolate_stage_att<
                     T, FP, decltype(xblue_reverse), decltype(yblue_reverse), decltype(zblue_reverse),  //
                     true, false, false, LINEAR_BLOCK_SIZE, 5, 5, NO_COARSEN, 2, BORDER_INCLUSIVE, WORKFLOW>(
                     s_data,data_size,global_starts,  xblue_reverse, yblue_reverse, zblue_reverse, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[2],error);
 
 
             }
             else{
                 //if( BIX==0 and BIY==0 and BIZ==0)
                // printf("lv3s0\n");
                 interpolate_stage_att<
                     T, FP, decltype(xblue), decltype(yblue), decltype(zblue),  //
                     true, false, false, LINEAR_BLOCK_SIZE, 3, 3, NO_COARSEN, 2, BORDER_INCLUSIVE, WORKFLOW>(
                     s_data,data_size,global_starts, xblue, yblue, zblue, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[2],error);
                // if(BIX==0 and BIY==0 and BIZ==0)
                // printf("lv3s1\n");
                 interpolate_stage_att<
                     T, FP, decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
                     false, true, false, LINEAR_BLOCK_SIZE, 3, 2, NO_COARSEN, 5, BORDER_INCLUSIVE, WORKFLOW>(
                     s_data,data_size,global_starts, xyellow, yyellow, zyellow, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[2],error);
                 //if(BIX==0 and BIY==0 and BIZ==0)
               //  printf("lv3s2\n");
                 interpolate_stage_att<
                     T, FP, decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
                     false, false, true, LINEAR_BLOCK_SIZE, 2, 5, NO_COARSEN, 5, BORDER_INCLUSIVE, WORKFLOW>(
                     s_data,data_size,global_starts, xhollow, yhollow, zhollow, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[2],error);
             }
         }
     }
    // if(BIX==0 and BIY==0 and BIZ==0)
    // printf("lv3\n");
      if(WORKFLOW==SPLINE3_AB_ATT or level == 1){
         unit = 2;
         //m*=2;
         //n*=2;
         //p*=2;
         if(WORKFLOW==SPLINE3_AB_ATT)
             calc_eb(unit);
 
         //set_orders(reverse[1]);
 
         // iteration 2, TODO switch y-z order
        
         if(intp_param.use_md[1]){
 
 
             if(intp_param.use_natural[1]==0){
 
                 interpolate_stage_md_att<
                     T, FP, decltype(xyzmap_line_16b_2u), //
                     true, false, false, LINEAR_BLOCK_SIZE,300 ,NO_COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                     s_data,data_size,global_starts, xyzmap_line_16b_2u, unit,cur_eb_r,cur_ebx2, nan_cubic_interp,error);
 
                 interpolate_stage_md_att<
                     T, FP, decltype(xyzmap_face_16b_2u), //
                     false, true, false, LINEAR_BLOCK_SIZE,240 ,NO_COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                     s_data,data_size,global_starts, xyzmap_face_16b_2u, unit,cur_eb_r,cur_ebx2, nan_cubic_interp,error);
 
                 interpolate_stage_md_att<
                     T, FP, decltype(xyzmap_cube_16b_2u), //
                     false, false, true, LINEAR_BLOCK_SIZE,64 ,NO_COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                     s_data,data_size,global_starts, xyzmap_cube_16b_2u, unit,cur_eb_r,cur_ebx2, nan_cubic_interp,error);
 
             }
             else{
                 interpolate_stage_md_att<
                     T, FP, decltype(xyzmap_line_16b_2u), //
                     true, false, false, LINEAR_BLOCK_SIZE,300 ,NO_COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                     s_data,data_size,global_starts, xyzmap_line_16b_2u, unit,cur_eb_r,cur_ebx2, nat_cubic_interp,error);
 
                 interpolate_stage_md_att<
                     T, FP, decltype(xyzmap_face_16b_2u), //
                     false, true, false, LINEAR_BLOCK_SIZE,240 ,NO_COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                     s_data,data_size,global_starts, xyzmap_face_16b_2u, unit,cur_eb_r,cur_ebx2, nat_cubic_interp,error);
 
                 interpolate_stage_md_att<
                     T, FP, decltype(xyzmap_cube_16b_2u), //
                     false, false, true, LINEAR_BLOCK_SIZE,64 ,NO_COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                     s_data,data_size,global_starts, xyzmap_cube_16b_2u, unit,cur_eb_r,cur_ebx2, nat_cubic_interp,error);
                 
 
             }
         }
         else{
              if(intp_param.reverse[1]){
                 interpolate_stage_att<
                     T, FP, decltype(xhollow_reverse), decltype(yhollow_reverse), decltype(zhollow_reverse),  //
                     false, false, true, LINEAR_BLOCK_SIZE, 4, 5, NO_COARSEN, 5, BORDER_INCLUSIVE, WORKFLOW>(
                     s_data,data_size,global_starts, xhollow_reverse, yhollow_reverse, zhollow_reverse, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[1],error);
                 interpolate_stage_att<
                     T, FP, decltype(xyellow_reverse), decltype(yyellow_reverse), decltype(zyellow_reverse),  //
                     false, true, false, LINEAR_BLOCK_SIZE, 9, 4, NO_COARSEN, 5, BORDER_INCLUSIVE, WORKFLOW>(
                     s_data,data_size,global_starts, xyellow_reverse, yyellow_reverse, zyellow_reverse, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[1],error);
                 interpolate_stage_att<
                     T, FP, decltype(xblue_reverse), decltype(yblue_reverse), decltype(zblue_reverse),  //
                     true, false, false, LINEAR_BLOCK_SIZE, 9, 9, NO_COARSEN, 4, BORDER_INCLUSIVE, WORKFLOW>(
                     s_data,data_size,global_starts, xblue_reverse, yblue_reverse, zblue_reverse, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[1],error);
             }
             else{
                 interpolate_stage_att<
                     T, FP, decltype(xblue), decltype(yblue), decltype(zblue),  //
                     true, false, false, LINEAR_BLOCK_SIZE, 5, 5, NO_COARSEN, 4, BORDER_INCLUSIVE, WORKFLOW>(
                     s_data,data_size,global_starts, xblue, yblue, zblue, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[1],error);
                 interpolate_stage_att<
                     T, FP, decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
                     false, true, false, LINEAR_BLOCK_SIZE, 5, 4, NO_COARSEN, 9, BORDER_INCLUSIVE, WORKFLOW>(
                     s_data,data_size,global_starts, xyellow, yyellow, zyellow, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[1],error);
                 interpolate_stage_att<
                     T, FP, decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
                     false, false, true, LINEAR_BLOCK_SIZE, 4, 9, NO_COARSEN, 9, BORDER_INCLUSIVE, WORKFLOW>(
                     s_data,data_size,global_starts, xhollow, yhollow, zhollow, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[1],error);
 
             }
 
         }
     }
 
 
     //if(TIX==0 and TIY==0 and TIZ==0 and BIX==0 and BIY==0 and BIZ==0)
     //printf("lv2\n");
     if(WORKFLOW==SPLINE3_AB_ATT or level ==0){
         unit = 1;
         //m*=2;
         //n*=2;
         //p*=2;
        if(WORKFLOW==SPLINE3_AB_ATT)
             calc_eb(unit);
        // set_orders(reverse[0]);
         if(intp_param.use_md[0]){
             if(intp_param.use_natural[0]==0){
 
                 interpolate_stage_md_att<
                     T, FP, decltype(xyzmap_line_16b_1u), //
                     true, false, false, LINEAR_BLOCK_SIZE,1944 ,COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                     s_data,data_size,global_starts, xyzmap_line_16b_1u, unit,cur_eb_r,cur_ebx2, nan_cubic_interp,error);
 
                 interpolate_stage_md_att<
                     T, FP, decltype(xyzmap_face_16b_1u), //
                     false, true, false, LINEAR_BLOCK_SIZE,1728 ,COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                     s_data,data_size,global_starts, xyzmap_face_16b_1u, unit,cur_eb_r,cur_ebx2, nan_cubic_interp,error);
 
                 interpolate_stage_md_att<
                     T, FP, decltype(xyzmap_cube_16b_1u), //
                     false, false, true, LINEAR_BLOCK_SIZE,512 ,COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                     s_data,data_size,global_starts, xyzmap_cube_16b_1u, unit,cur_eb_r,cur_ebx2, nan_cubic_interp,error);
 
             }
             else{
                 interpolate_stage_md_att<
                     T, FP, decltype(xyzmap_line_16b_1u), //
                     true, false, false, LINEAR_BLOCK_SIZE,1944 ,COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                     s_data,data_size,global_starts, xyzmap_line_16b_1u, unit,cur_eb_r,cur_ebx2, nat_cubic_interp,error);
 
                 interpolate_stage_md_att<
                     T, FP, decltype(xyzmap_face_16b_1u), //
                     false, true, false, LINEAR_BLOCK_SIZE,1728 ,COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                     s_data,data_size,global_starts, xyzmap_face_16b_1u, unit,cur_eb_r,cur_ebx2, nat_cubic_interp,error);
 
                 interpolate_stage_md_att<
                     T, FP, decltype(xyzmap_cube_16b_1u), //
                     false, false, true, LINEAR_BLOCK_SIZE,512 ,COARSEN, BORDER_INCLUSIVE, WORKFLOW>(
                     s_data,data_size,global_starts, xyzmap_cube_16b_1u, unit,cur_eb_r,cur_ebx2, nat_cubic_interp,error);
                 
 
             }
         }
         else{
             if(intp_param.reverse[0]){
                 //may have bug 
                 interpolate_stage_att<
                     T, FP, decltype(xhollow_reverse), decltype(yhollow_reverse), decltype(zhollow_reverse),  //
                     false, false, true, LINEAR_BLOCK_SIZE, 8, 9, COARSEN, 9, BORDER_INCLUSIVE, WORKFLOW>(
                     s_data,data_size,global_starts, xhollow_reverse, yhollow_reverse, zhollow_reverse, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[0],error);
                 interpolate_stage_att<
                     T, FP, decltype(xyellow_reverse), decltype(yyellow_reverse), decltype(zyellow_reverse),  //
                     false, true, false, LINEAR_BLOCK_SIZE, 17, 8, COARSEN, 9, BORDER_INCLUSIVE, WORKFLOW>(
                     s_data,data_size,global_starts, xyellow_reverse, yyellow_reverse, zyellow_reverse, unit,cur_eb_r,cur_ebx2,intp_param.use_natural[0],error);
                 interpolate_stage_att<
                     T, FP, decltype(xblue_reverse), decltype(yblue_reverse), decltype(zblue_reverse),  //
                     true, false, false, LINEAR_BLOCK_SIZE, 17, 17, COARSEN, 8, BORDER_INCLUSIVE, WORKFLOW>(
                     s_data,data_size,global_starts, xblue_reverse, yblue_reverse, zblue_reverse, unit,cur_eb_r,cur_ebx2,intp_param.use_natural[0],error);
 
                 //may have bug end
             }
             else{
                 interpolate_stage_att<
                     T, FP, decltype(xblue), decltype(yblue), decltype(zblue),  //
                     true, false, false, LINEAR_BLOCK_SIZE, 9, 9, COARSEN, 8, BORDER_INCLUSIVE, WORKFLOW>(
                     s_data,data_size,global_starts, xblue, yblue, zblue, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[0],error);
                 interpolate_stage_att<
                     T, FP, decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
                     false, true, false, LINEAR_BLOCK_SIZE, 9, 8, COARSEN, 17, BORDER_INCLUSIVE, WORKFLOW>(
                     s_data,data_size,global_starts, xyellow, yyellow, zyellow, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[0],error);
                
                 interpolate_stage_att<
                     T, FP, decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
                     false, false, true, LINEAR_BLOCK_SIZE, 8, 17, COARSEN, 17, BORDER_INCLUSIVE, WORKFLOW>(
                     s_data,data_size,global_starts, xhollow, yhollow, zhollow, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[0],error);
 
             }
         }
     }
     
     // iteration 3
     /*
     if(intp_param.reverse[0]){
         //may have bug 
         interpolate_stage<
             T1, T2, FP, decltype(xhollow_reverse), decltype(yhollow_reverse), decltype(zhollow_reverse),  //
             false, false, true, LINEAR_BLOCK_SIZE, 8, 9, COARSEN, 9, BORDER_INCLUSIVE, WORKFLOW>(
             s_data, s_ectrl,data_size, xhollow_reverse, yhollow_reverse, zhollow_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.interpolators[0]);
         interpolate_stage<
             T1, T2, FP, decltype(xyellow_reverse), decltype(yyellow_reverse), decltype(zyellow_reverse),  //
             false, true, false, LINEAR_BLOCK_SIZE, 17, 8, COARSEN, 9, BORDER_INCLUSIVE, WORKFLOW>(
             s_data, s_ectrl,data_size, xyellow_reverse, yyellow_reverse, zyellow_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.interpolators[1]);
         interpolate_stage<
             T1, T2, FP, decltype(xblue_reverse), decltype(yblue_reverse), decltype(zblue_reverse),  //
             true, false, false, LINEAR_BLOCK_SIZE, 17, 17, COARSEN, 8, BORDER_EXCLUSIVE, WORKFLOW>(
             s_data, s_ectrl,data_size, xblue_reverse, yblue_reverse, zblue_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.interpolators[2]);
 
         //may have bug end
     }
     else{
         interpolate_stage<
             T1, T2, FP, decltype(xblue), decltype(yblue), decltype(zblue),  //
             true, false, false, LINEAR_BLOCK_SIZE, 9, 9, COARSEN, 8, BORDER_INCLUSIVE, WORKFLOW>(
             s_data, s_ectrl,data_size, xblue, yblue, zblue, unit, cur_eb_r, cur_ebx2, radius, intp_param.interpolators[0]);
         interpolate_stage<
             T1, T2, FP, decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
             false, true, false, LINEAR_BLOCK_SIZE, 9, 8, COARSEN, 17, BORDER_INCLUSIVE, WORKFLOW>(
             s_data, s_ectrl,data_size, xyellow, yyellow, zyellow, unit, cur_eb_r, cur_ebx2, radius, intp_param.interpolators[1]);
        
         interpolate_stage<
             T1, T2, FP, decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
             false, false, true, LINEAR_BLOCK_SIZE, 8, 17, COARSEN, 17, BORDER_EXCLUSIVE, WORKFLOW>(
             s_data, s_ectrl,data_size, xhollow, yhollow, zhollow, unit, cur_eb_r, cur_ebx2, radius, intp_param.interpolators[2]);
 
     }*/
   //  if(TIX==0 and TIY==0 and TIZ==0 and BIX==0 and BIY==0 and BIZ==0)
    // printf("lv1\n");
     
 
 
      /******************************************************************************
      test only: last step inclusive
      ******************************************************************************/
     // interpolate_stage<
     //     T1, T2, FP, decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
     //     false, false, true, LINEAR_BLOCK_SIZE, 33, 4, COARSEN, 9, BORDER_INCLUSIVE, WORKFLOW>(
     //     s_data, s_ectrl, xhollow, yhollow, zhollow, unit, eb_r, ebx2, radius);
     /******************************************************************************
      production
      ******************************************************************************/
 
     /******************************************************************************
      test only: print a block
      ******************************************************************************/
     // if (TIX == 0 and BIX == 7 and BIY == 47 and BIZ == 15) { spline3d_print_block_from_GPU(s_ectrl); }
    //  if (TIX == 0 and BIX == 4 and BIY == 20 and BIZ == 20) { spline3d_print_block_from_GPU(s_data); }
 }
 
 
 template <typename TITER, typename FP, int LINEAR_BLOCK_SIZE>
 __global__ void cusz::pa_spline3d_infprecis_16x16x16data(
     TITER   data,
     DIM3    data_size,
     STRIDE3 data_leap,
     DIM3 sample_starts,
     DIM3 sample_block_grid_sizes,
     DIM3 sample_strides,
     FP eb_r,
     FP eb_x2,
     INTERPOLATION_PARAMS intp_param,
     TITER errors,
     bool workflow
     )
 {
     // compile time variables
     using T = typename std::remove_pointer<TITER>::type;
 
     {
         __shared__ struct {
             T data[17][17][17];
             //DIM3 grid_sizes[5];
            // DIM3 global_starts;
             //uint8_t level;
             //bool use_natural;
             //bool use_md;
             //bool reverse;
             T err[6];
         } shmem;
 
         /*
        if(TIX==0 and BIX==0 and BIY==0 and BIZ==0){
         printf("Errors: %.6f %.6f \n",errors[0],errors[1]);
         printf("Cubic: %d %d %d\n",intp_param.interpolators[0],intp_param.interpolators[1],intp_param.interpolators[2]);
         printf("reverse: %d %d %d\n",intp_param.reverse[0],intp_param.reverse[1],intp_param.reverse[2]);
        }
        */
         //if(BIX==0 and TIX==0){
         //    errors[BIY]=0;//dangerous
         //}
         //__syncthreads();
         DIM3 global_starts;
         uint8_t level = 0;
         uint8_t unit = 1;
         pre_compute_att<T>(sample_starts, sample_block_grid_sizes, sample_strides,global_starts,intp_param,level,unit,shmem.err,workflow);
          //if(BIX==10 and BIY == 10 and TIX==0){
          //   printf("%.4e\n",shmem.err);
         //}
         
         global2shmem_17x17x17data_att<T, T,LINEAR_BLOCK_SIZE>(data, data_size, data_leap, shmem.data,global_starts,unit);
          //if(BIX==30 and BIY >=12 and BIY < 15 and TIX==0){
          //   printf("%d %.4e %d %d %d %d \n",BIY, shmem.err,shmem.level,shmem.use_natural,shmem.use_md,shmem.reverse);
         //}
         //if(TIX==0 and BIX==0 and BIY==0)
          //   printf("gs\n");
 
         //if(TIX==0 and BIX == 10){
         //    printf("%d %d %d %d %d %d\n",GDX,BIY,level,intp_param.use_natural[level],intp_param.use_md[level],intp_param.reverse[level]);
         //}
         if(workflow){
 
             if(level==2){
                 uint8_t level3 = 3;
                 intp_param.use_natural[3]=false;
                 intp_param.use_natural[2]=false;
                 
 
 
                 intp_param.use_md[3]=false;
                 intp_param.reverse[3]=false;
                 cusz::device_api::spline3d_layout2_interpolate_att<T, FP,LINEAR_BLOCK_SIZE,SPLINE3_PRED_ATT>(shmem.data, data_size,global_starts,eb_r,eb_x2,level3,intp_param,shmem.err);
                 intp_param.reverse[3]=true;
                 cusz::device_api::spline3d_layout2_interpolate_att<T, FP,LINEAR_BLOCK_SIZE,SPLINE3_PRED_ATT>(shmem.data, data_size,global_starts,eb_r,eb_x2,level3,intp_param,shmem.err+1);
                 intp_param.use_md[3]=true;
                 cusz::device_api::spline3d_layout2_interpolate_att<T, FP,LINEAR_BLOCK_SIZE,SPLINE3_PRED_ATT>(shmem.data, data_size,global_starts,eb_r,eb_x2,level3,intp_param,shmem.err+2);
 
 
                 intp_param.use_md[2]=false;
                 intp_param.reverse[2]=false;
                 cusz::device_api::spline3d_layout2_interpolate_att<T, FP,LINEAR_BLOCK_SIZE,SPLINE3_PRED_ATT>(shmem.data, data_size,global_starts,eb_r,eb_x2,level,intp_param,shmem.err+3);
                 intp_param.reverse[2]=true;
                 cusz::device_api::spline3d_layout2_interpolate_att<T, FP,LINEAR_BLOCK_SIZE,SPLINE3_PRED_ATT>(shmem.data, data_size,global_starts,eb_r,eb_x2,level,intp_param,shmem.err+4);
                 intp_param.use_md[2]=true;
                 cusz::device_api::spline3d_layout2_interpolate_att<T, FP,LINEAR_BLOCK_SIZE,SPLINE3_PRED_ATT>(shmem.data, data_size,global_starts,eb_r,eb_x2,level,intp_param,shmem.err+5);
 
 
                 
 
                 if(TIX<6){
                     atomicAdd(const_cast<T*>(errors+TIX),shmem.err[TIX]);
                 }
             }
             else if (level==1){
                 intp_param.use_md[1]=false;
                 intp_param.reverse[1]=false;
                 cusz::device_api::spline3d_layout2_interpolate_att<T, FP,LINEAR_BLOCK_SIZE,SPLINE3_PRED_ATT>(shmem.data, data_size,global_starts,eb_r,eb_x2,level,intp_param,shmem.err);
                 intp_param.reverse[1]=true;
                 cusz::device_api::spline3d_layout2_interpolate_att<T, FP,LINEAR_BLOCK_SIZE,SPLINE3_PRED_ATT>(shmem.data, data_size,global_starts,eb_r,eb_x2,level,intp_param,shmem.err+1);
                 intp_param.use_md[1]=true;
                 cusz::device_api::spline3d_layout2_interpolate_att<T, FP,LINEAR_BLOCK_SIZE,SPLINE3_PRED_ATT>(shmem.data, data_size,global_starts,eb_r,eb_x2,level,intp_param,shmem.err+2);
 
                 if(TIX<3){
                    atomicAdd(const_cast<T*>(errors+3+BIY*3+TIX),shmem.err[TIX]);
                 }
 
             }
             else{
                 cusz::device_api::spline3d_layout2_interpolate_att<T, FP,LINEAR_BLOCK_SIZE,SPLINE3_PRED_ATT>(shmem.data, data_size,global_starts,eb_r,eb_x2,level,intp_param,shmem.err);
                 if(TIX==0){
                     atomicAdd(const_cast<T*>(errors+9+BIY),shmem.err[0]);
                 }
             }
 
             //if(TIX==0 and BIX == 10)
              //   printf("%d %.4e %.4e %.4e %.4e %.4e %.4e\n",BIY,shmem.err[0],shmem.err[1],shmem.err[2],shmem.err[3],shmem.err[4],shmem.err[5]);
             
         }
         else{
             cusz::device_api::spline3d_layout2_interpolate_att<T, FP,LINEAR_BLOCK_SIZE,SPLINE3_AB_ATT>(shmem.data, data_size,global_starts,eb_r,eb_x2,level,intp_param,shmem.err);
             if(TIX==0)
                 atomicAdd(const_cast<T*>(errors+BIY),shmem.err[0]);
         
         }
         
         //Just a copy back here
         //__syncthreads();
         //if(BIX==10 and BIY == 10 and TIX==0){
         //    printf("%.4e\n",shmem.err);
         //}
 
         //if(TIX==0 and BIX ==30 and BIY >=0 and BIY < 3){
         //    printf("%d %.4e\n",BIY,shmem.err);
         //}
 
 
         //if(TIX<6){
         //    atomicAdd(const_cast<T*>(errors+BIY),shmem.err[TIX]);//BIY 0-17
         //}
 
         
         //if(TIX==0 and BIX==0 and BIY==0 and BIZ==0)
         //    printf("interp\n");
         //if(TIX==0 and BIX==0 and BIY==0 and BIZ==0)
          //   printf("esz: %d %d %d\n",ectrl_size.x,ectrl_size.y,ectrl_size.z);
 
 
         //if(TIX==0 and BIX==0 and BIY==0 and BIZ==0)
 
         ///shmem2global_17x17x17data_with_compaction<T, E, LINEAR_BLOCK_SIZE>(shmem.ectrl, ectrl, ectrl_size, ectrl_leap, radius, shmem.grid_leaps,shmem.prefix_nums, compact_val, compact_idx, compact_num);
 
         // shmem2global_32x8x8data<T, E, LINEAR_BLOCK_SIZE>(shmem.ectrl, ectrl, ectrl_size, ectrl_leap);
 
         //if(TIX==0 and BIX==0 and BIY==0 and BIZ==0)
         //    printf("s2g\n");
     }
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