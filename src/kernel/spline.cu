/**
 * @file spline.cu
 * @author Jinyang Liu, Shixun Wu, Jiannan Tian
 * @brief A high-level Spline2D wrapper. Allocations are explicitly out of
 * called functions.
 * @version 0.3
 * @date 2021-06-15
 *
 * (copyright to be updated)
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

 #include <cuda_runtime.h>

 #include "busyheader.hh"
 #include "cusz/type.h"
 #include "detail/spline.inl"
 #include "kernel/spline.hh"
 #include "mem/compact.hh"
 // #include "mem/memseg_cxx.hh"
 // #include "mem/memseg.h"
 // #include "mem/layout.h"
 // #include "mem/layout_cxx.hh"
 #define SPLINE_DIM 3
 #define AnchorBlockSizeX 16
 #define AnchorBlockSizeY 16
 #define AnchorBlockSizeZ 16
 #define numAnchorBlockX 1  // Number of Anchor blocks along X
 #define numAnchorBlockY 1  // Number of Anchor blocks along Y
 #define numAnchorBlockZ 1  // Number of Anchor blocks along Z
 
 constexpr int DEFAULT_BLOCK_SIZE = 384;
 
 #define SETUP                                                   \
   auto div3 = [](dim3 len, dim3 sublen) {                       \
     return dim3(                                                \
         (len.x - 1) / sublen.x + 1, (len.y - 1) / sublen.y + 1, \
         (len.z - 1) / sublen.z + 1);                            \
   };                                                            \
   auto ndim = [&]() {                                           \
     if (len3.z == 1 and len3.y == 1)                            \
       return 1;                                                 \
     else if (len3.z == 1 and len3.y != 1)                       \
       return 2;                                                 \
     else                                                        \
       return 3;                                                 \
   };
 
 template <typename T, typename E, typename FP>
 int spline_construct(
     pszmem_cxx<T>* data, pszmem_cxx<T>* anchor, pszmem_cxx<E>* ectrl,
     void* _outlier, double eb, double rel_eb, uint32_t radius,
     INTERPOLATION_PARAMS& intp_param, float* time, void* stream,
     pszmem_cxx<T>* profiling_errors)
 {
   auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };
 
   auto ebx2 = eb * 2;
   auto eb_r = 1 / eb;
 
   auto l3 = data->template len3<dim3>();
   auto grid_dim = dim3(
       div(l3.x, AnchorBlockSizeX * numAnchorBlockX),
       div(l3.y, AnchorBlockSizeY * numAnchorBlockY),
       div(l3.z, AnchorBlockSizeZ * numAnchorBlockZ));
 
   auto auto_tuning_grid_dim = dim3(1, 1, 1);
 
   using Compact = typename CompactDram<PROPER_GPU_BACKEND, T>::Compact;
   auto ot = (Compact*)_outlier;
 
   CREATE_GPUEVENT_PAIR;
   START_GPUEVENT_RECORDING(stream);
   // printf("l3 %d %d %d\n", l3.x, l3.y, l3.z);
   // printf("grid_dim %d, %d, %d\n", grid_dim.x, grid_dim.y, grid_dim.z);
   // assert(0);
   if(intp_param.auto_tuning>0){
    //std::cout<<"att "<<(int)intp_param.auto_tuning<<std::endl;
    double a1=2.0 + 1;
    double a2=1.75 + 1;
    double a3=1.5 + 1;
    double a4=1.25 + 1;
    double a5=1 + 1;
    double e1=1e-1;
    double e2=1e-2;
    double e3=1e-3;
    double e4=1e-4;
    double e5=1e-5;
   
    intp_param.beta=4.0;
    if(rel_eb>=e1)
     intp_param.alpha=a1;
    else if(rel_eb>=e2)
     intp_param.alpha=a2+(a1-a2)*(rel_eb-e2)/(e1-e2);
    else if(rel_eb>=e3)
     intp_param.alpha=a3+(a2-a3)*(rel_eb-e3)/(e2-e3);
    else if(rel_eb>=e4)
     intp_param.alpha=a4+(a3-a4)*(rel_eb-e4)/(e3-e4);
    else if(rel_eb>=e5)
     intp_param.alpha=a5+(a4-a5)*(rel_eb-e5)/(e4-e5);
    else
     intp_param.alpha=a5;
   }
 
   intp_param.reverse[0]=intp_param.reverse[1]=intp_param.reverse[2] = true;
   intp_param.interpolators[0]=intp_param.interpolators[1]=intp_param.interpolators[2] = true;
   cusz::c_spline_infprecis_data<
       T*, E*, float, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
       numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, DEFAULT_BLOCK_SIZE>
       <<<grid_dim, dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (GpuStreamT)stream>>>(
           data->dptr(), data->template len3<dim3>(),
           data->template st3<dim3>(),  //
           ectrl->dptr(), ectrl->template len3<dim3>(),
           ectrl->template st3<dim3>(),  //
           anchor->dptr(), anchor->template st3<dim3>(), ot->val(), ot->idx(),
           ot->num(), eb_r, ebx2, radius,
           intp_param);  //,profiling_errors->dptr());
 
   STOP_GPUEVENT_RECORDING(stream);
   CHECK_GPU(GpuStreamSync(stream));
   TIME_ELAPSED_GPUEVENT(time);
   DESTROY_GPUEVENT_PAIR;
 
   return 0;
 }
 
 template <typename T, typename E, typename FP>
 int spline_reconstruct(
     pszmem_cxx<T>* anchor, pszmem_cxx<E>* ectrl, pszmem_cxx<T>* xdata, T* outlier_tmp,
     double eb, uint32_t radius, INTERPOLATION_PARAMS intp_param, float* time,
     void* stream)
 {
 
   auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };
 
   auto ebx2 = eb * 2;
   auto eb_r = 1 / eb;
 
   auto l3 = xdata->template len3<dim3>();
   auto grid_dim = dim3(
       div(l3.x, AnchorBlockSizeX * numAnchorBlockX),
       div(l3.y, AnchorBlockSizeY * numAnchorBlockY),
       div(l3.z, AnchorBlockSizeZ * numAnchorBlockZ));
 
   CREATE_GPUEVENT_PAIR;
   START_GPUEVENT_RECORDING(stream);
   printf("%d %d %d\n", xdata->template len3<dim3>().x, xdata->template len3<dim3>().y, xdata->template len3<dim3>().z);
   cusz::x_spline_infprecis_data<
       E*, T*, float, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
       numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ,
       DEFAULT_BLOCK_SIZE>                                                    //
       <<<grid_dim, dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (GpuStreamT)stream>>>  //
       (ectrl->dptr(), ectrl->template len3<dim3>(),
        ectrl->template st3<dim3>(),  //
        anchor->dptr(), anchor->template len3<dim3>(),
        anchor->template st3<dim3>(),  //
        xdata->dptr(), xdata->template len3<dim3>(),
        xdata->template st3<dim3>(),  //
        outlier_tmp,
        eb_r, ebx2, radius, intp_param);
 
   STOP_GPUEVENT_RECORDING(stream);
   CHECK_GPU(GpuStreamSync(stream));
   TIME_ELAPSED_GPUEVENT(time);
   DESTROY_GPUEVENT_PAIR;
 
   return 0;
 }
 
 #define INIT(T, E)                                                          \
   template int spline_construct<T, E>(                                      \
       pszmem_cxx<T> * data, pszmem_cxx<T> * anchor, pszmem_cxx<E> * ectrl,  \
       void* _outlier, double eb, double rel_eb, uint32_t radius,            \
       struct INTERPOLATION_PARAMS& intp_param, float* time, void* stream,   \
       pszmem_cxx<T>* profiling_errors);                                     \
   template int spline_reconstruct<T, E>(                                    \
       pszmem_cxx<T> * anchor, pszmem_cxx<E> * ectrl, pszmem_cxx<T> * xdata, T* outlier_tmp, \
       double eb, uint32_t radius, struct INTERPOLATION_PARAMS intp_param,   \
       float* time, void* stream);
 
 INIT(f4, u1)
 INIT(f4, u2)
 INIT(f4, u4)
 INIT(f4, f4)
 
 INIT(f8, u1)
 INIT(f8, u2)
 INIT(f8, u4)
 INIT(f8, f4)
 
 #undef INIT
 #undef SETUP
 