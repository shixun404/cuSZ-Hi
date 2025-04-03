/**
 * @file spline3.cu
 * @author Jinyang Liu, Shixun Wu, Jiannan Tian
 * @brief A high-level Spline3D wrapper. Allocations are explicitly out of
 * called functions.
 * @version 0.3
 * @date 2021-06-15
 *
 * (copyright to be updated)
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

 #include "busyheader.hh"
 #include "cusz/type.h"
 #include "detail/spline3.inl"
 #include "kernel/spline.hh"
 #include "mem/compact.hh"
 
 #include <cuda_runtime.h>
 //#include "mem/memseg_cxx.hh"
 //#include "mem/memseg.h"
 //#include "mem/layout.h"
 //#include "mem/layout_cxx.hh"
 #define BLOCK_DIM_SIZE 384
 constexpr int DEFAULT_BLOCK_SIZE = BLOCK_DIM_SIZE;
 
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
     void* _outlier, double eb, double rel_eb, uint32_t radius, INTERPOLATION_PARAMS &intp_param, float* time, void* stream, pszmem_cxx<T>* profiling_errors)
 {
   constexpr auto BLOCK = 16;
   auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };
 
   auto ebx2 = eb * 2;
   auto eb_r = 1 / eb;
 
   auto l3 = data->template len3<dim3>();
   auto grid_dim =
       dim3(div(l3.x, BLOCK ), div(l3.y, BLOCK ), div(l3.z, BLOCK ));
 
 
   auto auto_tuning_grid_dim =
       dim3(1, 1, 1);
 
 
 
   using Compact = typename CompactDram<PROPER_GPU_BACKEND, T>::Compact;
   auto ot = (Compact*)_outlier;
 
   //CREATE_GPUEVENT_PAIR;
   //START_GPUEVENT_RECORDING(stream);
   float att_time=0;
  if(intp_param.auto_tuning>0){
    //std::cout<<"att "<<(int)intp_param.auto_tuning<<std::endl;
    double a1=2.0;
    double a2=1.75;
    double a3=1.5;
    double a4=1.25;
    double a5=1;
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
     if(intp_param.auto_tuning==1){
 
       CREATE_GPUEVENT_PAIR;
        START_GPUEVENT_RECORDING(stream);
    
       cusz::c_spline3d_profiling_16x16x16data<T*, DEFAULT_BLOCK_SIZE>  //
         <<<auto_tuning_grid_dim, dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (GpuStreamT)stream>>>(
             data->dptr(), data->template len3<dim3>(),
             data->template st3<dim3>(),  //
             profiling_errors->dptr());
         STOP_GPUEVENT_RECORDING(stream);
         CHECK_GPU(GpuStreamSync(stream));
         TIME_ELAPSED_GPUEVENT(&att_time);
         DESTROY_GPUEVENT_PAIR;
       //profiling_errors->control({D2H});
       CHECK_GPU(cudaMemcpy(profiling_errors->m->h, profiling_errors->m->d, profiling_errors->m->bytes, cudaMemcpyDeviceToHost));
       auto errors=profiling_errors->hptr();
       
       //printf("host %.4f %.4f\n",errors[0],errors[1]);
       bool do_reverse=(errors[1]>3*errors[0]);
       intp_param.reverse[0]=intp_param.reverse[1]=intp_param.reverse[2]=intp_param.reverse[3]=do_reverse;
     }
     else if (intp_param.auto_tuning==2){
        CREATE_GPUEVENT_PAIR;
        START_GPUEVENT_RECORDING(stream);
       cusz::c_spline3d_profiling_data_2<T*, DEFAULT_BLOCK_SIZE>  //
         <<<auto_tuning_grid_dim, dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (GpuStreamT)stream>>>(
             data->dptr(), data->template len3<dim3>(),
             data->template st3<dim3>(),
               //
             profiling_errors->dptr());
       STOP_GPUEVENT_RECORDING(stream);
       CHECK_GPU(GpuStreamSync(stream));
       TIME_ELAPSED_GPUEVENT(&att_time);
       DESTROY_GPUEVENT_PAIR;
       //profiling_errors->control({D2H});
       CHECK_GPU(cudaMemcpy(profiling_errors->m->h, profiling_errors->m->d, profiling_errors->m->bytes, cudaMemcpyDeviceToHost));
       auto errors=profiling_errors->hptr();
 
       //intp_param.interpolators[0]=(errors[0]>errors[1]);
       //intp_param.interpolators[1]=(errors[2]>errors[3]);
       //intp_param.interpolators[2]=(errors[4]>errors[5]);
       
      
       bool do_nat = errors[0] + errors[2] + errors[4] > errors[1] + errors[3] + errors[5];
       intp_param.use_natural[0]=intp_param.use_natural[1]=intp_param.use_natural[2]=intp_param.use_natural[3]=do_nat;
       //intp_param.interpolators[0]=(errors[0]>errors[1]);
       //intp_param.interpolators[1]=(errors[2]>errors[3]);
       //intp_param.interpolators[2]=(errors[4]>errors[5]);
       //to revise: cubic spline selection for both axis-wise and global
        // bool do_reverse=(errors[1]>2*errors[0]);
         bool do_reverse=(errors[4+do_nat]>3*errors[do_nat]);
        intp_param.reverse[0]=intp_param.reverse[1]=intp_param.reverse[2]=intp_param.reverse[3]=do_reverse;
     }
     else{
       const auto S_STRIDE = 6 * BLOCK;//96
       cusz::reset_errors<<<dim3(1, 1, 1), dim3(DEFAULT_BLOCK_SIZE, 1, 1),0, (GpuStreamT)stream >>>(profiling_errors->dptr());
 
       auto calc_start_size = [&](auto dim,auto & s_start,auto &s_size) {
           auto mid = dim / 2;
     
           auto k = (mid - 8) / S_STRIDE;  
           auto t = (dim - 8 - 1 - mid) / S_STRIDE;  
 
           s_start = mid - k * S_STRIDE;
           s_size = k+t+1;
       };
 
       int s_start_x,s_start_y,s_start_z,s_size_x,s_size_y,s_size_z;
 
       calc_start_size(l3.x,s_start_x,s_size_x);
       calc_start_size(l3.y,s_start_y,s_size_y);
       calc_start_size(l3.z,s_start_z,s_size_z);
 
       //printf("%d %d %d %d %d %d\n",s_start_x,s_start_y,s_start_z,s_size_x,s_size_y,s_size_z);
       float temp_time = 0;
       CREATE_GPUEVENT_PAIR;
        START_GPUEVENT_RECORDING(stream);
       auto block_num = s_size_x*s_size_y*s_size_z;
 
       cusz::pa_spline3d_infprecis_16x16x16data<T*, float, DEFAULT_BLOCK_SIZE> //
       <<<dim3(s_size_x*s_size_y*s_size_z, 9, 1), dim3(DEFAULT_BLOCK_SIZE, 1, 1),0, (GpuStreamT)stream  >>>
       (data->dptr(), data->template len3<dim3>(),data->template st3<dim3>(),dim3(s_start_x,s_start_y,s_start_z),dim3(s_size_x,s_size_y,s_size_z),dim3(S_STRIDE,S_STRIDE,S_STRIDE),eb_r,ebx2,intp_param,profiling_errors->dptr(),true);
        STOP_GPUEVENT_RECORDING(stream);
       CHECK_GPU(GpuStreamSync(stream));
       TIME_ELAPSED_GPUEVENT(&temp_time);
       DESTROY_GPUEVENT_PAIR;
       att_time+=temp_time;
       CHECK_GPU(cudaMemcpy(profiling_errors->m->h, profiling_errors->m->d, profiling_errors->m->bytes, cudaMemcpyDeviceToHost));
       auto errors=profiling_errors->hptr();
 
       //for(int i=0;i<18;i++){
       //printf("%d %.4e\n",i,errors[i]);
      // }
 
 
       double best_ave_pre_error[4];
       auto calcnum  = [&](auto N){
         return N*(7*N*N+9*N+3);
       };
 
 
       T best_error;
       if(errors[0]>errors[1]){
         best_error = errors[1];
         intp_param.reverse[3] = true;
       }
       else{
         best_error = errors[0];
         intp_param.reverse[3] = false;
       }
        
 
       intp_param.use_md[3] = errors[2] < best_error; 
       best_error = fmin(errors[2],best_error);
       best_ave_pre_error[3]= best_error/(calcnum(1)*block_num);
 
 
       if(errors[3]>errors[4]){
         best_error = errors[4];
         intp_param.reverse[2] = true;
       }
       else{
         best_error = errors[3];
         intp_param.reverse[2] = false;
       }
 
       intp_param.use_md[2] = errors[5] < best_error; 
       best_error = fmin(errors[5],best_error);
       best_ave_pre_error[2]= best_error/(calcnum(2)*block_num);
 
       best_error = errors[6];
       auto best_idx = 6; 
       for(auto i = 6;i<12;i++){
         if(errors[i]<best_error){
           best_error=errors[i];
           best_idx = i;
         }
       }
       intp_param.use_natural[1] = best_idx >  8;
       intp_param.use_md[1] = (best_idx ==  8 or best_idx ==  11) ;
       intp_param.reverse[1] = best_idx%3;
 
       best_ave_pre_error[1]= best_error/(calcnum(4)*block_num);
 
       best_error = errors[12];
       best_idx = 12; 
 
       for(auto i = 12;i<18;i++){
         if(errors[i]<best_error){
           best_error=errors[i];
           best_idx = i;
         }
       }
       intp_param.use_natural[0] = best_idx >  14;
       intp_param.use_md[0] = (best_idx ==  14 or best_idx ==  17);
       intp_param.reverse[0] = best_idx%3;
 
       best_ave_pre_error[0]= best_error/(calcnum(8)*block_num);
       
       printf("BESTERROR: %.4e %.4e %.4e %.4e\n",best_ave_pre_error[3],best_ave_pre_error[2],best_ave_pre_error[1],best_ave_pre_error[0]);
       // intp_param.use_md[0] = 1;
       // intp_param.use_md[1] = 1;
       // intp_param.use_md[2] = 1;
       // intp_param.use_md[3] = 1;
       if(intp_param.auto_tuning==4){
          cusz::reset_errors<<<dim3(1, 1, 1), dim3(DEFAULT_BLOCK_SIZE, 1, 1),0, (GpuStreamT)stream >>>(profiling_errors->dptr());
 
         float temp_time = 0;
         CREATE_GPUEVENT_PAIR;
          START_GPUEVENT_RECORDING(stream);
 
         cusz::pa_spline3d_infprecis_16x16x16data<T*, float, DEFAULT_BLOCK_SIZE> //
         <<<dim3(s_size_x*s_size_y*s_size_z, 11, 1), dim3(DEFAULT_BLOCK_SIZE, 1, 1),0, (GpuStreamT)stream  >>>
         (data->dptr(), data->template len3<dim3>(),data->template st3<dim3>(),dim3(s_start_x,s_start_y,s_start_z),dim3(s_size_x,s_size_y,s_size_z),dim3(S_STRIDE,S_STRIDE,S_STRIDE),eb_r,ebx2,intp_param,profiling_errors->dptr(),false);
          STOP_GPUEVENT_RECORDING(stream);
         CHECK_GPU(GpuStreamSync(stream));
         TIME_ELAPSED_GPUEVENT(&temp_time);
         DESTROY_GPUEVENT_PAIR;
         att_time+=temp_time;
 
         auto errors=profiling_errors->hptr();
         for(int i=0;i<11;i++){
           printf("%d %.4e\n",i,errors[i]);
         }
 
         best_error = errors[0];
         auto best_idx = 0; 
         
         for(auto i = 1;i<11;i++){
           if(errors[i]<best_error){
             best_error=errors[i];
             best_idx = i;
           }
         }
 
         if(best_idx==0){
             intp_param.alpha = 1.0;
             intp_param.beta = 2.0;
         }
         else if (best_idx==1){
             intp_param.alpha = 1.25;
             intp_param.beta = 2.0;
         }
         else{
             intp_param.alpha = 1.5+0.25*((best_idx-2)/3);
             intp_param.beta = 2.0+((best_idx-2)%3);
         }
 
       }
       else if(intp_param.auto_tuning >=5){
         best_idx = intp_param.auto_tuning-5;
         if(best_idx==0){
             intp_param.alpha = 1.0;
             intp_param.beta = 2.0;
         }
         else if (best_idx==1){
             intp_param.alpha = 1.25;
             intp_param.beta = 2.0;
         }
         else{
             intp_param.alpha = 1.5+0.25*((best_idx-2)/3);
             intp_param.beta = 2.0+((best_idx-2)%3);
         }
 
       }
 
 
 
 
 
 
 
 
 
     }
     //for(int i=0;i<4;i++)
     //intp_param.reverse[i]=false;
      printf("NAT: %d %d %d %d\n",intp_param.use_natural[3],intp_param.use_natural[2],intp_param.use_natural[1],intp_param.use_natural[0]);
       printf("MD: %d %d %d %d\n",intp_param.use_md[3],intp_param.use_md[2],intp_param.use_md[1],intp_param.use_md[0]);
       printf("REVERSE: %d %d %d %d\n",intp_param.reverse[3],intp_param.reverse[2],intp_param.reverse[1],intp_param.reverse[0]);
       printf("A B: %.2f %.2f\n",intp_param.alpha,intp_param.beta);
     
   
   }
   CREATE_GPUEVENT_PAIR;
   START_GPUEVENT_RECORDING(stream);
 
   cusz::c_spline3d_infprecis_16x16x16data<T*, E*, float, DEFAULT_BLOCK_SIZE>  //
       <<<grid_dim, dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (GpuStreamT)stream>>>(
           data->dptr(), data->template len3<dim3>(),
           data->template st3<dim3>(),  //
           ectrl->dptr(), ectrl->template len3<dim3>(),
           ectrl->template st3<dim3>(),  //
           anchor->dptr(), anchor->template st3<dim3>(), ot->val(), ot->idx(),
           ot->num(), eb_r, ebx2, radius, intp_param);//,profiling_errors->dptr());
 
   STOP_GPUEVENT_RECORDING(stream);
   CHECK_GPU(GpuStreamSync(stream));
   TIME_ELAPSED_GPUEVENT(time);
   DESTROY_GPUEVENT_PAIR;
 
   *time+=att_time;
 
   return 0;
 }
 
 template <typename T, typename E, typename FP>
 int spline_reconstruct(
     pszmem_cxx<T>* anchor, pszmem_cxx<E>* ectrl, pszmem_cxx<T>* xdata, T* outlier_tmp,
     double eb, uint32_t radius, INTERPOLATION_PARAMS intp_param, float* time, void* stream)
 {
   constexpr auto BLOCK = 16;
 
   auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };
 
   auto ebx2 = eb * 2;
   auto eb_r = 1 / eb;
 
   auto l3 = xdata->template len3<dim3>();
   auto grid_dim =
       dim3(div(l3.x, BLOCK ), div(l3.y, BLOCK ), div(l3.z, BLOCK ));
 
   CREATE_GPUEVENT_PAIR;
   START_GPUEVENT_RECORDING(stream);
 
   cusz::x_spline3d_infprecis_16x16x16data<E*, T*, float, DEFAULT_BLOCK_SIZE>   //
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
 
 #define INIT(T, E)                                                            \
   template int spline_construct<T, E>(                                        \
       pszmem_cxx<T> * data, pszmem_cxx<T> * anchor, pszmem_cxx<E> * ectrl,    \
       void* _outlier, double eb, double rel_eb, uint32_t radius, struct INTERPOLATION_PARAMS &intp_param, float* time, void* stream, pszmem_cxx<T> * profiling_errors); \
   template int spline_reconstruct<T, E>(                                      \
       pszmem_cxx<T> * anchor, pszmem_cxx<E> * ectrl, pszmem_cxx<T> * xdata, T* outlier_tmp,  \
       double eb, uint32_t radius, struct INTERPOLATION_PARAMS intp_param, float* time, void* stream);
 
 INIT(f4, u1)
 INIT(f4, u2)
 INIT(f4, u4)
 INIT(f4, f4)
 
 //INIT(f8, u1)
 //INIT(f8, u2)
 //INIT(f8, u4)
 //INIT(f8, f4)
 
 #undef INIT
 #undef SETUP
 