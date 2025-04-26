<!-- <h3 align="center"><img src="https://user-images.githubusercontent.com/10354752/81179956-05860600-8f70-11ea-8b01-856f29b9e8b2.jpg" width="150"></h3> -->

<h3 align="center">
SC' 25 Research Paper Snapshot
<br>
cuSZ-Hi: Boosting Scientific Error-Bounded Lossy Compression through Optimized Synergistic Lossy-Lossless Orchestration
</h3>

<p align="center">
<a href="./LICENSE"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg"></a>
</p>


# Installation
Run the following command to compile and execute cuSZ-Hi

```
git clone https://github.com/shixun404/cuSZ-Hi.git
cmake -S cuSZ-Hi -B cuSZ-Hi/build \  
    -D PSZ_BACKEND=cuda \
    -D PSZ_BUILD_EXAMPLES=off \
    -D CMAKE_CUDA_ARCHITECTURES="80" \
    -D CMAKE_BUILD_TYPE=Release
cmake --build cuSZ-Hi/build -- -j
cd cuSZ-Hi/build
```
## Notes

Please use the corresponding CMAKE_CUDA_ARCHITECTURES for your hardware. For example, NVIDIA A100 should be 86, NVIDIA RTX 4090/6000 Ada should be 89, and NVIDIA H100 should be 90.

# Compression
```
./cuszhi --report time,cr  -z -t f32 -m r2r --dim3 [DimX]x[DimY]x[DimZ] -e [REL_ERROR_BOUND] --predictor spline3 -i [input.data] -s [cr/tp];
```
## Notes
*  input.data is the binary input file;
*  -t: f32 (float32 data) or d64 (double precision data, under development);
*  DimX is the fastest dimension, DimZ is the slowest;
*  -s: cr (Huffman-integrated lossless pipeline, slower but higher compression ratio) or tp (Huffman-free lossless pipeline, lower compression ratio but fast);
*  Add "-a rd-first" if you need better rate-distortion rather than a maximized compression ratio under a fixed error bound.  
# Decompression
```
./cuszhi --report time -x -i [input.data.cusza] --compare input.data
```

