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
    -D CMAKE_CUDA_ARCHITECTURES="80;89" \
    -D CMAKE_BUILD_TYPE=Release
cmake --build cuSZ-Hi/build -- -j
cd cuSZ-Hi/build
```
# Compression
```
./cuszi --report time,cr  -z -t f32 -m r2r --dim3 512x512x512 -e 1e-3 --predictor spline3 -a rd-first -i $JHTDB/0500_pressure.f32;
```
# Decompression
```
./cuszi --report time -x -i $JHTDB/0500_pressure.f32.cusza --compare $JHTDB/0500_pressure.f32
```

