<!-- <h3 align="center"><img src="https://user-images.githubusercontent.com/10354752/81179956-05860600-8f70-11ea-8b01-856f29b9e8b2.jpg" width="150"></h3> -->

<h3 align="center">
SC' 24 Research Paper Snapshot
<br>
cuSZ-<i>i</i>: High-Ratio Scientific Lossy Compression on
GPUs with Optimized Multi-Level Interpolation
</h3>

<p align="center">
<a href="./LICENSE"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg"></a>
</p>

Th cuSZ-*i* work is collaboration of Jinyang Liu, Jiannan Tian, and Shixun Wu.
(The final paper will come to SC '24 proceedings.) 
This work covers
- spline-interpolation-based high-ratio data compression and high-quality data reconstruction
- compresion ratio boost from incorporating the synergetic lossless encoding

This repository is a fork of pSZ/cuSZ for development and is a part of the research paper artifacts.

- To reproduce the results of our [SC '24 paper](https://arxiv.org/abs/2312.05492), please refer to the repository of [artifacts](https://github.com/jtian0/24_SC_artifacts).
- Please also refer to pSZ/cuSZ's [main repository](https://github.com/szcompressor/cuSZ) for more information.

<details>
<summary>
Here is a chart to compare cuSZ-<i>i</i> with the basic framework (cuSZ) and its other variants. (Click to expand.)
</summary>

cuSZ and its variants use variable techniques to balance the need for data-reconstruction quality, compression ratio, and data-processing speed. A quick comparison is given below.

Notably, cuSZ (Tian et al., '20, '21) as the basic framework provides a balanced compression ratio and quality, while FZ-GPU (Zhang, Tian et al., '23) and SZp-CUDA/GSZ (Huang et al., '23, '24) prioritize data processing speed. cuSZ+ (hi-ratio) is an outcome of data compressibility research to demonstrate that certain methods (e.g., RLE) can work better in highly compressible cases (Tian et al., '21). The latest art, cuSZ-i (Liu, Tian, Wu et al., '24), attempts to utilize the QoZ-like methods (Liu et al., '22) to significantly enhance the data-reconstruction quality and the compression ratio.

```
                    prediction &                 statistics          lossless encoding          lossless encoding
                    quantization                                     passs (1)                  pass (2)

                  +----------------------+      +-----------+      +------------------+       +-----------------+
CPU-SZ     -----> | predictor {ℓ, lr, S} | ---> | histogram | ---> | ui2 Huffman enc. | ----> | DEFLATE (LZ+HF) |
'16, '17-ℓ, '18-lr, '21-S, '22-QoZ ------+      +-----------+      +------------------+       +-----------------+
(Di and Franck, Tao et al., Liang et al. Zhao et al., Liu et al.)

                  +----------------------+      +-----------+      +------------------+
cuSZ       -----> | predictor ℓ-(1,2,3)D | ---> | histogram | ---> | ui2 Huffman enc. | ----> ( n/a )
'20, '21          +----------------------+      +-----------+      +------------------+
(Tian et al.)
                  +----------------------+      +-----------+      +-------------------+      +---------+
cuSZ+        ---> | predictor ℓ-(1,2,3)D | ---> | histogram | ---> | de-redundancy RLE | ---> | HF enc. |
hi-ratio '21      +----------------------+      +-----------+      +-------------------+      +---------+
(Tian et al.)
                  +----------------------+                         +---------------+
FZ-GPU '23   ---> | predictor ℓ-(1,2,3)D | ---> ( n/a ) ---------> | de-redundancy | -------> ( n/a )
(Zhang, Tian et al.) --------------------+                         +---------------+

                  [ single kernel ]------------------------------------------------+           
SZp-CUDA -------> | predictor ℓ-1D   ---------> ( n/a ) --------->   de-redundancy | -------> ( n/a )
'23, '24          +----------------------------------------------------------------+           
(Huang et al., Huang et al.)

                  +----------------+            +-----------+      +------------------+       +---------------+
cuSZ-i '24   ---> | predictor S-3D | ---------> | histogram | ---> | ui2 Huffman enc. | ----> | de-redundancy |
(Liu, Tian, Wu et al.) ------------+            +-----------+      +------------------+       +---------------+

ℓ: Lorenzo predictor; lr: linear-regression predictor; S: spline-interpolative predictor
```

</details>

<br>

If you mention cuSZ-*i* that priorizes `data quality` and `compression ratio` in your paper, please kindly cite using `\cite{liu_tian_wu2024cuszi}` and the BibTeX entries below (or standalone [`.bib` file](doc/cite-cuszi.bib)).


In addition, for the full context, the basic framework is covered in (PACT '20: cuSZ) ([arXiv](https://arxiv.org/abs/2007.09625)) and (CLUSTER '21: cuSZ+) ([arXiv](https://arxiv.org/abs/2105.12912)) covers
- basic framework: (fine-grained) *N*-D prediction-based error-controling "construction" + (coarse-grained) lossless encoding
- optimization in throughput, featuring fine-grained *N*-D "reconstruction"
- optimization in compression ratio, when data is deemed as "smooth"

If you found the whole pipeline is important, please kindly cite using `\cite{tian2020cusz,tian2021cuszplus,liu_tian_wu2024cuszi}` (for three papers) and the BibTeX entries below (or standalone [`.bib` file](doc/cite-cuszi.bib)).

<br>

```bibtex
@inproceedings{liu_tian_wu2024cuszi,
      title = {{{\scshape cuSZ}-{\itshape i}: High-Ratio scientific lossy compression on
             GPUs with optimized multi-level interpolation}},
     author = {Liu, Jinyang and Tian, Jiannan and Wu, Shixun and Di, Sheng and Zhang, Boyuan and Underwood, Robert and Huang, Yafan and Huang, Jiajun and Zhao, Kai and Li, Guanpeng and Tao, Dingwen and Chen, Zizhong and Cappello, Franck},
       year = {2024}, month = {11}, isbn = {979-8-3503-5291-7},
       note = {Co-first authors: Jinyang Liu, Jiannan Tian, and Shixun Wu},
        url = {https://doi.ieeecomputersociety.org/10.1109/SC41406.2024.00019}, 
  booktitle = {SC '24: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
     series = {SC '24}, address = {Atlanta, GA, USA}}

@inproceedings{tian2020cusz,
      title = {{{\textsc cuSZ}: An efficient GPU-based error-bounded lossy compression framework for scientific data}},
     author = {Tian, Jiannan and Di, Sheng and Zhao, Kai and Rivera, Cody and Fulp, Megan Hickman and Underwood, Robert and Jin, Sian and Liang, Xin and Calhoun, Jon and Tao, Dingwen and Cappello, Franck},
       year = {2020}, month = {10},
        doi = {10.1145/3410463.3414624}, isbn = {9781450380751},
  booktitle = {Proceedings of the ACM International Conference on Parallel Architectures and Compilation Techniques},
     series = {PACT '20}, address = {Atlanta (virtual event), GA, USA}}

@inproceedings{tian2021cuszplus,
      title = {Optimizing error-bounded lossy compression for scientific data on GPUs},
     author = {Tian, Jiannan and Di, Sheng and Yu, Xiaodong and Rivera, Cody and Zhao, Kai and Jin, Sian and Feng, Yunhe and Liang, Xin and Tao, Dingwen and Cappello, Franck},
       year = {2021}, month = {09},
        doi = {10.1109/Cluster48925.2021.00047},
  booktitle = {2021 IEEE International Conference on Cluster Computing (CLUSTER)},
     series = {CLUSTER '21}, address = {Portland (virtual event), OR, USA}}
```

<h3 align="center">
acknowledgements
</h3>

This research was supported by the U.S. Department of Energy, Office of Science, Advanced Scientific Computing Research (ASCR), under contracts DE-AC02-06CH11357. This work was also supported by the National Science Foundation (Grant Nos. 2003709, 2303064, 2104023, 2247080, 2247060, 2312673, 2311875, and 2311876). We also acknowledge the computing resources provided by Argonne Leadership Computing Facility (ALCF) and Advanced Cyberinfrastructure Coordination Ecosystem—Purdue Anvil through Services & Support (ACCESS).

![acknowledgement](https://user-images.githubusercontent.com/10354752/196348936-f0909251-1c2f-4c53-b599-08642dcc2089.png)
