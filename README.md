# notes
## IREE+MLIR+LLVM

## TODO Reading List:
* FPL: Fast Presburger Arithmetic through Transprecision
  - https://dl.acm.org/doi/pdf/10.1145/3485539
* Automatic Horizontal Fusion for GPU Kernels
  - https://arxiv.org/pdf/2007.01277.pdf
* MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning
  - https://arxiv.org/abs/2110.15352
  - IREE: https://discord.com/channels/689900678990135345/760577505840463893/933243735356084245
    * From Ben Vanik
* (ongoing) https://www.assemblyai.com/blog/pytorch-vs-tensorflow-in-2022/

```
yeah, that would be awesome - we've been calling that "vertical tiling" or "vertical slicing"
to an extent it's kind of what our linalg fusion does by not looking at layers and instead looking at the loop structure - only (today) it has some specific requirements about the loops it can put together
there's a few scales of this approach, though, and the higher level ones (partitioning entire slices of the model across devices) are still TBD
the lower level ones (what our fusion does and some linalg transformations for slicing things up) are easier for us to do automatically, while the higher level ones may need frontend involvement
in the jax world it's https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html pmap & co, which I don't think we support yet
if we did support it we could more easily use multiple device queues/multiple devices by just using the pmap as the partitioning mechanism - cheat our way to distribution :P
since it looks like what they did involved training to handle it, they're likely more on the jax pmap side of things
(meaning that if we supported pmap - even running serially by just translating it to an scf.for loop - we could do what they did in the paper)
(we could do it without pmap and stuff today, mostly just anchoring on that as a nice user-level mechanism)
```

* https://en.wikipedia.org/wiki/Cache-oblivious_algorithm
* Parallel matrix transpose algorithms on distributed memory concurrent computers
  - http://www.netlib.org/utk/people/JackDongarra/PAPERS/069_1995_parallel-matrix-transpose-algorithms-on-distributed-memory.pdf

## Other references:
* Awesome Tensor Compilers (Papper collection)
  - https://github.com/merrymercy/awesome-tensor-compilers
* A High-Performance Sparse Tensor Algebra Compiler in Multi-Level IR
  - https://arxiv.org/pdf/2102.05187.pdf

## Read
* 字节跳动 Service Mesh 数据面编译优化实践
  - https://mp.weixin.qq.com/s/56RYaad3YnUSn3NgXiD_Ow?fbclid=IwAR0M-W_AXSIAMrJdD8qrBH_bUsl77vkT-gpoo-xSlm8NsAp4sm8Y8ifoXe0
* What is Envoy
  - https://www.envoyproxy.io/docs/envoy/latest/intro/what_is_envoy
* MPI
  * https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report.pdf#687
  * The goal of the Message-Passing Interface simply stated is to develop a widely used
    standard for writing message-passing programs. As such the interface should establish a
    practical, portable, efficient, and flexible standard for message passing.
    A complete list of goals follows
    * Design an application programming interface (not necessarily for compilers or a system
      implementation library).
    * Allow efficient communication: Avoid memory-to-memory copying, allow overlap of
      computation and communication, and offload to communication co-processors, where
      available.
    * Allow for implementations that can be used in a heterogeneous environment.
    * Allow convenient C and Fortran bindings for the interface
    * Assume a reliable communication interface: the user need not cope with communication failures. 
      Such failures are dealt with by the underlying communication subsystem.
    * Define an interface that can be implemented on many vendor’s platforms, with no
      significant changes in the underlying communication and system software.
    * Semantics of the interface should be language independent.
    * The interface should be designed to allow for thread safety.

* An Introduction to CUDA-Aware MPI
  - https://developer.nvidia.com/blog/introduction-cuda-aware-mpi/
* 2020 LLVM in HPC Workshop: Keynote: MLIR: an Agile Infrastructure for Building a Compiler Ecosystem
  * https://llvm-hpc-2020-workshop.github.io/presentations/llvmhpc2020-amini.pdf  
  * https://www.youtube.com/watch?v=0bxyZDGs-aA
* TVM Conf 2020 - Day 2 - MLIR and MLIR in the TensorFlow Ecosystem
  * https://www.youtube.com/watch?v=FE-Gw6YTd8s
* Systolic Arrays
  * https://www.sciencedirect.com/topics/computer-science/systolic-arrays
  * C = AB, mkn = 4x4x4 matrix multiplication implementation:
  <img width="453" alt="image" src="https://user-images.githubusercontent.com/5351229/166235438-d2a73aa3-260c-48ce-98e2-e3e52dfcce31.png">
* Linux performance observability tools
  - https://www.brendangregg.com/Perf/linux_observability_tools.png

  
