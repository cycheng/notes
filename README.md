# notes
## IREE+MLIR+LLVM

## TODO Reading List:
* https://en.wikipedia.org/wiki/Cache-oblivious_algorithm
* Parallel matrix transpose algorithms on distributed memory concurrent computers
  http://www.netlib.org/utk/people/JackDongarra/PAPERS/069_1995_parallel-matrix-transpose-algorithms-on-distributed-memory.pdf

## Other references:
* Awesome Tensor Compilers (Papper collection)
  https://github.com/merrymercy/awesome-tensor-compilers
* A High-Performance Sparse Tensor Algebra Compiler in Multi-Level IR
  https://arxiv.org/pdf/2102.05187.pdf

## Read
* 字节跳动 Service Mesh 数据面编译优化实践
  https://mp.weixin.qq.com/s/56RYaad3YnUSn3NgXiD_Ow?fbclid=IwAR0M-W_AXSIAMrJdD8qrBH_bUsl77vkT-gpoo-xSlm8NsAp4sm8Y8ifoXe0
* What is Envoy
  https://www.envoyproxy.io/docs/envoy/latest/intro/what_is_envoy
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
  https://developer.nvidia.com/blog/introduction-cuda-aware-mpi/
* 2020 LLVM in HPC Workshop: Keynote: MLIR: an Agile Infrastructure for Building a Compiler Ecosystem
  * https://llvm-hpc-2020-workshop.github.io/presentations/llvmhpc2020-amini.pdf  
  * https://www.youtube.com/watch?v=0bxyZDGs-aA
* TVM Conf 2020 - Day 2 - MLIR and MLIR in the TensorFlow Ecosystem
  * https://www.youtube.com/watch?v=FE-Gw6YTd8s
  
