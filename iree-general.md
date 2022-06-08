Contents:
=========
* Linalg ElementWise fusion (affine map)
* mmt4d
* Build IREE
* [Interesting PRs/Issues](#interesting-prsissues)

### Linalg ElementWise fusion (affine map)
Ref: [2020-08-20: IREE CodeGen: MLIR Open Design Meeting Presentation](https://docs.google.com/presentation/d/1NetHjKAOYg49KixY5tELqFp6Zr2v8_ujGzWZ_3xvqC8/edit#slide=id.g91bae7fd94_1_43) 

```mlir
func @elementwise {
    %0 = ... : tensor<10x15xf32>
    %1 = ... : tensor<10x15xf32>
    %2 = ... : tensor<15xf32>
    %3 = “mhlo.add”(%0, %1) : (tensor<10x15xf32>, tensor<10x15xf32) -> tensor<10x15xf32>
    %4 = “mhlo.broadcast(%2) : (tensor<15xf32) -> tensor<10x15xf32>
    %5 = “mhlo.mul”(%3, %4) : (tensor<10x15xf32>, tensor<10x15xf32>) -> tensor<10x15xf32> 
    ...
}
```

Convert to linalg
```mlir
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
func @elementwise() {
  ...
  %3 = linalg.generic %0 %1 {..[#map0, #map1]..} {
       // add operation
  } : (tensor<10x15xf32>, tensor<10x15xf32>) -> tensor<10x15xf32>
  %4 = linalg.generic %2 {..[#map0, #map1]..} {
       ^bb0(%arg0 : f32, %arg1 : f32):
         linalg.yield %arg0 : f32 
  } : (tensor<15xf32>) -> tensor<10x15xf32>
  %5 = linalg.generic %3 %4 {..[#map0, #map1]..} {
       // mul operation
  } : (tensor<10x15xf32>, tensor<10x15xf32>) -> tensor<10x15xf32>
}
```

Elementwise op fusion (fuse add/broadcast/mulf)
```mlir
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
func @elementwise() {
  ...
  %3 = linalg.generic %0, %1, %2 { .. indexing_maps = [#map0, #map0, #map1, #map0] ..} {
       ^bb0(%arg0 : f32, %arg1 : f32, %arg2 : f32, %arg3 : f32):
         %0 = addf %arg0, %arg1 : f32
         %1 = mulf %0, %arg2 : f32
         linalg.yield %1 : f32
       } : (tensor<10x15xf32>, tensor<10x15xf32>, tensor<15xf32> -> (tensor<10x15xf32>)
  ...
}
```

**Note** the index for accessing **arg2**
```mlir
         // %arg0 and %arg1: affine_map<(d0, d1) -> (d0, d1)>
         // %arg2: affine_map<(d0, d1) -> (d1)>
         %0 = addf %arg0, %arg1 : f32
         %1 = mulf %0, %arg2 : f32

         => %0[i, j] = %arg0[i, j] + %arg1[i, j]
         => %1[i, j] = %0[i, j] + %arg2[i, j]

         => %arg2[i, j] -> %arg2[j]
```

### mmt4d
* https://google.github.io/iree/blog/2021-10-13-mmt4d/
  ```python
  def pack_2d_4d(operand, parallel_size, reduction_size):
   i1 = operand.shape[0] // parallel_size # M1
   i2 = parallel_size    # M0
   j1 = operand.shape[1] // reduction_size # K1
   j2 = reduction_size   # K0
   operand_4d = np.reshape(operand, [i1, i2, j1, j2])
   return np.transpose(operand_4d, [0, 2, 1, 3]) # [M1, K1, M0, K0]
  ```
  * operand_4d = np.reshape(operand, [i1, i2, j1, j2])
    * Original: 2D data with size MxN
    * Reshape to: 4D data with: 
      * i1 rows in dim0
      * For each row in dim0, there are i2 rows in dim1
      * For each row in dim1, there are j1 columns in dim2
      * For each column in dim2, there are j2 elements
      * E.g.
      ```mlir
      0 1 2 3    0 1 2 3    |0 1|2 3|   |0|1|
      4 5 6 7    -------
      -------    4 5 6 7
      8 9 a b
      c d e f
      ```
  * np.transpose(operand_4d, [0, 2, 1, 3])
    * Transpose dim 1 and 2, e.g.
      ```mlir
      0 1 | 2 3    0 1 | 4 5
      ----+----    ----+----
      4 5 | 6 7    2 3 | 6 7
      ----+----    ----+----
      ```

### Build IREE on Ubuntu (Test on 18.04)
* Install llvm/clang
```shell
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
./llvm.sh 12
sudo ./llvm.sh 12
```
* Install vulkan sdk and vulkan driver
``` shell
wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.2.182-bionic.list https://packages.lunarg.com/vulkan/1.2.182/lunarg-vulkan-1.2.182-bionic.list
sudo apt update
sudo apt install vulkan-sdk

# my gpu is intel hd 530 (supported by mesa-vulkan)
sudo apt install mesa-vulkan-drivers
# check vulkan driver works properly
vulkaninfo
```
* build (standard)
    ```shell
    # install ccache (optional)
    sudo apt install -y ccache

    cmake -GNinja -B /home/cycheng/build/iree/x86.rel \
        -S /home/cycheng/iree -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DIREE_ENABLE_ASSERTIONS=ON \
        -DCMAKE_C_COMPILER=clang-12 \
        -DCMAKE_CXX_COMPILER=clang++-12 \
        -DIREE_ENABLE_LLD=ON \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
        -DIREE_HAL_DRIVERS_TO_BUILD="DyLib;VMVX;Vulkan" \
        -DIREE_TARGET_BACKENDS_TO_BUILD="DYLIB-LLVM-AOT;WASM-LLVM-AOT;Vulkan-SPIRV;VMVX"

    cmake --build /home/cycheng/build/iree/x86.rel
    ```
* build (hacking): build everything with 'Release' mode except iree_compiler
  * hack iree/compiler/CMakeLists.txt
    ```cmake
    string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
    if(NOT "${uppercase_CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
      iree_select_compiler_opts(IREE_DEFAULT_COPTS
          CLANG_OR_GCC
            "-O0"
            "-g"
            "-gsplit-dwarf"
      )
    endif()
    ```
  * config llvm with 'split-dwarf' and some other options
    ```shell
    cmake -GNinja -B /home/cycheng/build/iree/x86.rel-dbg-compiler \
        -S /home/cycheng/iree -DCMAKE_BUILD_TYPE=Release \
        -DIREE_ENABLE_ASSERTIONS=ON \
        -DCMAKE_C_COMPILER=clang-12 \
        -DCMAKE_CXX_COMPILER=clang++-12 \
        -DIREE_ENABLE_LLD=ON \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
        -DIREE_HAL_DRIVERS_TO_BUILD="DyLib;VMVX;Vulkan" \
        -DIREE_TARGET_BACKENDS_TO_BUILD="DYLIB-LLVM-AOT;WASM-LLVM-AOT;Vulkan-SPIRV;VMVX"
        -DLLVM_USE_SPLIT_DWARF=ON \
        -DLLVM_OPTIMIZED_TABLEGEN=ON \
        -DLLVM_USE_NEWPM=ON

    cmake --build /home/cycheng/build/iree/x86.rel-dbg-compiler
    ```
* build (hacking): build in 'Debug' mode with better options
  * hack ./CMakeLists.txt
    ```cmake
    # copy paste the following code a line before "include(iree_setup_toolchain)"
    string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
    if("${uppercase_CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
      iree_select_compiler_opts(IREE_DEFAULT_COPTS
          CLANG_OR_GCC
              "-gsplit-dwarf"
      )
    endif()
    ```
  * config llvm with 'split-dwarf' and some other options
    ```shell
    cmake -GNinja -B /home/cycheng/build/iree/x86.dbg \
        -S /home/cycheng/iree -DCMAKE_BUILD_TYPE=Release \
        -DIREE_ENABLE_ASSERTIONS=ON \
        -DCMAKE_C_COMPILER=clang-12 \
        -DCMAKE_CXX_COMPILER=clang++-12 \
        -DIREE_ENABLE_LLD=ON \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
        -DIREE_HAL_DRIVERS_TO_BUILD="DyLib;VMVX;Vulkan" \
        -DIREE_TARGET_BACKENDS_TO_BUILD="DYLIB-LLVM-AOT;WASM-LLVM-AOT;Vulkan-SPIRV;VMVX"
        -DLLVM_USE_SPLIT_DWARF=ON \
        -DLLVM_OPTIMIZED_TABLEGEN=ON \
        -DLLVM_USE_NEWPM=ON

    cmake --build /home/cycheng/build/iree/x86.dbg
    ```
* Comparision (on i7-6700 4-cores 8-threads 32G-ram):
  * Binary size of iree-run-mlir
    * general (dbg): 1.5G
    * Debug.split-dwarf: 973M
    * Release + iree_compiler Debug: 337M
    * Release + iree_compiler Debug.split-dwarf: 260M
  * Build iree-run-mlir
    * general (dbg): 5m3.393s
    * Debug.split-dwarf:
    * Release + iree_compiler Debug: 4m34.420s
    * Release + iree_compiler Debug.split-dwarf:
  * incremental build time: modify iree/compiler/Dialect/Flow/IR/FlowOpFolders.cpp
    * general (dbg): 0m5.966s
    * Debug.split-dwarf: 0m3.627s
    * Release + iree_compiler Debug: 0m2.707s
    * Release + iree_compiler Debug.split-dwarf: 0m1.400s
* Ref: https://www.productive-cpp.com/improving-cpp-builds-with-split-dwarf/

### Interesting PRs/Issues
* [2022/05 Fuse with transpose](https://github.com/google/iree/pull/9103)
* [Only convert 1x1 conv to matmul if the width or height is not dynamic](https://github.com/google/iree/pull/9239)
* [2022/04/22: Allow dispatch region formation to fuse producer and consumer when producer has multiple uses.](https://github.com/google/iree/pull/8970)
* [Investigate a fusion opportunity on Layer Normalization](https://github.com/google/iree/issues/9139)
  - vertical fusion / horizontal fusion
* IREE with NPU targets?
  - 2022/05/17: 
    * https://github.com/google/iree/issues/9142#issue-1237929780
* [Add fusion of transpose with matmul/batchmatmul named ops](https://github.com/google/iree/issues/8827)
* [Is there any solution to generate SPIR-V code with Kernel capability?](https://github.com/google/iree/discussions/8831)

