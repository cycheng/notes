Contents:
=========
* Linalg ElementWise fusion (affine map)
* Build IREE

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
* build

```shell
# install ccache (optional)
sudo apt install -y ccache

cmake -GNinja -B /home/cycheng/build/iree \
    -S /home/cycheng/iree -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER=clang-12 \
    -DCMAKE_CXX_COMPILER=clang++-12 \
    -DIREE_ENABLE_LLD=ON \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DIREE_HAL_DRIVERS_TO_BUILD="DyLib;VMVX;Vulkan" \
    -DIREE_TARGET_BACKENDS_TO_BUILD="DYLIB-LLVM-AOT;WASM-LLVM-AOT;Vulkan-SPIRV;VMVX"

cmake --build /home/cycheng/build/iree
```
