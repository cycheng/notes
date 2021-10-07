Contents:
=========
* Generate code for arm/amdgcn on x86

### Generate code for arm/amdgcn on x86
* arm's --target: https://developer.arm.com/documentation/dui0774/b/compiler-command-line-options/-mcpu
* amdgcn: https://llvm.org/docs/AMDGPUUsage.html
* examples:
  ```shell
  clang-13 --target=aarch64-arm-none-eabi -S -O3 -mllvm -print-after-all test.c > test.ll 2>&1
  clang-13 --target=armv7a-arm-none-eabi -mcpu=cortex-a15 -mfloat-abi=hard -S -O3 test.c
  clang-13 --target=amdgcn -S -O3 test.c
  # test global i-sel
  clang-13 --target=aarch64-arm-none-eabi -fglobal-isel -S -O3 test.c
  ```
