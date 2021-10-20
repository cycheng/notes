Contents:
=========
* [Code: New pass manager add pass method](#code-new-pass-manager-add-pass-method)
* [Generate code for arm/amdgcn on x86](#generate-code-for-armamdgcn-on-x86)

### Code: New pass manager add pass method

* [llvm source](https://github.com/llvm/llvm-project/blob/2caf7571e1020ae1024ab3f2d52ecc9aea85687d/llvm/include/llvm/IR/PassManager.h#L550-L574)
  ```cpp
    template <typename IRUnitT,
              typename AnalysisManagerT = AnalysisManager<IRUnitT>,
              typename... ExtraArgTs>
    class PassManager : public PassInfoMixin<
                           PassManager<IRUnitT, AnalysisManagerT, ExtraArgTs...>> {
    public:
     ..
     template <typename PassT>
     LLVM_ATTRIBUTE_MINSIZE
         std::enable_if_t<!std::is_same<PassT, PassManager>::value>
  ```
  => If (PassT != PassManager), the subsition is performed and the function is implemented.
  ```c++
         addPass(PassT &&Pass) {
       using PassModelT =
           detail::PassModel<IRUnitT, PassT, PreservedAnalyses, AnalysisManagerT,
                             ExtraArgTs...>;
       // Do not use make_unique or emplace_back, they cause too many template
       // instantiations, causing terrible compile times.
       Passes.push_back(std::unique_ptr<PassConceptT>(
  ```
  => If Pass is rvalue, forward the rvalue to PassModelT, else use lvalue, see the
     exammple in [c++ forward](https://www.cplusplus.com/reference/utility/forward/)
  ```c++
           new PassModelT(std::forward<PassT>(Pass))));
     }
     
     /// When adding a pass manager pass that has the same type as this pass
     /// manager, simply move the passes over. This is because we don't have use
     /// cases rely on executing nested pass managers. Doing this could reduce
     /// implementation complexity and avoid potential invalidation issues that may
     /// happen with nested pass managers of the same type.
     template <typename PassT>
     LLVM_ATTRIBUTE_MINSIZE
         std::enable_if_t<std::is_same<PassT, PassManager>::value>
  ```
  => If (PassT == PassManager), the subsition is performed and the function is implemented.
  ```c++
         addPass(PassT &&Pass) {
       for (auto &P : Pass.Passes)
         Passes.push_back(std::move(P));
     }
  ```
  See:
  * [Substitution failure is not an error](https://en.wikipedia.org/wiki/Substitution_failure_is_not_an_error)
  * [What does template<class = enable_if_t<...>> do?](https://stackoverflow.com/questions/49659590/what-does-templateclass-enable-if-t-do)

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
