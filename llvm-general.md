Contents:
=========
* [Automatic verification of LLVM optimizations](#automatic-verification-of-llvm-optimizations)
* [Code: New pass manager add pass method](#code-new-pass-manager-add-pass-method)
* [Generate code for arm/amdgcn on x86](#generate-code-for-armamdgcn-on-x86)
* [clang -ftime-trace](#clang--ftime-trace)

### Automatic verification of LLVM optimizations
* Online tool:
  - https://alive2.llvm.org/ce/
  - https://www.philipzucker.com/z3-rise4fun/

* github
  - https://github.com/AliveToolkit/alive2
  - https://github.com/nunoplopes/alive/tree/newsema
    * https://github.com/nunoplopes/alive/tree/newsema/rise4fun/examples
  - https://github.com/Z3Prover/z3
    theorem prover

* Thomas B. Jablin's Patch:
  - Fix Side-Conditions in SimplifyCFG for Creating Switches from InstCombine And Mask'd Comparisons
    https://reviews.llvm.org/D21417
    * Teach SimplifyCFG to Create Switches from InstCombine Or Mask'd Comparisons
      https://reviews.llvm.org/D21397
    * Reorder SimplifyCFG and SROA?
      https://reviews.llvm.org/D21315
    * [ppc] slow code for complex boolean expression
      https://bugs.llvm.org/show_bug.cgi?id=27555

* Reference:
  - http://research.tedneward.com/reading/compilers.correctness.html


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

### clang -ftime-trace

* https://github.com/llvm-mirror/clang/blob/master/tools/driver/cc1_main.cpp
  ```cpp
  if (Clang->getFrontendOpts().TimeTrace) {
    llvm::timeTraceProfilerInitialize(
        Clang->getFrontendOpts().TimeTraceGranularity);
  }
  
  ...
  
  // If any timers were active but haven't been destroyed yet, print their
  // results now.  This happens in -disable-free mode.
  llvm::TimerGroup::printAll(llvm::errs());
  llvm::TimerGroup::clearAll();

  if (llvm::timeTraceProfilerEnabled()) {
    SmallString<128> Path(Clang->getFrontendOpts().OutputFile);
    llvm::sys::path::replace_extension(Path, "json");
    if (auto profilerOutput =
            Clang->createOutputFile(Path.str(),
                                    /*Binary=*/false,
                                    /*RemoveFileOnSignal=*/false, "",
                                    /*Extension=*/"json",
                                    /*useTemporary=*/false)) {

      llvm::timeTraceProfilerWrite(*profilerOutput);
      // FIXME(ibiryukov): make profilerOutput flush in destructor instead.
      profilerOutput->flush();
      llvm::timeTraceProfilerCleanup();
    }
  }

  ```
