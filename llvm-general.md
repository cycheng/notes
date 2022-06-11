Contents:
=========
* [LLVM Dev](#llvm-dev)
  * [2015 A Proposal for Global Instruction Selection](#2015-a-proposal-for-global-instruction-selection)
* [Automatic verification of LLVM optimizations](#automatic-verification-of-llvm-optimizations)
* [Code: New pass manager add pass method](#code-new-pass-manager-add-pass-method)
* [Generate code for arm/amdgcn on x86](#generate-code-for-armamdgcn-on-x86)
* [clang -ftime-trace](#clang--ftime-trace)
* [TIPs](#tips)
  * [clone-function](#clone-function)

## LLVM Dev

### 2015 A Proposal for Global Instruction Selection
* https://www.youtube.com/watch?v=F6GGbYtae3g
* https://llvm.org/devmtg/2015-10/slides/Colombet-GlobalInstructionSelection.pdf
* SelectionDAG (SD) ISel, two paths:
  * LLVM IR -> SDBuilder -> SDNode -> DAGCombiner (1) -> Legalize* (2) -> DAGCombiner 
            -> Select (4) -> Schedule (5) -> MachineInstr
    * Build new representation: SDNode
    * (1): Do canonicalization and peephole optimization
    * (2): Translate into what is supported for the target
    * (4): Select these nodes into target specific nodes
    * (5): Linearize those graphs to be able to produce the sequential representation 
           which is using MachineInsr, so we have to schedule those graphs
    * We have to do it for every BasicBlock, that's why we have FastISel
  * LLVM IR -> FastISel -> MachineInstr
    * A direct translation from llvm ir to MachineInstr
    * "Fast": Compromise what it can support as input from llvm ir
    * => It can fail, then fall back to path I
* Problems with SDISel
  * Basic block scope: E.g. we want to optimize the selection of addressing mode,
    but we may not see all the uses of the addressing mode what we want to expose
    on all the definition
    * llvm came up some passes, e.g. [CodeGenPrepare]([url](https://llvm.org/doxygen/CodeGenPrepare_8cpp_source.html)) 
      pass to help this problem. The pass pushes instructions within basic blocks of
      duplicating instructions into the same basic block so that we expose this
      optimization opportunity to the SD ISel
    * Constant hoist: Put constants together which can be reused in different BB
  * SDNode IR: take time and memory for the conversion
  * Monolithic: Hard to debug
* Goals of GlobalISel:
  * Global: Function scope
  * Fast: compile time
  * Shared code for fast and good paths
  * IR that represents ISA concepts better
  * More flexible/Easier to maintain/ ..

#### Steps:
* LLVM IR -> IRTranslator -> GMI -> Legalizer
* IRTranslator
  * llvm ir to generic(G) MachineInstr
    * One IR to 0..* G MIR
      ```llvm
      define double @foo(                         foo:
        double %val,                                val(64) = ..
        double* %addr) {                            addr(32) = .. 
        %intval = bitcast double %val to i64        (nop)
        %loaded = load double, double* %addr        loaded(64) = gLD (double) addr
        %mask = bitcast double %loaded to i64       (nop)
        %and = and i64 %intval, %mask               and(64) = gAND (i64) val, loaded
        %res = bitcast i64 %and to double
        ret double %res                             ... = and
      }
      ```

    * ABI Lowering, e.g. arm
      ```llvm
                                                  foo:
                                                    val(FPR,64) = VMOVDRR R0,R1
                                                    addr(32) = COPY R2
                                                    loaded(64) = gLD (double) addr
                                                    and(64) = gAND (i64) val, loaded
                                                    R0,R1 = VMOVRRD and
                                                    tBX_RET R0<imp-use>,R1<imp-use>
      ```



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

## TIPs

### clone-function
* https://github.com/llvm/llvm-project/blob/main/llvm/tools/llvm-reduce/deltas/ReduceArguments.cpp
  ```c++
  for (auto *F : Funcs) {
    ValueToValueMapTy VMap;
    std::vector<WeakVH> InstToDelete;
    for (auto &A : F->args())
      if (!ArgsToKeep.count(&A)) {
        // By adding undesired arguments to the VMap, CloneFunction will remove
        // them from the resulting Function
        VMap[&A] = UndefValue::get(A.getType());
        for (auto *U : A.users())
          if (auto *I = dyn_cast<Instruction>(*&U))
            InstToDelete.push_back(I);
      }
    // Delete any (unique) instruction that uses the argument
    for (Value *V : InstToDelete) {
      if (!V)
        continue;
      auto *I = cast<Instruction>(V);
      I->replaceAllUsesWith(UndefValue::get(I->getType()));
      if (!I->isTerminator())
        I->eraseFromParent();
    }
  ```

