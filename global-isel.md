Contents:
=========
* [LLVM Dev](#llvm-dev)
  * [2015 A Proposal for Global Instruction Selection](#2015-a-proposal-for-global-instruction-selection)
  * [2019 Generating Optimized Code with GlobalISel](#2019-generating-optimized-code-with-globalisel)

## LLVM Dev

### 2019 Generating Optimized Code with GlobalISel
* https://www.youtube.com/watch?v=8427bl_7k1g
* https://llvm.org/devmtg/2019-10/slides/SandersKeles-GeneratingOptimizedCodewithGlobalISel.pdf
* From Apple GPU Compiler team.
* Apple GPU Compiler is now using GlobalISel
  * 2017, Apple GPU compiler got GlobalISel fully working for their target
    Fast compile time but codegen quality was significantly lower
  * 2019, GlobalISel going beyond "it works"
    * Enabled in ios13, running on millions of Apple devices
* What is GlobalISel (V.S. SelectionDAGISel)
  * A new instruction selection framework
  * Supports more global optimization
    * Able to match cross basic blocks
  * More flexible
    * GlobalISel can range from the speed of FastISel to the quality of SelectionDAGISel
  * Easier to maintain, understand, and test
  * It keeps all the state in the machine IR
    * We can dump the machine function at any point and it accurately reflects the program,
      there is no need to look at temporary information for each pass

#### Anatomy of GlobalISel
<img width="837" alt="image" src="https://user-images.githubusercontent.com/5351229/174483519-11a28e86-53c9-4395-a3a4-4d2cb5c67f53.png">

  * GlobalISel goes straight to the MIR representation, it doesn't pass through other
    representation
  * IR Translator
    * Takes llvm ir and converts it to generic MIR
  * Legalizer replaces unsupported operations with supported one's
  * [Register Bank Selector:](https://llvm.org/docs/GlobalISel/RegBankSelect.html)
    * This pass constrains the Generic Virtual Registers operands of generic instructions
      to some Register Bank
  * Instruction Selector: select target instructions

#### Combiner
<img width="858" alt="image" src="https://user-images.githubusercontent.com/5351229/174484591-09d34fdd-17ed-4a68-9c34-d1be60a8ce0f.png">

* Combiner passes optimize (G)MIR and MIR by transforming some patterns into something
  more desirable
* We can put combiner passes in different places, with notes:
  * IRTranslator/Legalizer/RegisterBankSelector/InstructionSelector have some requirements
    to combiner passes. The requirements get stricter later in the pipeline.
    * For example, pre-legalizer is fairly relaxed, but post RegisterBankSelector has to be
      careful to avoid disturbing assigned register banks
* Why we need combiners?
  * In 2017, on average, GlobalISel was generating 30% more instructions compared to
    SelectionDAGISel
  * Today (2019), on average, the difference in instruction count < 2%, and runtime
    performance is similar compared to SelectionDAGISel
  * There is some work to do in order to beyond this, but it looks promising
* Compile Time Performance
  * For GPU compiler it's one of the important metrics (compile shader in runtime)
  * Today (2019), the (end to end? Probaly!? Speaker didn't say more details) pipeline with
    GlobalISel is: 
    * On average 8% faster than pipeline with SelectionDAGISel
    * GlobalISel is about 45% faster than SelectionDAGISel

#### Features needed to improve codegen quality and compile time
* Common Subexpression Elimination (CSE)
* Combiners
* KnownBits
* SimplifyDemandedBits

#### Common Subexpression Elimination (CSE)
* Why not use MachineCSE?
  * It's expensive to run after each GlobalISel pass
  * There are some target specific combines generating suboptimal code
* Continuous CSE approach
  * Instructions are CSE'd at creation time using CSEMIRBuilder
  * Information is provided by an analysis pass
  * It's currently BasicBlock-local, but we might make it global in the future
  * It's now only supports a subset of generic operations, but we will extend
    it soon
  * We also plan to add support to CSE target instructions
* Things to be aware of
  * It's easy to use CSE by CSEMIRBuilder, but CSE needs to be informed when something
    (MachineInstrs) is changed, such as
    * Erasing a MIR, Switching OPCode, Replacing registers
  * For creation and erasure, it installs a machine function delegate to handle them
    automatically, so there is no need to inform (CSEMIRBuilder) these kind of changes
  * For other kind of changes, the "change observer" needs to be called so that CSE
    infrastructure knows about the changes
  * "Observer" means to maintain information as MIR changed
* Compile Time Cost
  * CSEMIRBuilder allows us to generate better code, but what about compile time cost?
  * We were expecting this to come at a big compile-time cost, but it didn't cause a
    significant regression
  * For some cases it actually improved compile time, because CSE reduce number of
    instructions so later passes had less work to do
  * Overall, CSE helped to improve code quality with a good price

#### Combiners
* A combiner is a pass which applies a set of combine rules to MIR
* It's probably the most important component for producing good code, and
* It can be quite expensive in terms of compile-time
* What is a combine?
  * An optimization that transforms a pattern into something more desirable, e.g.
    ```llvm
    define i32 @foo(i8 %in) {                   
      %ext1 = zext i8 %in to i16              
      %ext2 = zext i16 %ext1 to i32      =>       %ext2 = zext i8 %in to i32
      ret i32 %ext2
    }
    ```
    The first instruction is actually redundant and can be combined with the second
    instruction
* Example:
  * A Basic Combiner
    ```c++
    bool MyTargetCombinerInfo::combine(GISelChangeObserver &Observer,
                                       MachineInstr &MI,
                                       MachineIRBuilder &B) const {
        MyTargetCombinerHelper TCH(Observer, B, KB);
        // ...
        // Try all combines.
        if (OptimizeAggresively)
            return TCH.tryCombine(MI);
        // Combine COPY only.
        if (MI.getOpcode() == TargetOpcode::COPY)
            return TCH.tryCombineCopy(MI);
        return false;
    }
    ```
    * Flexible: It's easily to include or exclude combines, we can change our strategy
      based on our target/sub-target/opt-level
  * A Simple Combine: zext(zext x) -> zext x
    ```c++
    bool MyTargetCombinerHelper::combineExt(GISelChangeObserver &Observer,
                                            MachineInstr &MI,
                                            MachineIRBuilder &B) const {
        // ..
        // Combine zext(zext x) -> zext x
        if (MI.getOpcode() == TargetOpcode::G_ZEXT) {
            Register SrcReg = MI.getOperand(1).getReg();
            MachineInstr *SrcMI = MRI.getVRegDef(SrcReg);
            // Check if SrcMI is a G_ZEXT.
            if (SrcMI->getOpcode() == TargetOpcode::G_ZEXT) {
                SrcReg = SrcMI->getOperand(1).getReg();
                B.buildZExt(Reg, SrcReg);
                MI.eraseFromParent();
                return true;
            }
        }
        // ...
    }
    ```
    * It's easy to add a combine rule
    * But, even though the logic is pretty simple here, it's a bit hard to follow
      all of these checks. We found this approach difficult because it tends to require
      a lot of code for some combines, this is why we add MIPatternMatch

* GlobalISel Combiner has 3 main pieces:
* A Basic Combiner
* 


* 
* KnownBits
* SimplifyDemandedBits



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
* LLVM IR -> IRTranslator -> GMI -> Legalizer -> RegBank Select -> Select
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
* Legalizer
  ```llvm
    foo:
      val(FPR,64) = VMOVDRR R0,R1
      addr(32) = COPY R2
      loaded(64) = gLD (double) addr
      ; lower 64 bit and to 32 bit and
      and(64) = gAND (i64) val, loaded            lval(32),hval(32) = extract val
                                                  low(32),high(32) = extract loaded
                                                  land(32) = gAND (i32) lval, low
                                                  hand(32) = gAND (i32) hval, high
                                                  and(64) = build_sequence land, hand
      R0,R1 = VMOVRRD and
      tBX_RET R0<imp-use>,R1<imp-use>
  ```
  ```llvm
      lval(32),hval(32) = extract val             lval(32),hval(32) = VMOVRRD val
      low(32),high(32) = extract loaded           low(32),high(32) = VMOVRRD loaded
      land(32) = gAND (i32) lval, low             land(32) = gAND (i32) lval, low
      hand(32) = gAND (i32) hval, high            hand(32) = gAND (i32) hval, high
      and(64) = build_sequence land, hand         and(64) = VMOVDRR land, hand
  ```

* RegBankSelect
  * Fast way:
  ```llvm
    foo:
      val(FPR,64) = VMOVDRR R0,R1                 val(FPR,64) = VMOVDRR R0,R1
      addr(32) = COPY R2                          addr(GPR,32) = COPY R2
      loaded(64) = gLD (double) addr              loaded(FPR,64) = gLD (double) addr
      lval(32),hval(32) = VMOVRRD val             lval(GPR,32),hval(GPR,32) = VMOVRRD val
      low(32),high(32) = VMOVRRD loaded           low(GPR,32),high(GPR,32) = VMOVRRD loaded
      land(32) = gAND (i32) lval, low             land(GPR,32) = gAND (i32) lval, low
      hand(32) = gAND (i32) hval, high            hand(GPR,32) = gAND (i32) hval, high
      and(64) = VMOVDRR land, hand                and(FPR,64) = VMOVDRR land, hand
      R0,R1 = VMOVRRD and                         R0,R1 = VMOVRRD and
      tBX_RET R0<imp-use>,R1<imp-use>             tBX_RET R0<imp-use>,R1<imp-use>
  ```

  * Avoid across domain penalties:
    * Asks the target what register banks are supported for a given opcode
  ```llvm
      val(FPR,64) = VMOVDRR R0,R1                 1. [{(FPR,0xFF…FF),1},
                                                  2.  {(GPR,0xFFFF…0000)(GPR,0x0000…FFFF),0}]
  ```
    * (1) The target could say, it can create this definition in FPR register, 0xFF.. is
      the mask means the value is in the register bank, the value can be produced in cost 1
    * (2) The target can also produce the value in two GPR, one contains upper 32-bits, the
      other contains lower 32-bits, the cost is 0 
  
  ```llvm
      lval(32),hval(32) = VMOVRRD val             1. [{(FPR,0xFF…FF),1},
                                                  2.  {(GPR,0xFFFF…0000)(GPR,0x0000…FFFF),0}]
  ```
    * We can have a pass to analysis and avoid copy from FPR to GPR (For arm the cross copy
      can cost 20 cycles

  ```llvm
      val(FPR,64) = VMOVDRR R0,R1                 val1(GPR,32),val2(GPR,32) = COPIES R0,R1
      lval(32),hval(32) = VMOVRRD val             lval(32),hval(32) = COPIES val1, val2 
  ```
    * This looks like legalization? => Reuse toolkits from Legalizer!

  ```llvm
      loaded(64) = gLD (double) addr              loaded1(GPR,32) = gLD (i32) addr
                                                  loaded2(GPR,32) = gLD (i32) addr, #4

      low(32),high(32) = VMOVRRD loaded           low(32),high(32) = COPIES loaded1,loaded2
      
      and(64) = VMOVDRR land, hand                and1(GPR,32), and2(GPR,32) = COPIES land, hand 
      R0,R1 = VMOVRRD and                         R0,R1 = COPIES and1, and2
  ```

* Select
  ```llvm
    foo:
      val1(GPR,32),val2(GPR,32) = COPIES R0,R1
      addr(GPR,32) = COPY R2
      loaded1(GPR,32) = gLD (i32) addr                    loaded1(GPR,32) = t2LDRi12 (i32) addr
      loaded2(GPR,32) = gLD (i32) addr, #4                loaded2(GPR,32) = t2LDRi12 (i32) addr, #4
      lval(GPR,32),hval(GPR,32) = COPIES val1, val2
      low(GPR,32),high(GPR,32) = COPIES loaded1,loaded2
      land(GPR,32) = gAND (i32) lval, low                 land(GPR,32) = t2ANDrr (i32) lval, low
      hand(GPR,32) = gAND (i32) hval, high                hand(GPR,32) = t2ANDrr (i32) hval, high
      and1(GPR,32), and2(GPR,32) = COPIES land, hand
      R0,R1 = COPIES and1, and2
      tBX_RET R0<imp-use>,R1<imp-use>
  ```
    * (G)MIR -> MIR: In-place morphing
    * State expressed in the IR
    * State machine
    * Iterate until everything is selected
    * *Combines across basic blocks*

#### Summary/How do we get there?
Skip!

#### Q&A
* What about control flow representation, especially can you handle predicates at that stage?
  * If you are able to model predicate then you can do the same thing on G-MIR. Anything you
    can do in the backend you would be able to do in this GlobalISel
* You didn't cover scheduling?
  * With the MachineInstr we don't need any scheduling anymore because we have already produced
    the MIR from a sequentialized version of the representation
  * For MIR scheduling this just leave to existing scheduler
* (Cont.) I mean moving instructions between basicblocks
  * Won't do in prototype
  * At this point, this will be done later, e.g. machine CSE / LICM / ..
  * We may think to add a generic optimizer for it
* How much of the existing code can be reused?
  * ...
* Regarding RegBankSelector, Why not leave to register allocator?
  * ...
* ...
