Contents:
=========
* [LLVM Dev](#llvm-dev)
  * [2015 A Proposal for Global Instruction Selection](#2015-a-proposal-for-global-instruction-selection)

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
* LLVM IR -> IRTranslator -> GMI -> Legalizer -> RegBank Select
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
