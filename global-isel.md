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
