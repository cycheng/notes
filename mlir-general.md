Contents:
=========
* Base class
* HowTo:
  * Dump region (function) / block in gdb
  * Traverse value recursively

### Base class
* mlir::Value
  * This class has value-type semantics and is just a simple wrapper around a ValueImpl
    * owner by a block (BlockArgument)
    * owner by a Operation (OpResult)
  * An SSA value is either a BlockArgument or the result of an operation

### HowTo:
* Dump region (function) / block 
  * b is a mlir::OpBuilder
  ```shell
  # dump a block
  (gdb) p b.getBlock()->dump()
  # dump block's parent (mlir::Region)'s fist block
  (gdb) p b.getBlock()->getParent()->front().dump()
  # dump something like a function
  (gdb) p b.getBlock()->getParent()->getParentRegion()->getParentOp()->dump()
  ```

* Traverse value recursively
  * https://github.com/google/iree/blob/8214b36294f7236622176939068479eeba574e29/iree/compiler/Dialect/Flow/Transforms/DispatchLinalgOnTensors.cpp#L396-L415
    ```c++
    static void getUsedValuesDefinedAboveAfterCloningOps(
        OpBuilder &builder, IREE::Flow::DispatchWorkgroupsOp dispatchOp,
        llvm::SetVector<Value> &valuesDefinedAbove) {
      llvm::SmallVector<Operation *> clonedOps;
      llvm::SetVector<Value> visited;
      SmallVector<Value, 4> worklist;
      worklist.assign(valuesDefinedAbove.begin(), valuesDefinedAbove.end());
      valuesDefinedAbove.clear();
      while (!worklist.empty()) {
        Value outsideValue = worklist.pop_back_val();
        if (visited.count(outsideValue)) continue;
        visited.insert(outsideValue);
        Operation *definingOp = outsideValue.getDefiningOp();
        if (!definingOp || !(isClonableIntoDispatchOp(definingOp))) {
          valuesDefinedAbove.insert(outsideValue);
          continue;
        }
        clonedOps.push_back(definingOp);
        worklist.append(definingOp->operand_begin(), definingOp->operand_end());
      }
    ```
* Topological sort (visiting)
  * https://github.com/google/iree/blob/8214b36294f7236622176939068479eeba574e29/iree/compiler/Dialect/Flow/Transforms/DispatchLinalgOnTensors.cpp#L311-L391
    ```c++
    /// Reorders the operations in `ops` such that they could be inlined into the
    /// dispatch region in that order to satisfy dependencies.
    static SmallVector<Operation *> orderOperations(ArrayRef<Operation *> ops) {
      llvm::SmallMapVector<Operation *, SmallVector<Operation *>, 16>
          insertAfterMap;
      llvm::SetVector<Operation *> opSet(ops.begin(), ops.end());
      llvm::SetVector<Operation *> leafOps(ops.begin(), ops.end());
      for (auto op : ops) {
        for (auto operand : op->getOperands()) {
          auto definingOp = operand.getDefiningOp();
          if (!definingOp || !opSet.count(definingOp)) continue;
          insertAfterMap[definingOp].push_back(op);
          if (leafOps.count(op)) leafOps.remove(op);
        }
      }

      SmallVector<Operation *> orderedOps(leafOps.begin(), leafOps.end());
      orderedOps.reserve(ops.size());
      llvm::SmallPtrSet<Operation *, 16> processed;
      processed.insert(leafOps.begin(), leafOps.end());

      ArrayRef<Operation *> readyOps(orderedOps);
      size_t startPos = 0;
      while (!readyOps.empty()) {
        auto op = readyOps.front();
        startPos++;
        for (auto insertAfterOp : insertAfterMap[op]) {
          if (processed.count(insertAfterOp)) continue;
          if (llvm::all_of(insertAfterOp->getOperands(), [&](Value operand) {
                Operation *operandDefiningOp = operand.getDefiningOp();
                return !operandDefiningOp || !opSet.count(operandDefiningOp) ||
                       processed.count(operandDefiningOp);
              })) {
            orderedOps.push_back(insertAfterOp);
            processed.insert(insertAfterOp);
          }
        }
        readyOps = ArrayRef<Operation *>(orderedOps).drop_front(startPos);
      }
      return orderedOps;
    }
    ```

