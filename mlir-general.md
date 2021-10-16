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
