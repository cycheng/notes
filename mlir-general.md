Contents:
=========
* [Base class](#base-class)
* [HowTo:](#howto)
  * Dump region (function) / block in gdb
  * Traverse value recursively
  * Convert TOSA reshape -> linalg reshape

### Base class
* mlir::Value
  * This class has value-type semantics and is just a simple wrapper around a ValueImpl
    * owner by a block (BlockArgument)
    * owner by a Operation (OpResult)
  * An SSA value is either a BlockArgument or the result of an operation

### HowTo:
##### Dump region (function) / block 
  * b is a mlir::OpBuilder
  ```shell
  # dump a block
  (gdb) p b.getBlock()->dump()
  # dump block's parent (mlir::Region)'s fist block
  (gdb) p b.getBlock()->getParent()->front().dump()
  # dump something like a function
  (gdb) p b.getBlock()->getParent()->getParentRegion()->getParentOp()->dump()
  ```

##### Traverse value recursively
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
##### Topological sort (visiting)
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
##### Convert TOSA reshape -> linalg reshape
  * [source](https://github.com/llvm/llvm-project/blob/db0486c46fe187475e4b01a401e14b2def593733/mlir/lib/Conversion/TosaToLinalg/TosaToLinalg.cpp#L1607-L1633)
    ```cpp
      auto getIdentityExprs = [&rewriter](int n) {
        SmallVector<AffineExpr, 4> exprs;
        for (int i = 0; i < n; ++i)
          exprs.push_back(rewriter.getAffineDimExpr(i));
        return exprs;
      };
      Location loc = reshape.getLoc();
      int64_t totalElems =
          std::accumulate(expandedShape.begin(), expandedShape.end(), 1,
                          std::multiplies<int64_t>());
      auto elemTy = operandTy.getElementType();
      SmallVector<ReassociationExprs, 4> collapsingMap = {
          // Use operandTy here because we need to collapse all operands
          // dimensions.
          getIdentityExprs(operandTy.getShape().size())};
      SmallVector<ReassociationExprs, 4> expandingMap = {
          // Use resultTy here because we need to expand to all result
          // dimensions.
          getIdentityExprs(resultTy.getShape().size())};

      auto collapsedTy = RankedTensorType::get({totalElems}, elemTy);
      Value collapsedOp = rewriter.create<linalg::TensorCollapseShapeOp>(
          loc, collapsedTy, adaptor.getOperands()[0], collapsingMap);
    ```
    collapse to 1 dimension shape first, then expand to required shape.
    ```c++
      rewriter.replaceOpWithNewOp<linalg::TensorExpandShapeOp>(
          reshape, resultTy, collapsedOp, expandingMap);

      return success();
    ```
