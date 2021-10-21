
Working on https://github.com/google/iree/issues/6903

* Original:
  ```mlir
  %42 = flow.tensor.slice %41[%c0, %c20 for %c1, %c10] : tensor<1x40xf32> -> tensor<1x10xf32>
  %43 = flow.tensor.reshape %42 : tensor<1x10xf32> -> tensor<10xf32>
  ...
  %49 = flow.dispatch.workgroups[%c10, %c1, %c1](%43, %44, %46, %48) ...
      ..
      %63 = flow.dispatch.tensor.load %arg2, offsets = [%arg7], sizes = [%62], strides = [1]
      %67 = flow.dispatch.tensor.load %arg4, offsets = [%arg7], sizes = [%66], strides = [1]
      %69 = flow.dispatch.tensor.load %arg5, offsets = [%arg7], sizes = [%68], strides = [1]
  ```

#### Oct 22
* Create a reshape to reshape %41 tensor<1x40xf32> -> tensor<40xf32>
* works!!
  ```mlir
  %4x = flow.tensor.reshape %41 : tensor<1x40xf32> -> tensor<40xf32>
  %49 = flow.dispatch.workgroups[%c10, %c1, %c1](%4x, %44) ...
  ```
  ```mlir
          %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : !flow.dispatch.tensor<readonly:10xf32>
          %1 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : !flow.dispatch.tensor<readonly:40xf32>
          %2 = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : !flow.dispatch.tensor<writeonly:10xf32>
          %3 = flow.dispatch.tensor.load %1, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:40xf32> -> tensor<40xf32>
          ..
            %7 = tensor.extract_slice %3[%arg0] [%6] [1] : tensor<40xf32> to tensor<?xf32>
            %11 = tensor.extract_slice %3[%arg0] [%10] [1] : tensor<40xf32> to tensor<?xf32>
            ..
  ```

#### Oct 21
* Directly use %41 in the flow.dispatch.workgroups with required reshape:
  ```mlir
  func @main_dispatch_13() {
    ..
    %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : memref<10xf32>
    %1 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : memref<1x40xf32>
    %2 = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : memref<10xf32>
    %3 = memref.collapse_shape %1 [[0, 1]] : memref<1x40xf32> into memref<40xf32>
  ```
  
  but get an error: failed to materialize conversion for result #0 of operation 'hal.interface.binding.subspan' that remained live after conversion
  because of %1 is still used by %3.


#### Oct 12

* pull in slice into dispatch region
  ```cpp
      for (auto sliceOp : iter.getSecond()) {
        auto producer = sliceOp;
        Operation *clonedOrigProducer = rewriter.clone(*producer);
        rewriter.replaceOpWithinBlock(producer,
                                      clonedOrigProducer->getResults(),
                                      &dispatchOp.getRegion().front());
        map.map(producer.getResult(), clonedOrigProducer->getResult(0));
      }
  ```

* pull in reshape into dispatch region, and use correct source for cloned reshape
  ```cpp
      Region &region = dispatchOp.body();
      Block &block = region.front();

      unsigned idx = 0;
      for (auto reshapeOp : iter.getSecond()) {
        auto producer = reshapeOp;
        Operation *clonedOrigProducer = rewriter.clone(*producer, map);
        rewriter.replaceOpWithinBlock(producer,
                                      clonedOrigProducer->getResults(),
                                      &dispatchOp.getRegion().front());
  ```
* result
  ```mlir
    %0 = flow.dispatch.workgroups[<<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>](<<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>) : (tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32> =
        (%arg0: !flow.dispatch.tensor<readonly:10xf32>, %arg1: !flow.dispatch.tensor<readonly:10xf32>, %arg2: !flow.dispatch.tensor<readonly:10xf32>, %arg3: !flow.dispatch.tensor<readonly:10xf32>, %arg4: !flow.dispatch.tensor<writeonly:10xf32>) {
      %0 = flow.tensor.slice <<UNKNOWN SSA VALUE>>[<<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>> for <<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>] : tensor<1x40xf32> -> tensor<1x10xf32>
      %1 = flow.tensor.slice <<UNKNOWN SSA VALUE>>[<<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>> for <<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>] : tensor<1x40xf32> -> tensor<1x10xf32>
      %2 = flow.tensor.slice <<UNKNOWN SSA VALUE>>[<<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>> for <<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>] : tensor<1x40xf32> -> tensor<1x10xf32>
      %3 = flow.tensor.reshape %0 : tensor<1x10xf32> -> tensor<10xf32>
      %4 = flow.tensor.reshape %1 : tensor<1x10xf32> -> tensor<10xf32>
      %5 = flow.tensor.reshape %2 : tensor<1x10xf32> -> tensor<10xf32>
  ```
* Replace args with pulled in values
  ```cpp
        size_t id = reshapeIdx[iter.getFirst()][idx++];
        block.getArgument(id).replaceAllUsesWith(
            clonedOrigProducer->getResult(0));
  ```

* result
  ```mlir
          %10 = tensor.extract_slice %4[%arg3] [%9] [1] : tensor<10xf32> to tensor<?xf32>
          %12 = flow.dispatch.tensor.load %arg0, offsets = [%arg3], sizes = [%11], strides = [1] :
            !flow.dispatch.tensor<readonly:10xf32> -> tensor<?xf32>
          %14 = tensor.extract_slice %5[%arg3] [%13] [1] : tensor<10xf32> to tensor<?xf32>
          %16 = tensor.extract_slice %6[%arg3] [%15] [1] : tensor<10xf32> to tensor<?xf32>
  ```

* Do legalizeDispatchWorkgroupOperands(dispatchOp);
  * Copy the code from iree/compiler/Dialect/Flow/Transforms/DispatchLinalgOnTensors.cpp
    * legalizeDispatchWorkgroupOperands
    * getUsedValuesDefinedAboveAfterCloningOps
    * isAlwaysFusedIntoDispatchOp
    * isDispatchableOp
    * isAlwaysClonedIntoDispatchOp
    * orderOperations
* result
```mlir
    func @main_dispatch_16(
      %arg0: !flow.dispatch.tensor<readonly:10xf32>, 
      %arg1: !flow.dispatch.tensor<readonly:1x40xf32>, 
      %arg2: !flow.dispatch.tensor<writeonly:10xf32>) {
        %c0 = constant 0 : index
        %c20 = constant 20 : index
        %c1 = constant 1 : index
        %c10 = constant 10 : index
        %cst = constant 5.000000e-01 : f32
        %cst_0 = constant 1.000000e+01 : f32
        %cst_1 = constant -1.000000e+01 : f32
        %0 = flow.dispatch.tensor.load %arg1, offsets = [], sizes = [], strides = [] : 
          !flow.dispatch.tensor<readonly:1x40xf32> -> tensor<1x40xf32>
        %1 = flow.tensor.slice %0[%c0, %c20 for %c1, %c10] : tensor<1x40xf32> -> tensor<1x10xf32>
        %2 = flow.tensor.slice %0[%c0, %c10 for %c1, %c10] : tensor<1x40xf32> -> tensor<1x10xf32>
        %3 = flow.tensor.slice %0[%c0, %c0 for %c1, %c10] : tensor<1x40xf32> -> tensor<1x10xf32>
        %4 = flow.tensor.reshape %1 : tensor<1x10xf32> -> tensor<10xf32>
        %5 = flow.tensor.reshape %2 : tensor<1x10xf32> -> tensor<10xf32>
        %6 = flow.tensor.reshape %3 : tensor<1x10xf32> -> tensor<10xf32>
```
