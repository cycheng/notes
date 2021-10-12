
Working on https://github.com/google/iree/issues/6903

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
        size_t id = reshapeIdx[iter.getFirst()][idx++];
        block.getArgument(id).replaceAllUsesWith(
            clonedOrigProducer->getResult(0));

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

* 
