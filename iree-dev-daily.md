#### Nov 4, 5, 6
Working on https://github.com/google/iree/issues/7014

Implement conversion: vm -> emitc.
Reference code in IREE for this task:

* iree/compiler/Dialect/VM/Conversion/VMToEmitC/ConvertVMToEmitC.cpp
```cpp
  patterns.insert<ListAllocOpConversion>(typeConverter, context,
                                         vmAnalysisCache);
  patterns.insert<ListOpConversion<IREE::VM::ListReserveOp>>(
      context, "iree_vm_list_reserve", 0, true);
```

* test cmd
```shell
time cmake --build . -t iree-translate
/home/cycheng/build/iree/1.x86.rel_asrt_dbg/iree/tools/iree-translate -iree-vm-ir-to-c-module /home/cycheng/iree/iree/vm/test/emitc/../list_ops.mlir -o list_ops.h
```


Reference code in IREE for this issue:

* iree/runtime/demo/hello_world_explained.c
  ```cpp
    iree_runtime_instance_t* instance = NULL;
    iree_status_t status = iree_runtime_instance_create(
        &instance_options, iree_allocator_system(), &instance);
  ```

* iree/runtime/call.c
  ```cpp
    // Allocate the input and output lists with the required capacity.
    // A user wanting to avoid dynamic allocations could instead create on-stack
    // storage for these and use iree_vm_list_initialize instead. This high-level
    // API keeps things simple, though, and for the frequency of calls through
    // this interface a few small pooled malloc calls should be fine.
    iree_allocator_t host_allocator =
        iree_runtime_session_host_allocator(session);
    iree_status_t status = iree_vm_list_create(
        /*element_type=*/NULL, arguments.size, host_allocator, &out_call->inputs);
    if (iree_status_is_ok(status)) {
      status = iree_vm_list_create(
          /*element_type=*/NULL, results.size, host_allocator,
          &out_call->outputs);
    }
  ```

* iree/base/allocator.h
  ```cpp
  // Allocates using the iree_allocator_malloc and iree_allocator_free methods.
  // These will usually be backed by malloc and free.
  static inline iree_allocator_t iree_allocator_system(void) {
    iree_allocator_t v = {NULL, iree_allocator_system_ctl};
    return v;
  }
  ```

* iree/base/allocator.c
  ```cpp
    IREE_API_EXPORT iree_status_t
    iree_allocator_system_ctl(void* self, iree_allocator_command_t command,
                              const void* params, void** inout_ptr) {
      switch (command) {
        case IREE_ALLOCATOR_COMMAND_MALLOC:
        case IREE_ALLOCATOR_COMMAND_CALLOC:
        case IREE_ALLOCATOR_COMMAND_REALLOC:
          return iree_allocator_system_alloc(
              command, (const iree_allocator_alloc_params_t*)params, inout_ptr);
        case IREE_ALLOCATOR_COMMAND_FREE:
          return iree_allocator_system_free(inout_ptr);
        default:
          return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                  "unsupported system allocator command");
      }
    }
  ```

* iree/vm/list_test.cc
  ```cpp
  // Tests simple variant list usage, mainly just for demonstration.
  // Stores any heterogeneous element type, equivalent to `!vm.list<?>`.
  TEST_F(VMListTest, UsageVariant) {
    iree_vm_type_def_t element_type = iree_vm_type_def_make_variant_type();
    iree_host_size_t initial_capacity = 123;
    iree_vm_list_t* list = nullptr;
    IREE_ASSERT_OK(iree_vm_list_create(&element_type, initial_capacity,
                                       iree_allocator_system(), &list));
  ```

* iree/vm/value.h
  ``` cpp
  // Defines the type of a primitive value.
  typedef enum iree_vm_value_type_e {
    // Not a value type.
    IREE_VM_VALUE_TYPE_NONE = 0,
    // int8_t.
    IREE_VM_VALUE_TYPE_I8 = 1,
    // int16_t.
    IREE_VM_VALUE_TYPE_I16 = 2,
    // int32_t.
    IREE_VM_VALUE_TYPE_I32 = 3,
    // int64_t.
    IREE_VM_VALUE_TYPE_I64 = 4,
    // float.
    IREE_VM_VALUE_TYPE_F32 = 5,
    // double.
    IREE_VM_VALUE_TYPE_F64 = 6,

    IREE_VM_VALUE_TYPE_MAX = IREE_VM_VALUE_TYPE_F64,
    IREE_VM_VALUE_TYPE_COUNT = IREE_VM_VALUE_TYPE_MAX + 1,  // used for lookup
  } iree_vm_value_type_t;
  ```

* iree/vm/ref.h
  ```cpp
  // Defines the type of the reference-counted pointer.
  // This is used to verify that operations dealing with the variant ref struct
  // are correct at runtime. We don't allow control over the ref types from the
  // VM ops and as such we can use the type specified as a safe way to avoid
  // reinterpreting memory incorrectly.
  enum iree_vm_ref_type_bits_t {
    IREE_VM_REF_TYPE_NULL = 0,

    // NOTE: these type values are assigned dynamically right now. Treat them as
    // opaque and unstable across process invocations.

    // Maximum type ID value. Type IDs are limited to 24-bits.
    IREE_VM_REF_TYPE_MAX_VALUE = 0x00FFFFFEu,

    // Wildcard type that indicates that a value may be a ref type but of an
    // unspecified internal type.
    IREE_VM_REF_TYPE_ANY = 0x00FFFFFFu,
  };
  typedef uint32_t iree_vm_ref_type_t;
  ```
* iree/vm/type_def.h
  ```cpp
  // Describes a type in the type table, mapping from a local module type ID to
  // either a primitive value type or registered ref type.
  //
  // * ?: variant (value_type/ref_type == 0)
  // * i8: primitive value (value_type != 0)
  // * !vm.ref<?>: any ref value (ref_type == IREE_VM_REF_TYPE_ANY)
  // * !vm.ref<!foo>: ref value of type !foo (ref_type > 0)
  typedef struct iree_vm_type_def_t {
    iree_vm_value_type_t value_type : 8;
    iree_vm_ref_type_t ref_type : 24;
  } iree_vm_type_def_t;
  ```

* iree/vm/ref.c
  ```cpp
  IREE_API_EXPORT void iree_vm_ref_move(iree_vm_ref_t* ref,
                                        iree_vm_ref_t* out_ref) {
    // NOTE: ref and out_ref may alias.
    if (ref == out_ref) {
      // Source == target; ignore entirely.
      return;
    }

    // Reset input ref so it points at nothing.
    iree_vm_ref_t temp_ref = *ref;
    memset(ref, 0, sizeof(*ref));

    if (out_ref->ptr != NULL) {
      // Release existing value.
      iree_vm_ref_release(out_ref);
    }

    // Assign ref to out_ref (without incrementing counter).
    *out_ref = temp_ref;
  }
  ```

* iree/vm/list.c
  ```cpp
    iree_vm_ref_object_t ref_object;
    iree_allocator_t allocator;

    // Current capacity of the list storage, in elements.
    iree_host_size_t capacity;
    // Current count of elements in the list.
    iree_host_size_t count;

    // Element type stored within the list.
    iree_vm_type_def_t element_type;
    // Size of each element in the storage in bytes.
    iree_host_size_t element_size;

    // Storage mode defining how the storage array is managed.
    iree_vm_list_storage_mode_t storage_mode;
    // A flat dense array of elements in the type defined by storage_mode.
    // For certain storage modes, such as IREE_VM_STORAGE_MODE_REF, special
    // lifetime management and cleanup logic is required.
    void* storage;
  ```

#### Nov 1
Working on https://github.com/google/iree/issues/7014

* iree/compiler/Dialect/VM/IR/VMOps.td
  ```tablegen
  def VM_ListAllocOp :
      VM_PureOp<"list.alloc", [
        DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
        MemoryEffects<[MemAlloc]>,
      ]> {

    let encoding = [
      VM_EncOpcode<VM_OPC_ListAlloc>,
  ..

  def VM_ListResizeOp :
      VM_Op<"list.resize", [
        DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
        MemoryEffects<[MemWrite]>,
      ]> {
    let encoding = [
      VM_EncOpcode<VM_OPC_ListResize>,
  ```
* iree/compiler/Dialect/VM/IR/VMOpcodesCore.td
  ```tablegen
  def VM_OPC_ListAlloc             : VM_OPC<0x10, "ListAlloc">;
  def VM_OPC_ListResize            : VM_OPC<0x13, "ListResize">;
  ```
* iree/compiler/Dialect/VM/Conversion/VMToEmitC/ConvertVMToEmitC.cpp
  ```cpp
    patterns.insert<ListAllocOpConversion>(typeConverter, context,
                                           vmAnalysisCache);

    patterns.insert<ListOpConversion<IREE::VM::ListResizeOp>>(
        context, "iree_vm_list_resize", 0, true);
  ```


#### Oct 27
Working on https://reviews.llvm.org/D112110

#### Oct 26
Working on https://github.com/google/iree/issues/7014


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

#### Oct 24
* Create tensor.extract_slice with reduced rank in dispatchOp
* works!!
  ```mlir
      func @main_dispatch_13(%arg0: !flow.dispatch.tensor<readonly:10xf32>, 
                             %arg1: !flow.dispatch.tensor<readonly:1x40xf32>, 
                             %arg2: !flow.dispatch.tensor<writeonly:10xf32>) {
        ..
        %0 = flow.dispatch.tensor.load %arg1, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:1x40xf32> -> tensor<1x40xf32>
        %1 = tensor.extract_slice %0[0, 20] [1, 10] [1, 1] : tensor<1x40xf32> to tensor<10xf32>
        %2 = tensor.extract_slice %0[0, 10] [1, 10] [1, 1] : tensor<1x40xf32> to tensor<10xf32>
        %3 = tensor.extract_slice %0[0, 0] [1, 10] [1, 1] : tensor<1x40xf32> to tensor<10xf32>
        scf.for %arg3 = %4 to %c10 step %5 {
          %6 = affine.min #map10(%workgroup_size_0, %arg3)
          %7 = tensor.extract_slice %1[%arg3] [%6] [1] : tensor<10xf32> to tensor<?xf32>
          ..
          %11 = tensor.extract_slice %2[%arg3] [%10] [1] : tensor<10xf32> to tensor<?xf32>
          %13 = tensor.extract_slice %3[%arg3] [%12] [1] : tensor<10xf32> to tensor<?xf32>

  ```
* patch
  ```cpp
    ImplicitLocOpBuilder b(dispatchOp.getLoc(), rewriter);
    // convert from flow.tensor.slice to (mlir) tensor.extract_slice, the stride is always one
    Value one = b.create<arith::ConstantIndexOp>(1);
    
    for (auto iter : sliceGroups) {
      for (auto sliceOp : iter.getSecond()) {
        SmallVector<Value, 4> strides(sliceOp.lengths().size(), one);
        Operation *reshapeOp = *sliceOp.getResult().getUsers().begin();

        auto tslice = b.create<tensor::ExtractSliceOp>(
            reshapeOp->getResult(0).getType().cast<RankedTensorType>(),
            sliceOp.source(), sliceOp.start_indices(), sliceOp.lengths(),
            strides);

        size_t id = argIdx[iter.getFirst()][idx++];
        block.getArgument(id).replaceAllUsesWith(tslice);
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
