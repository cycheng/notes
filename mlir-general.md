Contents:
=========
* Base class
* HowTo:
  * Dump region (function) / block in gdb

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
