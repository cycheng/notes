
* How to print vector data
```c
half4 r = (half4)(0.0f);
printf("%v4f\n", r);
```

* Basic Concepts: OpenCL Convenience Methods for Vector Elements and Type Conversions
  https://streamhpc.com/blog/2011-10-18/basic-concepts-convenience-methods/ 
```c
Convenience
alternative    vector2    vector3    vector4    vector8    vector16
      .x           .s0        .s0        .s0        N.D.       N.D.
      .y           .s1        .s1        .s1        N.D.       N.D.
      .z           .s2        .s2        .s2        N.D.       N.D.
      .w           .s3        N.D.       .s3        N.D.       N.D.
      .hi          .s1         ??       .s23      .s4567 .s89ABCDEF
      .lo          .s0         ??       .s01      .s0123 .s01234567
      .even        .s0         ??       .s02      .s0246 .s02468ACE
      .odd         .s1         ??       .s13      .s1357 .s13579BDF
```
