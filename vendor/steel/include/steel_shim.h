#pragma once
// Minimal shims for steel_gemm deps that only appear in code paths zdraw never
// exercises: complex GEMM (we use half) and the batched path (has_batch=false).
// Defined just enough to compile; never executed.
struct complex64_t {
  float real;
  float imag;
  complex64_t() thread {}
  complex64_t(float r, float i) thread : real(r), imag(i) {}
};
inline int64_t elem_to_loc(
    uint elem,
    constant const int* shape,
    constant const int64_t* strides,
    int ndim) {
  int64_t loc = 0;
  for (int i = ndim - 1; i >= 0; --i) {
    loc += int64_t(elem % uint(shape[i])) * strides[i];
    elem /= uint(shape[i]);
  }
  return loc;
}
