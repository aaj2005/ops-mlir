#ifndef LAPLACE_KERNELS_H
#define LAPLACE_KERNELS_H

//===----------------------------------------------------------------------===//
// Laplace 2D Stencil Kernels
//===----------------------------------------------------------------------===//

/// Zero-out kernel
template <typename ACC>
void set_zero(ACC &A) {
  A(0, 0) = 0.0;
}

/// Left boundary condition
template <typename ACC>
void left_bndcon(ACC &A, int *idx) {
  A(0, 0) = sin(M_PI * idx[1]);
}

/// Right boundary condition
template <typename ACC>
void right_bndcon(ACC &A, int *idx) {
  A(0, 0) = sin(M_PI * idx[1]) * exp(-M_PI);
}

/// Apply 5-point Laplacian stencil with reduction
template <typename ACC_CONST, typename ACC_WRITE, typename ACC_REDUCE>
void apply_stencil(ACC_CONST A, ACC_WRITE Anew, ACC_REDUCE &err_max) {
  double del2 = (-4.0 * A(0, 0) + A(1, 0) + A(-1, 0) +
                  A(0, 1) + A(0, -1)) / 6.0;
  Anew(0, 0) = A(0, 0) + del2;
  double err = fabs(del2);
  err_max() = fmax(err_max(), err);
}

/// Copy kernel
template <typename ACC_READ, typename ACC_WRITE>
void copy(ACC_READ Anew, ACC_WRITE A) {
  A(0, 0) = Anew(0, 0);
}

#endif // LAPLACE_KERNELS_H
