#include "ParOptQuasiNewton.h"

#include <string.h>

#include "ParOptBlasLapack.h"
#include "ParOptComplexStep.h"

/**
  The following class implements the limited-memory BFGS update.

  The limited-memory BFGS formula takes the following form:

  b0*I - Z*diag{d)*M^{-1}*diag{d}*Z^{T}

  @param prob the ParOptProblem class
  @param msub_max the maximum subspace size
*/
ParOptLBFGS::ParOptLBFGS(ParOptProblem *prob, int _msub_max) {
  msub_max = _msub_max;
  msub = 0;

  b0 = 1.0;

  // Set the default Hessian update
  hessian_update_type = PAROPT_SKIP_NEGATIVE_CURVATURE;
  diagonal_type = PAROPT_YTY_OVER_YTS;
  epsilon_precision = 1e-12;

  // Allocate space for the vectors
  S = new ParOptVec *[msub_max];
  Y = new ParOptVec *[msub_max];
  Z = new ParOptVec *[2 * msub_max];

  for (int i = 0; i < msub_max; i++) {
    S[i] = prob->createDesignVec();
    S[i]->incref();
    Y[i] = prob->createDesignVec();
    Y[i]->incref();
  }

  // A temporary vector for the damped update
  r = prob->createDesignVec();
  r->incref();

  // The full M-matrix
  M = new ParOptScalar[4 * msub_max * msub_max];

  // The diagonal scaling matrix
  d0 = new ParOptScalar[2 * msub_max];

  // The factored M-matrix
  M_factor = new ParOptScalar[4 * msub_max * msub_max];
  mfpiv = new int[2 * msub_max];

  // Temporary vector required for multiplications
  rz = new ParOptScalar[2 * msub_max];

  // The components of the M matrix that must be
  // updated each iteration
  D = new ParOptScalar[msub_max];
  L = new ParOptScalar[msub_max * msub_max];
  B = new ParOptScalar[msub_max * msub_max];

  // Zero the initial values of everything
  memset(d0, 0, 2 * msub_max * sizeof(ParOptScalar));
  memset(rz, 0, 2 * msub_max * sizeof(ParOptScalar));

  memset(M, 0, 4 * msub_max * msub_max * sizeof(ParOptScalar));
  memset(M_factor, 0, 4 * msub_max * msub_max * sizeof(ParOptScalar));

  memset(D, 0, msub_max * sizeof(ParOptScalar));
  memset(L, 0, msub_max * msub_max * sizeof(ParOptScalar));
  memset(B, 0, msub_max * msub_max * sizeof(ParOptScalar));
}

/**
  Free the memory allocated by the BFGS update
*/
ParOptLBFGS::~ParOptLBFGS() {
  // Delete the vectors
  for (int i = 0; i < msub_max; i++) {
    Y[i]->decref();
    S[i]->decref();
  }
  r->decref();

  delete[] S;
  delete[] Y;
  delete[] Z;

  // Delete the matrices/data
  delete[] M;
  delete[] M_factor;
  delete[] mfpiv;
  delete[] rz;

  delete[] D;
  delete[] L;
  delete[] B;
  delete[] d0;
}

/**
  Set the quasi-Newton Hessian update type.

  @param _hessian_update_type the new quasi-Newton Hessian update type
*/
void ParOptLBFGS::setBFGSUpdateType(ParOptBFGSUpdateType _hessian_update_type) {
  hessian_update_type = _hessian_update_type;
}

/**
  Set the type of initial diagonal quasi-Newton Hessian approximation.

  @param _diagonal_type The type of initial quasi-Newton approximation
*/
void ParOptLBFGS::setInitDiagonalType(
    ParOptQuasiNewtonDiagonalType _diagonal_type) {
  diagonal_type = _diagonal_type;
}

/**
  Get the maximum size of the limited-memory BFGS update.

  @return the maximum limited memory size
*/
int ParOptLBFGS::getMaxLimitedMemorySize() { return 2 * msub_max; }

/**
  Reset the Hessian approximation
*/
void ParOptLBFGS::reset() {
  msub = 0;
  b0 = 1.0;

  // Zero the initial values of everything
  memset(d0, 0, 2 * msub_max * sizeof(ParOptScalar));
  memset(rz, 0, 2 * msub_max * sizeof(ParOptScalar));

  memset(M, 0, 4 * msub_max * msub_max * sizeof(ParOptScalar));
  memset(M_factor, 0, 4 * msub_max * msub_max * sizeof(ParOptScalar));

  memset(D, 0, msub_max * sizeof(ParOptScalar));
  memset(L, 0, msub_max * msub_max * sizeof(ParOptScalar));
  memset(B, 0, msub_max * msub_max * sizeof(ParOptScalar));
}

/**
  Compute the update to the limited-memory BFGS approximate Hessian.
  The BFGS formula takes the form:

  B*x = (b0*I - Z*diag{d}*M^{-1}*diag{d}*Z^{T})*x

  This code computes a damped update to ensure that the curvature
  condition is satisfied.

  @param s the step in the design variable values
  @param y the difference in the gradient of the Lagrangian

  @return update type: 0 = normal, 1 = damped update, 2 = skipped update
*/
int ParOptLBFGS::update(ParOptVec *x, const ParOptScalar *z, ParOptVec *zw,
                        ParOptVec *s, ParOptVec *y) {
  int update_type = 0;

  // Set the pointer for the new value of y
  ParOptVec *new_y = NULL;

  // Compute dot products that are required for the matrix
  // updating scheme
  ParOptScalar yTy = y->dot(y);
  ParOptScalar yTs = y->dot(s);
  ParOptScalar sTs = s->dot(s);

  if (hessian_update_type == PAROPT_SKIP_NEGATIVE_CURVATURE) {
    // Check if we should skip the update
    if (ParOptRealPart(sTs) <= epsilon_precision * epsilon_precision) {
      update_type = 2;
      return update_type;
    } else if (ParOptRealPart(yTs) <=
               epsilon_precision * sqrt(ParOptRealPart(yTy * sTs))) {
      update_type = 2;
      return update_type;
    }

    // Compute the scalar parameter
    if (diagonal_type == PAROPT_YTS_OVER_STS) {
      b0 = yTs / sTs;
    } else {
      b0 = yTy / yTs;
    }

    // Set the pointer to the new y value
    new_y = y;
  } else if (hessian_update_type == PAROPT_DAMPED_UPDATE) {
    // If the Hessian approximation has not been initialized,
    // guess an initial value for the b0 value
    if (msub == 0) {
      b0 = yTy / yTs;
      if (ParOptRealPart(b0) <= 0.0) {
        b0 = 1.0;
      }
    }

    // Set the new value of y (this pointer may be reset to r if a
    // damped update is required)
    new_y = y;

    // Compute the step times the old Hessian approximation
    // and store the result in the r vector
    mult(s, r);

    // Compute s^{T}*B*s
    ParOptScalar sTBs = r->dot(s);
    if (ParOptRealPart(yTs) <= 0.2 * ParOptRealPart(sTBs)) {
      update_type = 1;

      // Compute the value of theta
      ParOptScalar theta = 0.8 * sTBs / (sTBs - yTs);

      // Compute r = theta*y + (1 - theta)*B*s
      r->scale(1.0 - theta);
      r->axpy(theta, y);

      new_y = r;
      yTy = new_y->dot(new_y);
      yTs = s->dot(new_y);
    }

    // Set the new value of b0
    if (diagonal_type == PAROPT_YTS_OVER_STS) {
      b0 = yTs / sTs;
    } else {
      b0 = yTy / yTs;
    }
  }

  if (b0 < epsilon_precision) {
    reset();  // Nothing to do but reset
    update_type = 3;
    return update_type;
  }

  // Set up the new values
  if (msub < msub_max) {
    S[msub]->copyValues(s);
    Y[msub]->copyValues(new_y);
    msub++;
  } else if (msub == msub_max && msub_max > 0) {
    // Shift the pointers to the vectors so that everything
    // will work out
    S[0]->copyValues(s);
    Y[0]->copyValues(new_y);

    // Shift the pointers
    ParOptVec *stemp = S[0];
    ParOptVec *ytemp = Y[0];
    for (int i = 0; i < msub - 1; i++) {
      S[i] = S[i + 1];
      Y[i] = Y[i + 1];
    }
    S[msub - 1] = stemp;
    Y[msub - 1] = ytemp;

    // Now, shift the values in the matrices
    for (int i = 0; i < msub - 1; i++) {
      D[i] = D[i + 1];
    }

    for (int i = 0; i < msub - 1; i++) {
      for (int j = 0; j < msub - 1; j++) {
        B[i + j * msub_max] = B[i + 1 + (j + 1) * msub_max];
      }
    }

    for (int i = 0; i < msub - 1; i++) {
      for (int j = 0; j < i; j++) {
        L[i + j * msub_max] = L[i + 1 + (j + 1) * msub_max];
      }
    }
  }

  // Update the matrices required for the limited-memory update.
  // Update the S^{T}S matrix:
  for (int i = 0; i < msub; i++) {
    B[msub - 1 + i * msub_max] = S[msub - 1]->dot(S[i]);
    B[i + (msub - 1) * msub_max] = B[msub - 1 + i * msub_max];
  }

  // Update the diagonal D-matrix
  if (msub > 0) {
    D[msub - 1] = S[msub - 1]->dot(Y[msub - 1]);
  }

  // By definition, we have the L matrix:
  // For j < i: L[i + j*msub_max] = S[i]->dot(Y[j]);
  for (int i = 0; i < msub - 1; i++) {
    L[msub - 1 + i * msub_max] = S[msub - 1]->dot(Y[i]);
  }

  // Set the values into the M-matrix
  memset(M, 0, 4 * msub * msub * sizeof(ParOptScalar));

  // Populate the result in the M-matrix
  for (int i = 0; i < msub; i++) {
    for (int j = 0; j < msub; j++) {
      M[i + 2 * msub * j] = b0 * B[i + msub_max * j];
    }
  }

  // Add the L-terms in the matrix
  for (int i = 0; i < msub; i++) {
    for (int j = 0; j < i; j++) {
      M[i + 2 * msub * (j + msub)] = L[i + msub_max * j];
      M[j + msub + 2 * msub * i] = L[i + msub_max * j];
    }
  }

  // Add the trailing diagonal term
  for (int i = 0; i < msub; i++) {
    M[msub + i + 2 * msub * (msub + i)] = -D[i];
  }

  // Copy over the new ordering for the Z-vectors
  for (int i = 0; i < msub; i++) {
    // Set the vector ordering
    Z[i] = S[i];
    Z[i + msub] = Y[i];

    // Set the values of the diagonal vector b0
    d0[i] = b0;
    d0[i + msub] = 1.0;
  }

  // Copy out the M matrix for factorization
  memcpy(M_factor, M, 4 * msub * msub * sizeof(ParOptScalar));

  // Factor the M matrix for later useage
  if (msub > 0) {
    int n = 2 * msub, info = 0;
    LAPACKdgetrf(&n, &n, M_factor, &n, mfpiv, &info);
  }

  return update_type;
}

/**
  Given the input vector, multiply the BFGS approximation by the input
  vector

  This code computes the product of the ParOptLBFGS matrix with the vector x:

  y <- b0*x - Z*diag{d}*M^{-1}*diag{d}*Z^{T}*x

  @param x the input vector
  @param y the result of the matrix-vector product
*/
void ParOptLBFGS::mult(ParOptVec *x, ParOptVec *y) {
  // Set y = b0*x
  y->copyValues(x);
  y->scale(b0);

  if (msub > 0) {
    // Compute rz = Z^{T}*x
    x->mdot(Z, 2 * msub, rz);

    // Set rz *= d0
    for (int i = 0; i < 2 * msub; i++) {
      rz[i] *= d0[i];
    }

    // Solve rz = M^{-1}*rz
    int n = 2 * msub, one = 1, info = 0;
    LAPACKdgetrs("N", &n, &one, M_factor, &n, mfpiv, rz, &n, &info);

    // Compute rz *= d0
    for (int i = 0; i < 2 * msub; i++) {
      rz[i] *= d0[i];
    }

    // Now compute: y <- Z*rz
    for (int i = 0; i < 2 * msub; i++) {
      y->axpy(-rz[i], Z[i]);
    }
  }
}

/**
  Given the input vector, multiply the BFGS approximation by the input
  vector and add the result to the output vector

  This code computes the product of the LBFGS matrix with the vector x:

  y <- y + alpha*(b0*x - Z*diag{d}*M^{-1}*diag{d}*Z^{T}*x)

  @param alpha scalar multiplication factor
  @param x the input vector
  @param y the result of the matrix-vector product
*/
void ParOptLBFGS::multAdd(ParOptScalar alpha, ParOptVec *x, ParOptVec *y) {
  // Set y = b0*x
  y->axpy(b0 * alpha, x);

  if (msub > 0) {
    // Compute rz = Z^{T}*x
    x->mdot(Z, 2 * msub, rz);

    // Set rz *= d0
    for (int i = 0; i < 2 * msub; i++) {
      rz[i] *= d0[i];
    }

    // Solve rz = M^{-1}*rz
    int n = 2 * msub, one = 1, info = 0;
    LAPACKdgetrs("N", &n, &one, M_factor, &n, mfpiv, rz, &n, &info);

    // Compute rz *= d0
    for (int i = 0; i < 2 * msub; i++) {
      rz[i] *= d0[i];
    }

    // Now compute: y <- Z*rz
    for (int i = 0; i < 2 * msub; i++) {
      y->axpy(-alpha * rz[i], Z[i]);
    }
  }
}

/**
  Retrieve the internal data for the limited-memory BFGS
  representation

  @param _b0 the diagonal factor
  @param _d the diagonal matrix of scaling factors
  @param _M the small matrix in the compact form
  @param _Z an array of the vectors
  @return the size of the _Z and _M matrices
*/
int ParOptLBFGS::getCompactMat(ParOptScalar *_b0, const ParOptScalar **_d,
                               const ParOptScalar **_M, ParOptVec ***_Z) {
  if (_b0) {
    *_b0 = b0;
  }
  if (_d) {
    *_d = d0;
  }
  if (_M) {
    *_M = M;
  }
  if (_Z) {
    *_Z = Z;
  }

  return 2 * msub;
}

/**
  The following class implements the limited-memory SR1 update.

  The limited-memory SR1 formula takes the following form:

  b0*I - Z*diag{d)*M^{-1}*diag{d}*Z^{T}

  @param prob the ParOptProblem class
  @param msub_max the maximum subspace size
*/
ParOptLSR1::ParOptLSR1(ParOptProblem *prob, int _msub_max) {
  msub_max = _msub_max;
  msub = 0;

  b0 = 1.0;

  // Set the default initial diagonal QN approximation
  diagonal_type = PAROPT_YTY_OVER_YTS;

  // Allocate space for the vectors
  S = new ParOptVec *[msub_max];
  Y = new ParOptVec *[msub_max];
  Z = new ParOptVec *[msub_max];

  for (int i = 0; i < msub_max; i++) {
    S[i] = prob->createDesignVec();
    S[i]->incref();
    Y[i] = prob->createDesignVec();
    Y[i]->incref();
    Z[i] = prob->createDesignVec();
    Z[i]->incref();
  }

  // A temporary vector for the damped update
  r = prob->createDesignVec();
  r->incref();

  // The full M-matrix
  M = new ParOptScalar[msub_max * msub_max];

  // The diagonal scaling matrix
  d0 = new ParOptScalar[msub_max];

  // The factored M-matrix
  M_factor = new ParOptScalar[msub_max * msub_max];
  mfpiv = new int[msub_max];

  // Temporary vector required for multiplications
  rz = new ParOptScalar[msub_max];

  // The components of the M matrix that must be
  // updated each iteration
  D = new ParOptScalar[msub_max];
  L = new ParOptScalar[msub_max * msub_max];
  B = new ParOptScalar[msub_max * msub_max];

  // Zero the initial values of everything
  memset(d0, 0, msub_max * sizeof(ParOptScalar));
  memset(rz, 0, msub_max * sizeof(ParOptScalar));

  memset(M, 0, msub_max * msub_max * sizeof(ParOptScalar));
  memset(M_factor, 0, msub_max * msub_max * sizeof(ParOptScalar));

  memset(D, 0, msub_max * sizeof(ParOptScalar));
  memset(L, 0, msub_max * msub_max * sizeof(ParOptScalar));
  memset(B, 0, msub_max * msub_max * sizeof(ParOptScalar));

  // Set the orthogonality precision
  epsilon_precision = 1e-12;
}

/**
  Free the memory allocated by the BFGS update
*/
ParOptLSR1::~ParOptLSR1() {
  // Delete the vectors
  for (int i = 0; i < msub_max; i++) {
    Y[i]->decref();
    S[i]->decref();
    Z[i]->decref();
  }
  r->decref();

  delete[] S;
  delete[] Y;
  delete[] Z;

  // Delete the matrices/data
  delete[] M;
  delete[] M_factor;
  delete[] mfpiv;
  delete[] rz;

  delete[] D;
  delete[] L;
  delete[] B;
  delete[] d0;
}

/**
  Set the type of initial diagonal quasi-Newton Hessian approximation.

  @param _diagonal_type The type of initial quasi-Newton approximation
*/
void ParOptLSR1::setInitDiagonalType(
    ParOptQuasiNewtonDiagonalType _diagonal_type) {
  diagonal_type = _diagonal_type;
}

/**
  Get the maximum size of the limited-memory BFGS update

  @return the maximum size of the limited memory subspace
*/
int ParOptLSR1::getMaxLimitedMemorySize() { return msub_max; }

/**
  Reset the Hessian approximation
*/
void ParOptLSR1::reset() {
  msub = 0;
  b0 = 1.0;

  // Zero the initial values of everything
  memset(d0, 0, msub_max * sizeof(ParOptScalar));
  memset(rz, 0, msub_max * sizeof(ParOptScalar));

  memset(M, 0, msub_max * msub_max * sizeof(ParOptScalar));
  memset(M_factor, 0, msub_max * msub_max * sizeof(ParOptScalar));

  memset(D, 0, msub_max * sizeof(ParOptScalar));
  memset(L, 0, msub_max * msub_max * sizeof(ParOptScalar));
  memset(B, 0, msub_max * msub_max * sizeof(ParOptScalar));
}

/**
  Compute the update to the limited-memory SR1 approximate
  Hessian. The SR1 formula takes the form:

  B*x = (b0*I - Z*diag{d}*M^{-1}*diag{d}*Z^{T})*x

  Note that the

  @param s the step in the design variable values
  @param y the difference in the gradient
  @return update type: 0 = normal, 1 = damped update
*/
int ParOptLSR1::update(ParOptVec *x, const ParOptScalar *z, ParOptVec *zw,
                       ParOptVec *s, ParOptVec *y) {
  int update_type = 0;

  // Compute the dot-products needed for the update
  ParOptScalar yTy = y->dot(y);
  ParOptScalar sTy = s->dot(y);

  // Set the diagonal components to the identity matrix
  if (ParOptRealPart(sTy) > epsilon_precision * ParOptRealPart(yTy)) {
    b0 = yTy / sTy;
  } else {
    b0 = 1.0;
  }

  // Set up the new values
  if (msub < msub_max) {
    S[msub]->copyValues(s);
    Y[msub]->copyValues(y);
    msub++;
  } else if (msub == msub_max && msub_max > 0) {
    // Shift the pointers to the vectors so that everything
    // will work out
    S[0]->copyValues(s);
    Y[0]->copyValues(y);

    // Shift the pointers
    ParOptVec *stemp = S[0];
    ParOptVec *ytemp = Y[0];
    for (int i = 0; i < msub - 1; i++) {
      S[i] = S[i + 1];
      Y[i] = Y[i + 1];
    }
    S[msub - 1] = stemp;
    Y[msub - 1] = ytemp;

    // Now, shift the values in the matrices
    for (int i = 0; i < msub - 1; i++) {
      D[i] = D[i + 1];
    }

    for (int i = 0; i < msub - 1; i++) {
      for (int j = 0; j < msub - 1; j++) {
        B[i + j * msub_max] = B[i + 1 + (j + 1) * msub_max];
      }
    }

    for (int i = 0; i < msub - 1; i++) {
      for (int j = 0; j < i; j++) {
        L[i + j * msub_max] = L[i + 1 + (j + 1) * msub_max];
      }
    }
  }

  // Update the matrices required for the limited-memory update.
  // Update the S^{T}S matrix:
  for (int i = 0; i < msub; i++) {
    B[msub - 1 + i * msub_max] = S[msub - 1]->dot(S[i]);
    B[i + (msub - 1) * msub_max] = B[msub - 1 + i * msub_max];
  }

  // Update the diagonal D-matrix
  if (msub > 0) {
    D[msub - 1] = S[msub - 1]->dot(Y[msub - 1]);
  }

  // By definition, we have the L matrix:
  // For j < i: L[i + j*msub_max] = S[i]->dot(Y[j]);
  for (int i = 0; i < msub - 1; i++) {
    L[msub - 1 + i * msub_max] = S[msub - 1]->dot(Y[i]);
  }

  // Set the values into the M-matrix
  memset(M, 0, msub * msub * sizeof(ParOptScalar));

  // Populate the result in the M-matrix
  for (int i = 0; i < msub; i++) {
    for (int j = 0; j < msub; j++) {
      M[i + msub * j] += b0 * B[i + msub_max * j];
    }
  }

  for (int i = 0; i < msub; i++) {
    for (int j = 0; j < i; j++) {
      M[i + msub * j] -= L[i + msub_max * j];
      M[j + msub * i] -= L[i + msub_max * j];
    }
  }

  for (int i = 0; i < msub; i++) {
    M[i * (msub + 1)] -= D[i];
  }

  // Set the new values of the Z-vectors
  for (int i = 0; i < msub; i++) {
    Z[i]->copyValues(Y[i]);
    Z[i]->axpy(-b0, S[i]);

    d0[i] = 1.0;
  }

  // Copy out the M matrix for factorization
  memcpy(M_factor, M, msub * msub * sizeof(ParOptScalar));

  // Factor the M matrix for later useage
  if (msub > 0) {
    int n = msub, info = 0;
    LAPACKdgetrf(&n, &n, M_factor, &n, mfpiv, &info);
  }

  return update_type;
}

/**
  Given the input vector, multiply the SR1 approximation by the input
  vector

  This code computes the product of the LSR1 matrix with the vector x:

  y <- b0*x - Z*M^{-1}*Z^{T}*x

  @param x the input vector
  @param y the result of the matrix-vector product
*/
void ParOptLSR1::mult(ParOptVec *x, ParOptVec *y) {
  // Set y = b0*x
  y->copyValues(x);
  y->scale(b0);

  if (msub > 0) {
    // Compute rz = Z^{T}*x
    x->mdot(Z, msub, rz);

    // Solve rz = M^{-1}*rz
    int n = msub, one = 1, info = 0;
    LAPACKdgetrs("N", &n, &one, M_factor, &n, mfpiv, rz, &n, &info);

    // Now compute: y <- Z*rz
    for (int i = 0; i < msub; i++) {
      y->axpy(-rz[i], Z[i]);
    }
  }
}

/**
  Given the input vector, multiply the SR1 approximation by the input
  vector and add the result to the output vector

  This code computes the product of the LSR1 matrix with the vector x:

  y <- y + alpha*(b0*x - Z*M^{-1}*Z^{T}*x)

  @param alpha scalar multiplication factor
  @param x the input vector
  @param y the result of the matrix-vector product
*/
void ParOptLSR1::multAdd(ParOptScalar alpha, ParOptVec *x, ParOptVec *y) {
  // Set y = b0*x
  y->axpy(b0 * alpha, x);

  if (msub > 0) {
    // Compute rz = Z^{T}*x
    x->mdot(Z, msub, rz);

    // Solve rz = M^{-1}*rz
    int n = msub, one = 1, info = 0;
    LAPACKdgetrs("N", &n, &one, M_factor, &n, mfpiv, rz, &n, &info);

    // Now compute: y <- Z*rz
    for (int i = 0; i < msub; i++) {
      y->axpy(-alpha * rz[i], Z[i]);
    }
  }
}

/**
  Retrieve the internal data for the limited-memory BFGS
  representation

  @param _b0 the diagonal factor
  @param _d the diagonal matrix of scaling factors
  @param _M the small matrix in the compact form
  @param _Z an array of the vectors
  @return the size of the _Z and _M matrices
*/
int ParOptLSR1::getCompactMat(ParOptScalar *_b0, const ParOptScalar **_d,
                              const ParOptScalar **_M, ParOptVec ***_Z) {
  if (_b0) {
    *_b0 = b0;
  }
  if (_d) {
    *_d = d0;
  }
  if (_M) {
    *_M = M;
  }
  if (_Z) {
    *_Z = Z;
  }

  return msub;
}
