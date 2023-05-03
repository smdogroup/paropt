#ifndef PAR_OPT_SCALED_QUASI_NEWTON_H
#define PAR_OPT_SCALED_QUASI_NEWTON_H

#include "ParOptComplexStep.h"
#include "ParOptQuasiNewton.h"

/*
This class is only used when there is a single constraint and objective is
linear.

In this case:
  L = f - z*c
  B ~= Hess = -z*Hess(c)
And we define
  B0 ~= -Hess(c)
such that
  B = z*B0

This will give a better approximation to the Hessian of the Lagrangian with B
if we use the quasi-Newton update correction.
*/
class ParOptScaledQuasiNewton : public ParOptCompactQuasiNewton {
 public:
  ParOptScaledQuasiNewton(ParOptProblem *_prob, ParOptCompactQuasiNewton *_qn) {
    int rank;
    MPI_Comm_rank(_prob->getMPIComm(), &rank);
    if (rank == 0) {
      fprintf(stdout,
              "[ParOptScaledQuasiNewton.h] initializing "
              "ParOptScaledQuasiNewton!\n");
    }
    qn = _qn;
    qn->incref();

    d0 = new ParOptScalar[qn->getMaxLimitedMemorySize()];
    y0 = _prob->createDesignVec();
    y0->incref();
  }

  ~ParOptScaledQuasiNewton() {
    delete[] d0;
    qn->decref();
    y0->decref();
  }

  // Set the type of diagonal to use
  void setInitDiagonalType(ParOptQuasiNewtonDiagonalType _diagonal_type) {
    qn->setInitDiagonalType(_diagonal_type);
  }

  // Reset the internal data
  void reset() { qn->reset(); }

  // Perform the quasi-Newton update with the specified multipliers
  int update(ParOptVec *x, const ParOptScalar *z, ParOptVec *zw, ParOptVec *s,
             ParOptVec *y) {
    z0 = z[0];

    // This should never happen
    if (ParOptRealPart(z0) < 0.0) {
      z0 = 0.0;
    }

    y0->copyValues(y);
    y0->scale(1.0 / z0);
    return qn->update(x, z, zw, s, y0);
  }

  // Perform a matrix-vector multiplication
  void mult(ParOptVec *x, ParOptVec *y) {
    qn->mult(x, y);
    y->scale(z0);
  }

  // Perform a matrix-vector multiplication and add the result to y
  void multAdd(ParOptScalar alpha, ParOptVec *x, ParOptVec *y) {
    qn->multAdd(alpha * z0, x, y);
  }

  // Get the compact representation for the limited-memory quasi-Newton method
  int getCompactMat(ParOptScalar *_b0, const ParOptScalar **_d,
                    const ParOptScalar **_M, ParOptVec ***Z) {
    const ParOptScalar *d;
    ParOptScalar b0;
    int m;
    m = qn->getCompactMat(&b0, &d, _M, Z);
    *_b0 = z0 * b0;
    for (int i = 0; i < m; i++) {
      d0[i] = sqrt(z0) * d[i];
    }
    *_d = d0;
    return m;
  }

  // Get the maximum size of the compact representation
  int getMaxLimitedMemorySize() { return qn->getMaxLimitedMemorySize(); }

 private:
  ParOptCompactQuasiNewton *qn;
  ParOptScalar z0;
  ParOptVec *y0;
  ParOptScalar *d0;
};

#endif  // PAR_OPT_SCALED_QUASI_NEWTON_H