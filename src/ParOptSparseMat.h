#ifndef PAR_OPT_SPARSE_MAT_H
#define PAR_OPT_SPARSE_MAT_H

/*
  Forward declare the matrices
*/
class ParOptQuasiDefMat;
class ParOptQuasiDefSparseMat;

#include "ParOptBlasLapack.h"
#include "ParOptProblem.h"
#include "ParOptVec.h"

/*
  Abstract base class for the quasi-definite matrix
*/
class ParOptQuasiDefMat : public ParOptBase {
 public:
  virtual ~ParOptQuasiDefMat() {}

  /*
    Factor the matrix
  */
  virtual int factor(ParOptVec *x, ParOptVec *Dinv, ParOptVec *Cdiag) = 0;

  /**
    Solve the quasi-definite system of equations

    [ D   Aw^{T} ][  yx ] = [ bx ]
    [ Aw    - C  ][ -yw ] = [ 0  ]

    Here bx is unmodified. Note the negative sign on the yw variables.

    @param bx the design variable right-hand-side
    @param yx the design variable solution
    @param yw the sparse multiplier solution
   */
  virtual void apply(ParOptVec *bx, ParOptVec *yx, ParOptVec *yw) = 0;

  /**
    Solve the quasi-definite system of equations

    [ D   Aw^{T} ][  yx ] = [ bx ]
    [ Aw    - C  ][ -yw ] = [ bw ]

    In the call bx and bw must remain unmodified. Note the negative sign on the
    yw variables.

    @param bx the design variable right-hand-side
    @param bx the sparse multiplier right-hand-side
    @param yx the design variable solution
    @param yw the sparse multiplier solution
   */
  virtual void apply(ParOptVec *bx, ParOptVec *bw, ParOptVec *yx,
                     ParOptVec *yw) = 0;
};

/*
  Interface for the quasi-definite matrix

  [ D   Aw^{T} ]
  [ Aw   -C    ]

  The goal of this interface is to provide access to the factorization of the
  matrix without explicitly dictating how the constraints are stored.
*/
class ParOptQuasiDefBlockMat : public ParOptQuasiDefMat {
 public:
  ParOptQuasiDefBlockMat(ParOptProblem *prob0, int _nwblock);
  ~ParOptQuasiDefBlockMat();

  /**
    Factor the matrix

    @param x The design variables
    @param Dinv The diagonal inverse of the D matrix (size of dvs)
    @param C The diagonal for multipliers (size of sparse multipliers)
   */
  int factor(ParOptVec *x, ParOptVec *Dinv, ParOptVec *C);

  /**
    Solve the quasi-definite system of equations

    [ D   Aw^{T} ][  yx ] = [ bx ]
    [ Aw    - C  ][ -yw ] = [ 0  ]

    Here bx is unmodified. Note the negative sign on the yw variables.

    @param bx Design variable right-hand-side (not modified)
    @param yx Design variable solution output
    @param yw Multiplier variable solution output
  */
  void apply(ParOptVec *bx, ParOptVec *yx, ParOptVec *yw);

  /**
    Solve the quasi-definite system of equations

    [ D   Aw^{T} ][  yx ] = [ bx ]
    [ Aw    - C  ][ -yw ] = [ bw ]

    In the call bx and bw must remain unmodified. Note the negative sign on the
    yw variables.

    @param bx Design variable right-hand-side (not modified)
    @param bw Multiplier right-hand-side (not modified)
    @param yx Design variable solution output
    @param yw Multiplier variable solution output
  */
  void apply(ParOptVec *bx, ParOptVec *bw, ParOptVec *yx, ParOptVec *yw);

 private:
  /*
    Apply the factored Cw-matrix that is stored as a series of block-symmetric
    matrices.
  */
  int applyFactor(ParOptVec *vec);

  // Problem data
  ParOptProblem *prob;

  // Vectors that point to the input data
  ParOptVec *x, *Dinv, *C;

  // The data for the block-diagonal matrix
  int nvars;         // The number of variables
  int nwcon;         // The number of sparse constraints
  int nwblock;       // The nuber of constraints per block
  ParOptScalar *Cw;  // Block diagonal matrix
};

/*
  Interface for a generic sparse quasi-definite matrix
*/
class ParOptQuasiDefSparseMat : public ParOptQuasiDefMat {
 public:
  ParOptQuasiDefSparseMat(ParOptSparseProblem *problem);
  ~ParOptQuasiDefSparseMat();

  int factor(ParOptVec *x, ParOptVec *Dinv, ParOptVec *C);
  void apply(ParOptVec *bx, ParOptVec *yx, ParOptVec *yw);
  void apply(ParOptVec *bx, ParOptVec *bw, ParOptVec *yx, ParOptVec *yw);

 private:
  void solve(ParOptScalar *b);

  // Return the index for the matrix
  inline int index(int i, int j) { return j + i * (i + 1) / 2; }

  ParOptSparseProblem *prob;

  int nvars, nwcon;
  int size;
  ParOptScalar *diag;
  ParOptScalar *L;
  ParOptScalar *rhs;
};

#endif  //  PAR_OPT_SPARSE_MAT_H