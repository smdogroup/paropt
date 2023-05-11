#ifndef PAR_OPT_SPARSE_MAT_H
#define PAR_OPT_SPARSE_MAT_H

#include "ParOptBlasLapack.h"
#include "ParOptProblem.h"
#include "ParOptVec.h"

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

#endif  //  PAR_OPT_SPARSE_MAT_H