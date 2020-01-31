#include "ParOptProblem.h"
#include "ParOptComplexStep.h"
#include <cstring>
void ParOptProblem::checkGradients( double dh, ParOptVec *xvec,
                                    int check_hvec_product ){
  ParOptVec *x = xvec;
  if (!xvec){
    x = createDesignVec();
    x->incref();
  }

  // Create the perturbation vector
  ParOptVec *px = createDesignVec();
  px->incref();

  // Allocate space for constraints
  ParOptScalar fobj = 0.0;
  ParOptScalar *c = new ParOptScalar[ ncon ];

  // Allocate objective gradient/constraint gradient
  ParOptVec *g = createDesignVec();
  g->incref();

  ParOptVec **Ac = new ParOptVec*[ ncon ];
  for ( int i = 0; i < ncon; i++ ){
    Ac[i] = createDesignVec();
    Ac[i]->incref();
  }

  // Get the design variable values. Note that the values in g and px are
  // discarded.
  if (!xvec){
    getVarsAndBounds(x, g, px);
  }

  // Evaluate the objective/constraint and gradients
  evalObjCon(x, &fobj, c);
  evalObjConGradient(x, g, Ac);

  ParOptScalar *pxvals, *gvals;
  px->getArray(&pxvals);
  g->getArray(&gvals);
  for ( int i = 0; i < nvars; i++ ){
    if (ParOptRealPart(gvals[i]) >= 0.0){
      pxvals[i] = 1.0;
    }
    else {
      pxvals[i] = -1.0;
    }
  }

  // Compute the projected derivative of the objective
  ParOptScalar pobj = g->dot(px);

  // Compute the projected derivative of the constraints
  ParOptScalar *Apx = new ParOptScalar[ ncon ];
  px->mdot(Ac, ncon, Apx);

  // Set the step direction in the sparse Lagrange multipliers
  // to an initial vector
  ParOptVec *zw = NULL;
  if (nwcon > 0){
    zw = createConstraintVec();
    zw->incref();

    // Set a value for the zw array
    ParOptScalar *zwvals;
    zw->getArray(&zwvals);
    for ( int i = 0; i < nwcon; i++ ){
      zwvals[i] = 1.05 + 0.25*(i % 21);
    }
  }

  // Evaluate the Hessian-vector product
  ParOptScalar *ztemp = NULL;
  ParOptVec *hvec = NULL, *htemp = NULL;

  if (check_hvec_product){
    ztemp = new ParOptScalar[ ncon ];
    for ( int i = 0; i < ncon; i++ ){
      ztemp[i] = 2.3 - 0.15*(i % 5);
    }

    // Add the contribution to gradient of the Lagrangian
    // from the sparse constraints
    addSparseJacobianTranspose(-1.0, x, zw, g);

    for ( int i = 0; i < ncon; i++ ){
      g->axpy(-ztemp[i], Ac[i]);
    }

    // Evaluate the Hessian-vector product
    hvec = createDesignVec();
    hvec->incref();
    evalHvecProduct(x, ztemp, zw, px, hvec);

    // Check that multiple calls to the Hvec code
    // produce the same result
    htemp = createDesignVec();
    htemp->incref();
    evalHvecProduct(x, ztemp, zw, px, htemp);

    int rank;
    MPI_Comm_rank(comm, &rank);
    htemp->axpy(-1.0, hvec);
    double diff_nrm = htemp->norm();

    if (rank == 0){
      printf("Hvec code reproducibility test\n");
      printf("Difference between multiple calls: %15.8e\n\n", diff_nrm);
    }
  }

  // Compute the point xt = x + dh*px
  ParOptVec *xt = createDesignVec();
  xt->incref();

  // Copy values into the constraint
  xt->copyValues(x);

#ifdef PAROPT_USE_COMPLEX
  xt->axpy(ParOptScalar(0.0, dh), px);
#else
  xt->axpy(dh, px);
#endif // PAROPT_USE_COMPLEX

  // Compute the finite-difference product
  ParOptScalar fobj2;
  ParOptScalar *ctemp = new ParOptScalar[ ncon ];
  evalObjCon(xt, &fobj2, ctemp);

#ifdef PAROPT_USE_COMPLEX
  ParOptScalar pfd = ParOptImagPart(fobj2)/dh;
#else
  ParOptScalar pfd = (fobj2 - fobj)/dh;
#endif // PAROPT_USE_COMPLEX

  // Print out the results on the root processor
  int rank;
  MPI_Comm_rank(comm, &rank);

  if (rank == 0){
    printf("Objective gradient test\n");
    printf("Objective FD: %15.8e  Actual: %15.8e  Err: %8.2e  "
           "Rel err: %8.2e\n", ParOptRealPart(pfd), ParOptRealPart(pobj),
           fabs(ParOptRealPart(pobj - pfd)),
           fabs(ParOptRealPart((pobj - pfd)/pfd)));

    printf("\nConstraint gradient test\n");
    for ( int i = 0; i < ncon; i++ ){
#ifdef PAROPT_USE_COMPLEX
      ParOptScalar fd = ParOptImagPart(ctemp[i])/dh;
#else
      ParOptScalar fd = (ctemp[i] - c[i])/dh;
#endif // PAROPT_USE_COMPLEX

      printf("Con[%3d]  FD: %15.8e  Actual: %15.8e  Err: %8.2e  "
             "Rel err: %8.2e\n", i, ParOptRealPart(fd),
             ParOptRealPart(Apx[i]), fabs(ParOptRealPart(fd - Apx[i])),
             fabs(ParOptRealPart((fd - Apx[i])/fd)));
    }
  }

  if (check_hvec_product){
    // Evaluate the objective/constraints
    ParOptVec *g2 = createDesignVec();
    g2->incref();

    ParOptVec **Ac2 = new ParOptVec*[ ncon ];
    for ( int i = 0; i < ncon; i++ ){
      Ac2[i] = createDesignVec();
      Ac2[i]->incref();
    }

    // Evaluate the gradient at the perturbed point and add the
    // contribution from the sparse constraints to the Hessian
    evalObjConGradient(xt, g2, Ac2);
    addSparseJacobianTranspose(-1.0, xt, zw, g2);

    // Add the contribution from the dense constraints
    for ( int i = 0; i < ncon; i++ ){
      g2->axpy(-ztemp[i], Ac2[i]);
    }

#ifdef PAROPT_USE_COMPLEX
    // Evaluate the real part
    ParOptScalar *gvals;
    int gsize = g2->getArray(&gvals);

    for ( int i = 0; i < gsize; i++ ){
      gvals[i] = ParOptImagPart(gvals[i])/dh;
    }
#else
    // Compute the difference
    g2->axpy(-1.0, g);
    g2->scale(1.0/dh);
#endif // TACS_USE_COMPLEX

    // Compute the norm of the finite-difference approximation and
    // actual Hessian-vector products
    double fdnorm = g2->norm();
    double hnorm = hvec->norm();

    // Compute the max error between the two
    g2->axpy(-1.0, hvec);
    double herr = g2->norm();

    if (rank == 0){
      printf("\nHessian product test\n");
      printf("Objective FD: %15.8e  Actual: %15.8e  Err: %8.2e  "
             "Rel err: %8.2e\n", fdnorm, hnorm, herr, herr/hnorm);
    }

    // Clean up the allocated data
    g2->decref();
    for ( int i = 0; i < ncon; i++ ){
      Ac2[i]->decref();
    }
    delete [] Ac2;
  }

  // Now, perform a check of the sparse constraints (if any)
  ParOptVec *cw = NULL, *cwtemp = NULL;
  if (nwcon > 0){
    cw = createConstraintVec();
    cw->incref();

    // Allocate the constraint vectors
    cwtemp = createConstraintVec();
    cwtemp->incref();

    // Check that the Jacobian is the derivative of the constraints
    evalSparseCon(x, cw);

    xt->copyValues(x);
    xt->axpy(dh, px);
    evalSparseCon(xt, cwtemp);

    // Compute rcw = (cw(x + dh*px) - cw(x))/dh
    cwtemp->axpy(-1.0, cw);
    cwtemp->scale(1.0/dh);

    // Compute the Jacobian-vector product
    cw->zeroEntries();
    addSparseJacobian(1.0, x, px, cw);

    // Compute the difference between the vectors
    cwtemp->axpy(-1.0, cw);

    // Compute the relative difference
    double cw_error = cwtemp->maxabs();

    if (rank == 0){
      printf("\nSparse constraint checks\n");
      printf("||(cw(x + h*px) - cw(x))/h - J(x)*px||: %8.2e\n", cw_error);
    }

    // Check the that the matrix-multiplication and its transpose are
    // equivalent by computing the inner product with two vectors
    // from either side
    cw->zeroEntries();
    addSparseJacobian(1.0, x, px, cw);

    g->zeroEntries();
    addSparseJacobianTranspose(1.0, x, zw, g);

    ParOptScalar d1 = zw->dot(cw);
    ParOptScalar d2 = g->dot(px);

    if (rank == 0){
      printf("\nTranspose-equivalence\n");
      printf("x^{T}*(J(x)*p): %8.2e  p*(J(x)^{T}*x): %8.2e  "
             "Err: %8.2e  Rel Err: %8.2e\n", ParOptRealPart(d1),
             ParOptRealPart(d2), fabs(ParOptRealPart(d1 - d2)),
             fabs(ParOptRealPart((d1 - d2)/d2)));
    }

    ParOptVec *Cvec = createDesignVec();
    Cvec->incref();

    // Set Cvec to something more-or-less random
    ParOptScalar *cvals;
    Cvec->getArray(&cvals);
    for ( int i = 0; i < nvars; i++ ){
      cvals[i] = 0.05 + 0.25*(i % 37);
    }

    // Check the inner product zw^{T}*J(x)*cvec*J(x)^{T}*zw against the
    // matrix Cw
    ParOptScalar *Cw = new ParOptScalar[ nwcon*(nwblock+1)/2 ];
    memset(Cw, 0, nwcon*(nwblock+1)/2*sizeof(ParOptScalar));
    addSparseInnerProduct(1.0, x, Cvec, Cw);

    // Compute the vector product using the Jacobians
    px->zeroEntries();
    addSparseJacobianTranspose(1.0, x, zw, px);

    // Multiply component-wise
    for ( int i = 0; i < nvars; i++ ){
      pxvals[i] *= cvals[i];
    }
    cw->zeroEntries();
    addSparseJacobian(1.0, x, px, cw);
    d1 = cw->dot(zw);

    // Set the pointer into the Cw
    d2 = 0.0;
    ParOptScalar *cwvals = Cw;

    ParOptScalar *zwvals;
    zw->getArray(&zwvals);

    // Iterate over each block matrix
    for ( int i = 0; i < nwcon; i += nwblock ){
      // Index into each block
      for ( int j = 0; j < nwblock; j++ ){
        for ( int k = 0; k < j; k++ ){
          d2 += 2.0*cwvals[0]*zwvals[i+j]*zwvals[i+k];
          cwvals++;
        }

        d2 += cwvals[0]*zwvals[i+j]*zwvals[i+j];
        cwvals++;
      }
    }

    // Add the result across all processors
    ParOptScalar temp = d2;
    MPI_Reduce(&temp, &d2, 1, PAROPT_MPI_TYPE, MPI_SUM, 0, comm);

    if (rank == 0){
      printf("\nJ(x)*C^{-1}*J(x)^{T} test: \n");
      printf("Product: %8.2e  Matrix: %8.2e  Err: %8.2e  Rel Err: %8.2e\n",
             ParOptRealPart(d1), ParOptRealPart(d2),
             fabs(ParOptRealPart(d1 - d2)),
             fabs(ParOptRealPart((d1 - d2)/d2)));
    }

    delete [] Cw;
    Cvec->decref();
  }

  // Deallocate vector
  delete [] c;
  delete [] ctemp;
  delete [] Apx;
  if (!xvec){
    x->decref();
  }
  px->decref();
  g->decref();
  for ( int i = 0; i < ncon; i++ ){
    Ac[i]->decref();
  }
  delete [] Ac;
  xt->decref();

  if (ztemp){
    delete [] ztemp;
  }
  if (zw){
    zw->decref();
  }
  if (hvec){
    hvec->decref();
  }
  if (htemp){
    htemp->decref();
  }
  if (cw){
    cw->decref();
  }
  if (cwtemp){
    cwtemp->decref();
  }
}
