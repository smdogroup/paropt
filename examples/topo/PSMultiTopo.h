#ifndef PLANE_STRESS_MULTI_TOPO_H
#define PLANE_STRESS_MULTI_TOPO_H

#include "PlaneStressStiffness.h"
#include "TACS2DElement.h"

class PSMultiTopo : public PlaneStressStiffness {
 public:
  static const int MAX_NUM_MATERIALS = 5;

  PSMultiTopo( TacsScalar _rho[], TacsScalar _E[], 
               TacsScalar _nu[], int _num_mats,
               int _dv_offset, TacsScalar _eps ){
    // Set the values of the variables
    q = 0.0;
    dv_offset = _dv_offset;
    num_mats = _num_mats;
    eps = _eps;

    x0[0] = 0.0;
    x[0] = 0.5;

    for ( int k = 0; k < num_mats; k++ ){
      rho[k] = _rho[k];
      E[k] = _E[k];
      nu[k] = _nu[k];
      D[k] = E[k]/(1.0 - nu[k]*nu[k]);
      G[k] = 0.5*E[k]/(1.0 + nu[k]);

      // Set the initial values of the bounds
      x0[k+1] = 0.0;
      x[k+1] = 0.5;
      xconst[k] = 0.0;
      xlin[k] = 1.0;
    }
  }

  void setLinearization( TacsScalar _q,
                         const TacsScalar dvs[], int numDVs ){ 
    q = _q;

    // Record the design variable values
    for ( int k = 0; k < num_mats+1; k++ ){
      x0[k] = dvs[dv_offset + k];
    }

    // Set the linearization for the material weights
    for ( int k = 0; k < num_mats; k++ ){
      xconst[k] = x0[k+1]/(1.0 + q*(1.0 - x0[k+1])) + eps;
      xlin[k] = (q + 1.0)/((1.0 + q*(1.0 - x0[k+1]))*(1.0 + q*(1.0 - x0[k+1])));
    }
  }
  void setDesignVars( const TacsScalar dvs[], int numDVs ){
    for ( int k = 0; k < num_mats+1; k++ ){
      x[k] = dvs[dv_offset + k];
    }
  }
  void getDesignVars( TacsScalar dvs[], int numDVs ){
    for ( int k = 0; k < num_mats+1; k++ ){
      dvs[dv_offset + k] = x[k];
    }
  }
  void getDesignVarRange( TacsScalar lb[], TacsScalar ub[],
                          int numDVs ){
    lb[dv_offset] = 0.0;
    ub[dv_offset] = 1.0;
    for ( int k = 1; k < num_mats+1; k++ ){
      lb[dv_offset + k] = q*x0[k]*x0[k]/(1.0 + q);
      ub[dv_offset + k] = 1e30;
    }
  }

  void calculateStress( const double pt[], 
                        const TacsScalar e[], 
                        TacsScalar s[] ){
    s[0] = s[1] = s[2] = 0.0;
    for ( int k = 0; k < num_mats; k++ ){
      TacsScalar Dx = D[k]*(xconst[k] + xlin[k]*(x[k+1] - x0[k+1]));
      TacsScalar Gx = G[k]*(xconst[k] + xlin[k]*(x[k+1] - x0[k+1]));
      s[0] += Dx*(e[0] + nu[k]*e[1]);
      s[1] += Dx*(e[1] + nu[k]*e[0]);
      s[2] += Gx*e[2];
    }
  }
  void addStressDVSens( const double pt[], const TacsScalar e[],
                        TacsScalar alpha, const TacsScalar psi[],
                        TacsScalar fdvSens[], int dvLen ){
    for ( int k = 0; k < num_mats; k++ ){
      fdvSens[dv_offset + 1+k] += 
        alpha*xlin[k]*(psi[0]*(D[k]*(e[0] + nu[k]*e[1])) +
                       psi[1]*(D[k]*(e[1] + nu[k]*e[0])) +
                       psi[2]*G[k]*e[2]);
    }
  }

  // Calculate the derivative of the stress projected onto the
  // design variable values. This is required for second derivative
  // computations
  void calcStressDVProject( const double pt[],
                            const TacsScalar e[],
                            const TacsScalar px[],
                            int dvLen, TacsScalar s[] ){
    s[0] = s[1] = s[2] = 0.0;
    for ( int k = 0; k < num_mats; k++ ){
      TacsScalar Dx = D[k]*xlin[k]*px[dv_offset + k+1];
      TacsScalar Gx = G[k]*xlin[k]*px[dv_offset + k+1];
      s[0] += Dx*(e[0] + nu[k]*e[1]);
      s[1] += Dx*(e[1] + nu[k]*e[0]);
      s[2] += Gx*e[2];
    }
  }

  void getPointwiseMass( const double pt[], TacsScalar mass[] ){
    mass[0] = 0.0;
    for ( int k = 0; k < num_mats; k++ ){
      mass[0] += x[k+1]*rho[k];
    }
  }
  void addPointwiseMassDVSens( const double pt[],
                               const TacsScalar alpha[],
                               TacsScalar fdvSens[], int dvLen ){
    for ( int k = 0; k < num_mats; k++ ){
      fdvSens[dv_offset + 1+k] += alpha[0]*rho[k];
    }
  }

 public:
  // Set the lower bound on the stiffness
  TacsScalar eps;

  // The RAMP parameter value
  TacsScalar q;

  // Set the value of the design variable offset
  int dv_offset;  

  // The material properties
  int num_mats;
  TacsScalar rho[MAX_NUM_MATERIALS];
  TacsScalar E[MAX_NUM_MATERIALS];
  TacsScalar nu[MAX_NUM_MATERIALS];
  TacsScalar D[MAX_NUM_MATERIALS];
  TacsScalar G[MAX_NUM_MATERIALS];

  // Design variable values
  TacsScalar x[MAX_NUM_MATERIALS+1];
  TacsScalar x0[MAX_NUM_MATERIALS+1];

  // Linearization terms for the penalty function
  TacsScalar xconst[MAX_NUM_MATERIALS];
  TacsScalar xlin[MAX_NUM_MATERIALS];
};

/*
  The following function computes the product requried for the second
  derivative computation. This is not standard and so is implemented
  in a different manner.
*/
void assembleResProjectDVSens( TACSAssembler *tacs,
                               const TacsScalar *px,
                               int dvLen,
                               BVec *residual ){
  residual->zeroEntries();
  static const int NUM_NODES = 9;
  
  // Get the number of dependent nodes
  const int varsPerNode = tacs->getVarsPerNode();
  const int numNodes = tacs->getNumNodes();
  const int numDependentNodes = tacs->getNumDependentNodes();
  int size = varsPerNode*(numNodes + numDependentNodes);
  TacsScalar *localRes;
  tacs->getLocalArrays(NULL, &localRes, NULL, NULL, NULL);
  memset(localRes, 0, size*sizeof(TacsScalar));

  // Get the residual vector
  int num_elements = tacs->getNumElements();
  for ( int k = 0; k < num_elements; k++ ){
    TacsScalar Xpts[3*NUM_NODES];
    TacsScalar vars[2*NUM_NODES], dvars[2*NUM_NODES];
    TacsScalar ddvars[2*NUM_NODES];

    TACSElement *element = tacs->getElement(k, Xpts, 
                                            vars, dvars, ddvars);

    // Dynamically cast the element to the 2D element type
    TACS2DElement<NUM_NODES> *elem = 
      dynamic_cast<TACS2DElement<NUM_NODES>*>(element);
    if (elem){
      TACSConstitutive *constitutive = 
        elem->getConstitutive();
      
      PSMultiTopo *con = dynamic_cast<PSMultiTopo*>(constitutive);
      if (con){
        TacsScalar res[2*NUM_NODES];
        memset(res, 0, 2*NUM_NODES*sizeof(TacsScalar));

        static const int NUM_STRESSES = 3;
        static const int NUM_VARIABLES = 2*NUM_NODES;

        // The shape functions associated with the element
        double N[NUM_NODES];
        double Na[NUM_NODES], Nb[NUM_NODES];
  
        // The derivative of the stress with respect to the strain
        TacsScalar B[NUM_STRESSES*NUM_VARIABLES];
        
        // Get the number of quadrature points
        int numGauss = elem->getNumGaussPts();

        for ( int n = 0; n < numGauss; n++ ){
          // Retrieve the quadrature points and weight
          double pt[3];
          double weight = elem->getGaussWtsPts(n, pt);

          // Compute the element shape functions
          elem->getShapeFunctions(pt, N, Na, Nb);

          // Compute the derivative of X with respect to the
          // coordinate directions
          TacsScalar X[3], Xa[9];
          elem->planeJacobian(X, Xa, N, Na, Nb, Xpts);

          // Compute the determinant of Xa and the transformation
          TacsScalar J[4];
          TacsScalar h = FElibrary::jacobian2d(Xa, J);
          h = h*weight;

          // Compute the strain
          TacsScalar strain[NUM_STRESSES];
          elem->evalStrain(strain, J, Na, Nb, vars);
 
          // Compute the corresponding stress
          TacsScalar stress[NUM_STRESSES];
          con->calcStressDVProject(pt, strain, px, dvLen, stress);
       
          // Get the derivative of the strain with respect to the nodal
          // displacements
          elem->getBmat(B, J, Na, Nb, vars);

          TacsScalar *b = B;
          for ( int i = 0; i < NUM_VARIABLES; i++ ){
            res[i] += h*(b[0]*stress[0] + b[1]*stress[1] + b[2]*stress[2]);
            b += NUM_STRESSES;
          }
        }

        // Add the residual values
        tacs->addValues(varsPerNode, k, res, localRes);
      }
    }    
  }

  // Add the dependent-variable residual from the dependent nodes
  tacs->addDependentResidual(varsPerNode, localRes);

  // Assemble the full residual
  BVecDistribute *vecDist = tacs->getBVecDistribute();
  vecDist->beginReverse(localRes, residual, BVecDistribute::ADD);
  vecDist->endReverse(localRes, residual, BVecDistribute::ADD);

  // Set the boundary conditions
  residual->applyBCs();
}

#endif // PLANE_STRESS_MULTI_TOPO_H
