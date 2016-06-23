#ifndef PLANE_STRESS_MULTI_TOPO_H
#define PLANE_STRESS_MULTI_TOPO_H


#include "PlaneStressStiffness.h"

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

    for ( int k = 0; k < num_mats; k++ ){
      rho[k] = _rho[k];
      E[k] = _E[k];
      nu[k] = _nu[k];
      D[k] = E[k]/(1.0 - nu[k]*nu[k]);
      G[k] = 0.5*E[k]/(1.0 + nu[k]);

      // Set the initial values of the bounds
      x0[k] = 0.0;
      x[k] = 0.5;
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

  void calculateStress( const double pt[], const TacsScalar e[], 
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

#endif // PLANE_STRESS_MULTI_TOPO_H
