#ifndef PLANE_STRESS_MULTI_TOPO_H
#define PLANE_STRESS_MULTI_TOPO_H

#include "TACSAssembler.h"
#include "PlaneStressStiffness.h"
#include "TACS2DElement.h"

/*
  The following function computes the product requried for the second
  derivative computation. This is not standard and so is implemented
  in a different manner.
*/
void assembleResProjectDVSens( TACSAssembler *tacs,
                               const TacsScalar *px,
                               int dvLen,
                               TACSBVec *residual );

class PSMultiTopoProperties : public TACSObject {
 public:
  enum PSPenaltyType { PS_CONVEX, PS_FULL };
  static const int MAX_NUM_MATERIALS = 12;
  PSMultiTopoProperties( TacsScalar _rho[], TacsScalar Cmat[],
                         int num_mats );
  ~PSMultiTopoProperties();
  void setPenalization( double _q );
  double getPenalization();
  int getNumMaterials(){ return num_materials; }
  void setPenaltyType( PSPenaltyType _penalty ){
    penalty = _penalty;
  }

  // The type of penalization to use: Convex or full
  PSPenaltyType penalty;

  // Set the material parameters
  double q;
  double eps;

  // The material properties
  int num_materials;
  TacsScalar rho[MAX_NUM_MATERIALS];
  TacsScalar C[6*MAX_NUM_MATERIALS];
};

class PSMultiTopo : public PlaneStressStiffness {
 public:
  static const int MAX_NUM_MATERIALS = 12;
  static const int MAX_NUM_WEIGHTS = 15;

  PSMultiTopo( PSMultiTopoProperties *_mats,
               int _nodes[], double _weights[],
               int _nweights );
  ~PSMultiTopo();

  // Get the filtered values of the design variables (for visualization)
  int getFilteredDesignVars( const TacsScalar **xf ){
    *xf = x;
    return mats->num_materials; 
  }

  void setLinearization( const TacsScalar dvs[], int numDVs );
  void setDesignVars( const TacsScalar dvs[], int numDVs );
  void getDesignVars( TacsScalar dvs[], int numDVs );
  void getDesignVarRange( TacsScalar lb[], TacsScalar ub[],
                          int numDVs );
  void calculateStress( const double pt[], const TacsScalar e[], 
                        TacsScalar s[] );
  void addStressDVSens( const double pt[], const TacsScalar e[],
                        TacsScalar alpha, const TacsScalar psi[],
                        TacsScalar fdvSens[], int dvLen );
  void calcStressDVProject( const double pt[],
                            const TacsScalar e[],
                            const TacsScalar px[],
                            int dvLen, TacsScalar s[] );
  void getPointwiseMass( const double pt[], TacsScalar mass[] );
  void addPointwiseMassDVSens( const double pt[],
                               const TacsScalar alpha[],
                               TacsScalar fdvSens[], int dvLen );

 public:
  // Material property information
  PSMultiTopoProperties *mats;

  // Maximum number of weights/nodes
  int nweights;
  int nodes[MAX_NUM_WEIGHTS];
  double weights[MAX_NUM_WEIGHTS];

  // Design variable values
  TacsScalar x[MAX_NUM_MATERIALS+1];
  TacsScalar x0[MAX_NUM_MATERIALS+1];

  // Linearization terms for the penalty function
  TacsScalar xconst[MAX_NUM_MATERIALS];
  TacsScalar xlin[MAX_NUM_MATERIALS];
};

/*
  Locate the points that are closest to a given point

  This is used to construct the filter
*/
class LocatePoint {
 public:
  LocatePoint( const TacsScalar *_Xpts, int _npts, int _max_num_points=10 );
  ~LocatePoint();

  // Locate the K-closest points (note that dist/indices must of length K)
  // ---------------------------------------------------------------------
  void locateKClosest( int K, int indices[], 
                       TacsScalar dist[], const TacsScalar xpt[] );

 private:
  // The recursive versions of the above functions
  void locateKClosest( int K, int root, const TacsScalar xpt[], 
		       TacsScalar *dist, int *indices, int *nk );

  // Insert the index into the sorted list of indices
  void insertIndex( TacsScalar *dist, int *indices, int *nk, 
		    TacsScalar d, int dindex, int K );

  // Sort the list of initial indices into the tree data structure
  int split( int start, int end );
  int splitList( TacsScalar xav[], TacsScalar normal[], 
                 int *indices, int npts );
  
  // Functions for array management
  void extendArrays( int old_len, int new_len );
  int *newIntArray( int *array, int old_len, int new_len );
  TacsScalar *newDoubleArray( TacsScalar *array, int old_len, int new_len );

  // The cloud of points to match
  const TacsScalar *Xpts;
  int npts; 

  // Maximum number of points stored at a leaf
  int max_num_points; 

  // Keep track of the nodes that have been created
  int max_nodes;
  int num_nodes;

  int *indices; // Indices into the array of points
  int *nodes;  // Indices from the current node to the two child nodes
  int *indices_ptr; // Pointer into the global indices array
  int *num_indices; // Number of indices associated with this node
  TacsScalar *node_xav; // Origin point for the array
  TacsScalar *node_normal; // Normal direction of the plane
};

#endif // PLANE_STRESS_MULTI_TOPO_H
