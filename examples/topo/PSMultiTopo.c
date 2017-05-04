#include "PSMultiTopo.h"
#include <math.h>
#include "tacslapack.h"


PSMultiTopoProperties::PSMultiTopoProperties( TacsScalar _rho[], 
                                              TacsScalar Cmat[],
                                              int num_mats ){
  // By default, use the convex penalty approximation
  penalty = PS_CONVEX;
  
  // Set the material parameters
  q = 1.0;
  eps = 1e-6;

  // Set the relative tolerance
  eps = eps/num_mats;

  num_materials = num_mats;
  for ( int k = 0; k < num_mats; k++ ){
    rho[k] = _rho[k];
    for ( int j = 0; j < 6; j++ ){
      C[6*k+j] = Cmat[6*k+j];
    }
  }
}

PSMultiTopoProperties::~PSMultiTopoProperties(){}
  
void PSMultiTopoProperties::setPenalization( double _q ){
  q = _q;
}

double PSMultiTopoProperties::getPenalization(){
  return q;
}

PSMultiTopo::PSMultiTopo( PSMultiTopoProperties *_mats,
                          int _nodes[], double _weights[],
                          int _nweights ){
  // Set the material properties
  mats = _mats;
  mats->incref();

  // Set the initial design variable value
  x0[0] = 0.0;
  x[0] = 0.5;

  // Set the weights/node numbers
  nweights = _nweights;
  for ( int k = 0; k < nweights; k++ ){
    nodes[k] = _nodes[k];
    weights[k] = _weights[k];
  }

  // Set the initial design variable values/bounds
  for ( int k = 0; k < mats->num_materials; k++ ){
    x0[k+1] = 0.0;
    x[k+1] = 1.0/mats->num_materials;
    xconst[k] = mats->eps;
    xlin[k] = 1.0;
  }
}

PSMultiTopo::~PSMultiTopo(){}

// Set the linearization about the current design point. 
void PSMultiTopo::setLinearization( const TacsScalar dvs[], int numDVs ){ 
  // Record the design variable values
  const int vars_per_node = mats->num_materials+1;
  memset(x0, 0, vars_per_node*sizeof(TacsScalar));
  for ( int i = 0; i < nweights; i++ ){
    for ( int j = 0; j < vars_per_node; j++ ){
      x0[j] += weights[i]*dvs[vars_per_node*nodes[i] + j];
    }
  }

  // Set the linearization for the material weights
  const double q = mats->q;
  for ( int k = 0; k < mats->num_materials; k++ ){
    xconst[k] = x0[k+1]/(1.0 + q*(1.0 - x0[k+1])) + mats->eps;
    xlin[k] = (q + 1.0)/((1.0 + q*(1.0 - x0[k+1]))*(1.0 + q*(1.0 - x0[k+1])));
  }
}
  
// Set the design variable values
void PSMultiTopo::setDesignVars( const TacsScalar dvs[], int numDVs ){
  // Record the design variable values
  const int vars_per_node = mats->num_materials+1;
  memset(x, 0, vars_per_node*sizeof(TacsScalar));
  for ( int i = 0; i < nweights; i++ ){
    for ( int j = 0; j < vars_per_node; j++ ){
      x[j] += weights[i]*dvs[vars_per_node*nodes[i] + j];
    }
  }
}

// Retrieve the design variable values. This call has no effect...
void PSMultiTopo::getDesignVars( TacsScalar dvs[], int numDVs ){}

// Get the design variable range
void PSMultiTopo::getDesignVarRange( TacsScalar lb[], TacsScalar ub[],
                                     int numDVs ){
  // number of variables per node
  const int vars_per_node = mats->num_materials+1;
  for ( int i = 0; i < nweights; i++ ){
    lb[vars_per_node*nodes[i]] = -1e30;
    ub[vars_per_node*nodes[i]] = 1.0;
    for ( int k = 1; k < mats->num_materials+1; k++ ){
      lb[vars_per_node*nodes[i] + k] = 0.0;
      ub[vars_per_node*nodes[i] + k] = 1e30;
    }
  }
}

// Compute the stress at a parametric point in the element based on
// the local strain value
void PSMultiTopo::calculateStress( const double pt[], 
                                   const TacsScalar e[], 
                                   TacsScalar s[] ){
  s[0] = s[1] = s[2] = 0.0;
  if (mats->penalty == PSMultiTopoProperties::PS_CONVEX){
    for ( int k = 0; k < mats->num_materials; k++ ){
      const TacsScalar *C = &mats->C[6*k];
      const TacsScalar w = xconst[k] + xlin[k]*(x[k+1] - x0[k+1]);
      s[0] += w*(C[0]*e[0] + C[1]*e[1] + C[2]*e[2]);
      s[1] += w*(C[1]*e[0] + C[3]*e[1] + C[4]*e[2]);
      s[2] += w*(C[2]*e[0] + C[4]*e[1] + C[5]*e[2]);
    }
  }
  else {
    const double q = mats->q;
    for ( int k = 0; k < mats->num_materials; k++ ){
      const TacsScalar *C = &mats->C[6*k];
      const TacsScalar w = x[k+1]/(1.0 + q*(1.0 - x[k+1])) + mats->eps;
      s[0] += w*(C[0]*e[0] + C[1]*e[1] + C[2]*e[2]);
      s[1] += w*(C[1]*e[0] + C[3]*e[1] + C[4]*e[2]);
      s[2] += w*(C[2]*e[0] + C[4]*e[1] + C[5]*e[2]);
    }
  }
}

/*
  Add the derivative of the product of the strain with an input vector
  psi and add it to the array fdvSens
*/
void PSMultiTopo::addStressDVSens( const double pt[], const TacsScalar e[],
                                   TacsScalar alpha, const TacsScalar psi[],
                                   TacsScalar fdvSens[], int dvLen ){
  const int vars_per_node = mats->num_materials+1;
  if (mats->penalty == PSMultiTopoProperties::PS_CONVEX){
    for ( int j = 0; j < mats->num_materials; j++ ){
      const TacsScalar *C = &mats->C[6*j];
      TacsScalar s[3];
      s[0] = C[0]*e[0] + C[1]*e[1] + C[2]*e[2];
      s[1] = C[1]*e[0] + C[3]*e[1] + C[4]*e[2];
      s[2] = C[2]*e[0] + C[4]*e[1] + C[5]*e[2];

      // Compute the contribution to the derivative
      TacsScalar scale = alpha*xlin[j]*(psi[0]*s[0] + psi[1]*s[1] + psi[2]*s[2]);

      // Add the dependency on the filter
      for ( int i = 0; i < nweights; i++ ){
        fdvSens[vars_per_node*nodes[i] + 1+j] += scale*weights[i];
      }
    }
  }
  else {
    const double q = mats->q;
    for ( int j = 0; j < mats->num_materials; j++ ){
      const TacsScalar *C = &mats->C[6*j];
      TacsScalar s[3];
      s[0] = C[0]*e[0] + C[1]*e[1] + C[2]*e[2];
      s[1] = C[1]*e[0] + C[3]*e[1] + C[4]*e[2];
      s[2] = C[2]*e[0] + C[4]*e[1] + C[5]*e[2];

      // Compute the contribution to the derivative
      TacsScalar scale = alpha*(psi[0]*s[0] + psi[1]*s[1] + psi[2]*s[2]);
      scale *= (q + 1.0)/((1.0 + q*(1.0 - x[j+1]))*(1.0 + q*(1.0 - x[j+1])));     

      // Add the derivative due to the filter weights
      for ( int i = 0; i < nweights; i++ ){
        fdvSens[vars_per_node*nodes[i] + 1+j] += scale*weights[i];
      }
    }
  }
}

/* 
   Calculate the derivative of the stress projected onto the design
   variable values. This is required for second derivative
   computations.
*/
void PSMultiTopo::calcStressDVProject( const double pt[],
                                       const TacsScalar e[],
                                       const TacsScalar px[],
                                       int dvLen, TacsScalar s[] ){
  s[0] = s[1] = s[2] = 0.0;
  const int vars_per_node = mats->num_materials+1;
  if (mats->penalty == PSMultiTopoProperties::PS_CONVEX){
    for ( int j = 0; j < mats->num_materials; j++ ){
      const TacsScalar *C = &mats->C[6*j];
      TacsScalar s0[3];
      s0[0] = C[0]*e[0] + C[1]*e[1] + C[2]*e[2];
      s0[1] = C[1]*e[0] + C[3]*e[1] + C[4]*e[2];
      s0[2] = C[2]*e[0] + C[4]*e[1] + C[5]*e[2];

      for ( int i = 0; i < nweights; i++ ){
        const TacsScalar wx = weights[i]*xlin[j]*px[vars_per_node*nodes[i] + j+1];
        s[0] += wx*s0[0];
        s[1] += wx*s0[1];
        s[2] += wx*s0[2];
      }
    }
  }
  else {
    const double q = mats->q;
    for ( int j = 0; j < mats->num_materials; j++ ){
      const TacsScalar *C = &mats->C[6*j];
      TacsScalar s0[3];
      s0[0] = C[0]*e[0] + C[1]*e[1] + C[2]*e[2];
      s0[1] = C[1]*e[0] + C[3]*e[1] + C[4]*e[2];
      s0[2] = C[2]*e[0] + C[4]*e[1] + C[5]*e[2];

      // Compute the derivative of the weight
      const TacsScalar scale = 
        (q + 1.0)/((1.0 + q*(1.0 - x[j+1]))*(1.0 + q*(1.0 - x[j+1])));

      for ( int i = 0; i < nweights; i++ ){
        const TacsScalar wx = scale*weights[i]*px[vars_per_node*nodes[i] + j+1];
        s[0] += wx*s0[0];
        s[1] += wx*s0[1];
        s[2] += wx*s0[2];
      }
    }
  }
}

/*
  Add the term from the second derivative of the inner product
*/
void PSMultiTopo::addStress2ndDVSensProduct( const double pt[], const TacsScalar e[],
                                             TacsScalar alpha, const TacsScalar psi[],
                                             const TacsScalar px[],
                                             TacsScalar fdvSens[], int dvLen ){
  const int vars_per_node = mats->num_materials+1;
  if (mats->penalty == PSMultiTopoProperties::PS_FULL){
    const double q = mats->q;
    for ( int j = 0; j < mats->num_materials; j++ ){
      const TacsScalar *C = &mats->C[6*j];
      TacsScalar s[3];
      s[0] = C[0]*e[0] + C[1]*e[1] + C[2]*e[2];
      s[1] = C[1]*e[0] + C[3]*e[1] + C[4]*e[2];
      s[2] = C[2]*e[0] + C[4]*e[1] + C[5]*e[2];

      // Add the derivative due to the filter weights
      TacsScalar proj = 0.0;
      for ( int i = 0; i < nweights; i++ ){
        proj += px[vars_per_node*nodes[i] + 1+j]*weights[i];
      }

      // Compute the contribution to the derivative
      TacsScalar a = 1.0/(1.0 + q*(1.0 - x[j+1]));
      TacsScalar scale = alpha*(psi[0]*s[0] + psi[1]*s[1] + psi[2]*s[2]);
      scale *= 2.0*(q + 1.0)*a*a*a*proj;

      // Add the derivative due to the filter weights
      for ( int i = 0; i < nweights; i++ ){
        fdvSens[vars_per_node*nodes[i] + 1+j] += scale*weights[i];
      }
    }
  }
}

// Compute the mass at this point
void PSMultiTopo::getPointwiseMass( const double pt[], TacsScalar mass[] ){
  mass[0] = 0.0;
  for ( int k = 0; k < mats->num_materials; k++ ){
    mass[0] += x[k+1]*mats->rho[k];
  }
}

// Add the derivative of the mass at this point
void PSMultiTopo::addPointwiseMassDVSens( const double pt[],
                                          const TacsScalar alpha[],
                                          TacsScalar fdvSens[], int dvLen ){
  const int vars_per_node = mats->num_materials+1;
  for ( int i = 0; i < nweights; i++ ){
    for ( int j = 0; j < mats->num_materials; j++ ){
      fdvSens[vars_per_node*nodes[i] + 1+j] += weights[i]*alpha[0]*mats->rho[j];
    }
  }
}

/*
  The following function computes the product requried for the second
  derivative computation. This is not standard and so is implemented
  in a different manner.
*/
void assembleResProjectDVSens( TACSAssembler *tacs,
                               const TacsScalar *px,
                               int dvLen,
                               TacsScalar *fdvSens,
                               TACSBVec *residual ){
  residual->zeroEntries();
  static const int NUM_NODES = 4;
  static const int NUM_STRESSES = 3;
  static const int NUM_VARIABLES = 2*NUM_NODES;
  
  // Get the residual vector
  int num_elements = tacs->getNumElements();
  for ( int k = 0; k < num_elements; k++ ){
    TACSElement *element = tacs->getElement(k, NULL, NULL, 
                                            NULL, NULL);

    // Dynamically cast the element to the 2D element type
    TACS2DElement<NUM_NODES> *elem = 
      dynamic_cast<TACS2DElement<NUM_NODES>*>(element);

    if (elem){
      TacsScalar Xpts[3*NUM_NODES];
      TacsScalar vars[2*NUM_NODES], dvars[2*NUM_NODES];
      TacsScalar ddvars[2*NUM_NODES];
      tacs->getElement(k, Xpts, vars, dvars, ddvars);

      TACSConstitutive *constitutive = 
        elem->getConstitutive();
      
      PSMultiTopo *con = dynamic_cast<PSMultiTopo*>(constitutive);
      if (con){
        TacsScalar res[2*NUM_NODES];
        memset(res, 0, 2*NUM_NODES*sizeof(TacsScalar));

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

          // Add the contribution -u^{T}*d^2K/dx^2*u to the derivative
          con->addStress2ndDVSensProduct(pt, strain, -1.0, strain, px, fdvSens, dvLen);

          // Compute the corresponding stress
          TacsScalar stress[NUM_STRESSES];
          con->calcStressDVProject(pt, strain, px, dvLen, stress);

          // Get the derivative of the strain with respect to the
	  // nodal displacements
          elem->getBmat(B, J, Na, Nb, vars);

          TacsScalar *b = B;
          for ( int i = 0; i < NUM_VARIABLES; i++ ){
            res[i] += h*(b[0]*stress[0] + b[1]*stress[1] + b[2]*stress[2]);
            b += NUM_STRESSES;
          }
        }

	// Get the local element ordering
	int len;
	const int *nodes;
	tacs->getElement(k, &nodes, &len);

        // Add the residual values
        residual->setValues(len, nodes, res, ADD_VALUES);
      }
    }    
  }

  // Add the residual values
  residual->beginSetValues(ADD_VALUES);
  residual->endSetValues(ADD_VALUES);

  // Set the boundary conditions
  tacs->applyBCs(residual);
}

/*
  Create an object that can rapidly locate the closest point within a
  cloud of points to the specified input point.  This works in
  O(log(n)) time roughly, rather than O(n) time.
*/ 
LocatePoint::LocatePoint( const TacsScalar *_Xpts, int _npts, 
			  int _max_num_points ){
  Xpts = _Xpts;
  npts = _npts;
  max_num_points = _max_num_points;

  // Calculate approximately how many nodes there should be
  max_nodes = (2*npts)/max_num_points;
  num_nodes = 0;

  // The point indicies
  indices = new int[ npts ];
  for ( int i = 0; i < npts; i++ ){
    indices[i] = i;
  }

  // Set up the data structure that represents the
  // splitting planes
  nodes       = new int[ 2*max_nodes ];
  indices_ptr = new int[ max_nodes ];
  num_indices = new int[ max_nodes ];

  // The base point and normal direction for the splitting
  // planes
  node_xav    = new TacsScalar[ 3*max_nodes ];
  node_normal = new TacsScalar[ 3*max_nodes ];

  for ( int i = 0; i < max_nodes; i++ ){
    nodes[2*i] = nodes[2*i+1] = -1;
    indices_ptr[i] = -1;
    num_indices[i] = -1;

    for ( int k = 0; k < 3; k++ ){
      node_xav[3*i+k] = 0.0;
      node_normal[3*i+k] = 0.0;
    }
  }

  // Recursively split the points
  split(0, npts);
}

/*
  Deallocate the memory for this object
*/
LocatePoint::~LocatePoint(){
  delete [] indices;    
  delete [] nodes;      
  delete [] indices_ptr;
  delete [] num_indices;
  delete [] node_xav;   
  delete [] node_normal;
}

/*
  Locate the K closest points in the domain to the point using
  the plane-splitting method.
*/
void LocatePoint::locateKClosest( int K, int indx[], TacsScalar dist[], 
				  const TacsScalar xpt[] ){
  int nk = 0; // Length of the array
  int root = 0;

  locateKClosest(K, root, xpt, dist, indx, &nk);

  // Check that the array of indices is in fact sorted
  if (nk < K){
    printf("Error nk = %d < K = %d \n", nk, K);
  }

  // Check if the list is properly sorted
  int flag = 0;
  for ( int k = 0; k < nk-1; k++ ){
    if(!(dist[k] <= dist[k+1])){
      flag = 1;
      break;
    }
  }
  if (flag){
    printf("Error: list not sorted \n");
    for ( int k = 0; k < nk; k++ ){
      printf("dist[%d] = %g \n", k, TacsRealPart(dist[k]));
    }
  }

  // Take the square root to obtain the true distance
  for ( int k = 0; k < nk; k++ ){
    dist[k] = sqrt(dist[k]);
  }
}

/*!
  Insert a point into a sorted list based upon the distance from the
  given point
*/
void LocatePoint::insertIndex( TacsScalar *dist, int *indx, int *nk, 
			       TacsScalar d, int dindex, int K ){

  if (*nk == 0){
    dist[*nk] = d;
    indx[*nk] = dindex;
    *nk += 1;
    return;
  }  
  else if (*nk < K && dist[*nk-1] <= d){
    dist[*nk] = d;
    indx[*nk] = dindex;
    *nk += 1;
    return;
  }

  // Place it into the list
  int i = 0;
  while (i < *nk && (d >= dist[i])){
    i++;
  }

  for ( ; i < *nk; i++ ){
    int tindex = indx[i];
    TacsScalar t = dist[i];
    indx[i] = dindex;
    dist[i] = d;
    dindex = tindex;
    d = t;
  }

  if (*nk < K){
    indx[*nk] = dindex;
    dist[*nk] = d;
    *nk += 1;
  }
}

/*!
  Locate the K-closest points to a given point!

  dist  == A sorted list of the K-closest distances
  indx  == The indices of the K-closest values 
  nk    == The actual number of points in the list nk <= K
*/
void LocatePoint::locateKClosest( int K, int root, const TacsScalar xpt[], 
				  TacsScalar *dist, int *indx, int *nk ){  
  int start = indices_ptr[root];
  int left_node = nodes[2*root];
  int right_node = nodes[2*root+1];

  if (start != -1){ // This node is a leaf
    // Do an exhaustive search of the points at the node
    
    int end = start + num_indices[root];
    for ( int k = start; k < end; k++ ){
      int n = indices[k];

      TacsScalar t = ((Xpts[3*n]   - xpt[0])*(Xpts[3*n]   - xpt[0]) +
                      (Xpts[3*n+1] - xpt[1])*(Xpts[3*n+1] - xpt[1]) +
                      (Xpts[3*n+2] - xpt[2])*(Xpts[3*n+2] - xpt[2]));

      if ((*nk < K) || (t < dist[K-1])){
	insertIndex(dist, indx, nk, t, n, K );
      }
    }
  }
  else {
    TacsScalar *xav    = &node_xav[3*root];
    TacsScalar *normal = &node_normal[3*root];

    // The normal distance
    TacsScalar ndist = ((xpt[0] - xav[0])*normal[0] +
                        (xpt[1] - xav[1])*normal[1] +
                        (xpt[2] - xav[2])*normal[2]); 

    if (ndist < 0.0){ // The point lies to the 'left' of the plane
      locateKClosest(K, left_node, xpt, dist, indx, nk);

      // If the minimum distance to the plane is less than the minimum
      // distance then search the other branch too - there could be a
      // point on that branch that lies closer than *dist
      if (*nk < K || ndist*ndist < dist[*nk-1]){ 
	locateKClosest(K, right_node, xpt, dist, indx, nk);
      }
    }
    else { // The point lies to the 'right' of the plane
      locateKClosest(K, right_node, xpt, dist, indx, nk);

      // If the minimum distance to the plane is less than the minimum
      // distance then search the other branch too - there could be a
      // point on that branch that lies closer than *dist
      if (*nk < K || ndist*ndist < dist[*nk-1]){
	locateKClosest(K, left_node, xpt, dist, indx, nk);
      }
    }
  }
}

/*!
  Split the list of indices into approximately two.
  Those on one half of a plane and those on the other.
*/
int LocatePoint::split(int start, int end ){  
  int root = num_nodes;

  num_nodes++;
  if (num_nodes >= max_nodes ){
    extendArrays(num_nodes, 2*(num_nodes+1));
    max_nodes = 2*(num_nodes+1);
  }
  
  if (end - start <= max_num_points ){
    nodes[ 2*root ]    = -1;
    nodes[ 2*root+1 ]  = -1;
    
    for ( int k = 0; k < 3; k++ ){
      node_xav[ 3*root + k ] = 0.0;
      node_normal[ 3*root + k ] = 0.0;
    }

    indices_ptr[root] = start;
    num_indices[root] = end - start;

    return root;
  }

  indices_ptr[root] = -1;
  num_indices[root] = 0;  

  int mid = splitList(&node_xav[3*root], &node_normal[3*root], 
		       &indices[start], end-start);

  if (mid == 0 || mid == end-start ){
    fprintf(stderr, "LocatePoint: Error, splitting points did nothing. \
Problem with your nodes?\n");
    return root;
  }

  // Now, split the right and left hand sides of the list 
  int left_node = split(start, start + mid);
  int right_node = split(start + mid, end);

  nodes[ 2*root ]   = left_node;
  nodes[ 2*root+1 ] = right_node;

  return root;
}

/*!
  Split the array of indices into two sets: those indices that correspond
  to points on either side of a plane in three-space.
*/
int LocatePoint::splitList(TacsScalar xav[], TacsScalar normal[], 
                           int *ind, int np ){
  xav[0] = xav[1] = xav[2] = TacsScalar(0.0);
  normal[0] = normal[1] = normal[2] = TacsScalar(0.0);

  // lwork  = 1 + 6*N + 2*N**2
  // liwork = 3 + 5*N
  double eigs[3];
  int N = 3;
  int lwork = 1 + 6*N + 2*N*N;
  double work[1 + 6*3 + 2*3*3];
  int liwork = 3 + 5*N;
  int iwork[3 + 5*3];

  double I[9];
  for ( int i = 0; i < 9; i++ ){
    I[i] = 0.0;
  }

  // Find the average point and the moment of inertia about the average point
  for ( int i = 0; i < np; i++ ){
    int n = ind[i];
    for ( int k = 0; k < 3; k++ ){
      xav[k] += TacsRealPart(Xpts[3*n+k]);
    }

    // I[0] = Ix = y^2 + z^2
    I[0] += TacsRealPart(Xpts[3*n+1]*Xpts[3*n+1] + Xpts[3*n+2]*Xpts[3*n+2]); 
    // I[4] = Iy = x^2 + z^2
    I[4] += TacsRealPart(Xpts[3*n]*Xpts[3*n] + Xpts[3*n+2]*Xpts[3*n+2]);
     // I[8] = Iz = x^2 + y^2
    I[8] += TacsRealPart(Xpts[3*n]*Xpts[3*n] + Xpts[3*n+1]*Xpts[3*n+1]);

    I[1] += - TacsRealPart(Xpts[3*n]*Xpts[3*n+1]); // Ixy = - xy
    I[2] += - TacsRealPart(Xpts[3*n]*Xpts[3*n+2]); // Ixz = - xz
    I[5] += - TacsRealPart(Xpts[3*n+1]*Xpts[3*n+2]); // Ixz = - yz
  }

  for ( int k = 0; k < 3; k++ ){
    xav[k] = xav[k]/double(np);
  }

  // Ix(cm) = Ix - np*(yav^2 + zav^2) ... etc
  I[0] = I[0] - np*TacsRealPart(xav[1]*xav[1] + xav[2]*xav[2]);
  I[4] = I[4] - np*TacsRealPart(xav[0]*xav[0] + xav[2]*xav[2]);
  I[8] = I[8] - np*TacsRealPart(xav[0]*xav[0] + xav[1]*xav[1]);
  
  I[1] = I[1] + np*TacsRealPart(xav[0]*xav[1]);
  I[2] = I[2] + np*TacsRealPart(xav[0]*xav[2]);
  I[5] = I[5] + np*TacsRealPart(xav[1]*xav[2]);

  I[3] = I[1];
  I[6] = I[2];
  I[7] = I[5];

  // Find the eigenvalues/eigenvectors
  int info;
  const char *jobz = "V";
  const char *uplo = "U";

  LAPACKsyevd(jobz, uplo, &N, I, &N,
	      eigs, work, &lwork, iwork, &liwork, &info);

  normal[0] = I[0];
  normal[1] = I[1];
  normal[2] = I[2];

  int low = 0;
  int high = np-1;

  // Now, split the index array such that 
  while (high > low){
    // (dot(Xpts[ind] - xav, n ) < 0 ) < 0.0 for i < low
    while (high > low &&	    
           ((Xpts[3*ind[low]]   - xav[0] )*normal[0] +
            (Xpts[3*ind[low]+1] - xav[1] )*normal[1] +
            (Xpts[3*ind[low]+2] - xav[2] )*normal[2]) < 0.0){
      low++;
    }

    // (dot(Xpts[ind] - xav, n ) < 0 ) >= 0.0 for i >= high
    while (high > low &&
           ((Xpts[3*ind[high]]   - xav[0] )*normal[0] +
            (Xpts[3*ind[high]+1] - xav[1] )*normal[1] +
            (Xpts[3*ind[high]+2] - xav[2] )*normal[2]) >= 0.0){
      high--;
    }

    if (high > low){
      // Switch the two indices that don't match
      int temp = ind[high];
      ind[high] = ind[low];
      ind[low] = temp;
    }
  }

  if (low == 0 || low == np){
    fprintf(stderr,  "LocatePoint: Error split points\n");
  }

  return low;
}

/*!
  If not enough memory has been allocated, extend the arrays required
  to store all the nodes.
*/
void LocatePoint::extendArrays( int old_len, int new_len ){
  nodes       = newIntArray(nodes, 2*old_len, 2*new_len);
  indices_ptr = newIntArray(indices_ptr, old_len, new_len);
  num_indices = newIntArray(num_indices, old_len, new_len);

  node_xav    = newDoubleArray(node_xav, 3*old_len, 3*new_len);
  node_normal = newDoubleArray(node_normal, 3*old_len, 3*new_len);
}

/*
  Allocate more space for an integer array and copy the old
  array to the newly created array
*/
int *LocatePoint::newIntArray( int *array, int old_len, int new_len ){
  int *temp = new int[ new_len ];

  for ( int i = 0; i < old_len; i++ ){
    temp[i] = array[i];
  }
  for ( int i = old_len; i < new_len; i++ ){
    temp[i] = -1;
  }

  delete [] array;

  return temp;
}

/*
  Allocate space for a new double array and copy the old
  array to the newly created array
*/
TacsScalar *LocatePoint::newDoubleArray( TacsScalar *array, int old_len, 
					  int new_len ){
  TacsScalar *temp = new TacsScalar[ new_len ];

  for ( int i = 0; i < old_len; i++ ){
    temp[i] = array[i];
  }
  for ( int i = old_len; i < new_len; i++ ){
    temp[i] = 0.0;
  }
  delete [] array;
  
  return temp;
}

