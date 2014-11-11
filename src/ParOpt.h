/*
  A parallel optimizer implemented in C++ for large-scale constrained
  parallel optimization.

  This code uses an interior-point method to perform gradient-based
  design optimization. The KKT system is solved using a bordered
  solution technique that may suffer from numerical precision issues
  under extreme values.

  The optimization problem is formulated as follows:

  min f(x)
  s.t.  lb <= x < ub
  s.t.  c(x) >= 0 
  s.t.  Aw*x = b
  
  where Aw is a large, sparse constraint matrix. The perturbed KKT
  conditions for this problem are:

  g(x) - A(x)^{T}*z - Aw^{T}*zw - zl + zu = 0
  Aw*x - b = 0
  c(x) - s = 0
  S*z - mu*e = 0
  (X - Xl)*zl - mu*e = 0
  (Xu - X)*zu - mu*e = 0

  where g = grad f(x) and A(x) = grad c(x). The Lagrange multipliers
  are z, zw, zl, and zu, respectively.  Note that here we assume that
  c(x) is small, x is very large, and Aw is also very large, but has a
  very specialized structure.

  At each step of the optimization, we compute a solution to the linear
  system above, using: 

  K*p = - r

  where K is the linearization of the above system of equations, p is
  a search direction, and r are the residuals. Instead of using an
  exact linearization, we use an approximation based on compact L-BFGS
  representation. To compute the update, we use the
  Sherman-Morrison-Woodbury formula. This is possible due to the
  compact L-BFGS representation.

  The KKT system can be written as follows:
  
  [  B   -Aw^{T} -A^{T}   0  -I         I        ][ px  ]
  [  Aw   0       0       0   0         0        ][ pzw ]
  [  A    0       0      -I   0         0        ][ pz  ] = -r
  [  0    0       S       Z   0         0        ][ ps  ]
  [  Zl   0       0       0   (X - Xl)  0        ][ pzl ]
  [ -Zu   0       0       0   0         (Xu - X) ][ pzu ]

  where B is a quasi-Newton Hessian approximation. This approximation
  takes the form:
  
  B = b0*I - Z*M*Z^{T}
*/

class ParOptVec {
 public:
  ParOptVec( int numm );

}


class LBFGSUpdate {
 public:
  BFGSMatrix( );

};
