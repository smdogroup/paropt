#ifndef PAROPT_COMPLEX_STEP_H
#define PAROPT_COMPLEX_STEP_H

#include <complex>

/*
  Copyright (c) 2016 Graeme Kennedy. All rights reserved
*/

// Define the real part function for the complex data type
inline double ParOptRealPart(const std::complex<double>& c) { return real(c); }

// Define the imaginary part function for the complex data type
inline double ParOptImagPart(const std::complex<double>& c) { return imag(c); }

// Dummy function for real part
inline double ParOptRealPart(const double& r) { return r; }

// Compute the absolute value
#ifndef FABS_COMPLEX_IS_DEFINED  // prevent redefinition
#define FABS_COMPLEX_IS_DEFINED
inline std::complex<double> fabs(const std::complex<double>& c) {
  if (real(c) < 0.0) {
    return -c;
  }
  return c;
}
#endif  // FABS_COMPLEX_IS_DEFINED

#endif  // PAROPT_COMPLEX_STEP_H
