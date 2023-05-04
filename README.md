[![Build, unit tests, and docs](https://github.com/smdogroup/paropt/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/smdogroup/paropt/actions/workflows/unit_tests.yml)

[![Anaconda-Server Badge](https://anaconda.org/smdogroup/paropt/badges/version.svg)](https://anaconda.org/smdogroup/paropt)
[![Anaconda-Server Badge](https://anaconda.org/smdogroup/paropt/badges/platforms.svg)](https://anaconda.org/smdogroup/paropt)
[![Anaconda-Server Badge](https://anaconda.org/smdogroup/paropt/badges/downloads.svg)](https://anaconda.org/smdogroup/paropt)

# ParOpt: A parallel interior-point optimizer #
------------------------------------------------

ParOpt is a parallel optimization library for use in general large-scale optimization applications, but is often specifically used for topology and multi-material optimization problems.
The optimizer has the capability to handle large numbers of weighting constraints that arise in the parametrization of multi-material problems. 

The implementation of the optimizer is in C++ and uses MPI. ParOpt is also wrapped with python using Cython.

The ParOpt theory manual is located here: [ParOpt_theory_manual](docs/ParOpt_theory_manual.pdf)

Online documentation for ParOpt is located here: [https://smdogroup.github.io/paropt/](https://smdogroup.github.io/paropt/)

If you use ParOpt, please cite our paper:

Ting Wei Chin, Mark K. Leader, Graeme J. Kennedy, A scalable framework for large-scale 3D multimaterial topology optimization with octree-based mesh adaptation, Advances in Engineering Software, Volume 135, 2019.

```
@article{Chin:2019,
         title = {A scalable framework for large-scale 3D multimaterial topology optimization with octree-based mesh adaptation},
         journal = {Advances in Engineering Software},
         volume = {135},
         year = {2019},
         doi = {10.1016/j.advengsoft.2019.05.004},
         author = {Ting Wei Chin and Mark K. Leader and Graeme J. Kennedy}}
```

ParOpt is released under the terms of the LGPLv3 license.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
