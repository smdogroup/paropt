
default:
	mpicxx -fPIC -g -I../src/ -c Rosenbrock.c -o Rosenbrock.o
	mpicxx -o rosenbrock Rosenbrock.o ../src/ParOpt.o ../src/ParOptVec.o -L/usr/lib/ -llapack -lblas
