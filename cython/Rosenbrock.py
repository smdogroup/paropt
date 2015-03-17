import pyParOptRosen
import mpi4py.MPI as MPI
import time

#Allocate the Rosenbrock function
comm = MPI.COMM_WORLD
nvars = 100
nwcon = 5
nw = 5
nwstart = 1
nwskip = 1

#Number of dense constraints (Should not be changed without recompiling)
ncon = 2
#Block size of Aw*D*Aw{T} (Should not be changed without recompiling)
nwblock = 1


#Create an instance of the pyRosenbrock
rosen = pyParOptRosen.pyRosenbrock(comm, nvars-1, ncon, nwcon, nwblock, nwstart, nw, nwskip)

#Allocate the optimizer
max_lbfgs = 20
qn_type = 1 #BFGS method

opt = pyParOptRosen.pyParOpt(rosen, max_lbfgs, qn_type )

opt.setGMRESSusbspaceSize(30)
opt.setNKSwitchTolerance(1e3)
opt.setGMRESTolerances(0.1, 1e-30)
opt.setUseHvecProduct(1)
opt.setMajorIterStepCheck(45)
opt.setMaxMajorIterations(1500)
opt.checkGradients(1e-6)

t = time.time()
opt.optimize()
print "Times taken: ", time.time()-t, " seconds"
