# Test script used while wrapping ParOpt with OpenMDAO

from openmdao.api import Problem, ScipyOptimizeDriver, ExecComp, IndepVarComp
from paropt.paropt_driver import ParOptDriver
import argparse

# Create an argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--optimizer', default='Interior Point',
                    choices=['Interior Point', 'Trust Region'],
                    help='optimizer type')
args = parser.parse_args()

# build the model
prob = Problem()
indeps = prob.model.add_subsystem('indeps', IndepVarComp())
indeps.add_output('x', 3.0)
indeps.add_output('y', -4.0)

prob.model.add_subsystem('paraboloid', ExecComp('f = (x-3)**2 + x*y + (y+4)**2 - 3'))
prob.model.add_subsystem('con', ExecComp('c = x + y'))

prob.model.connect('indeps.x', 'paraboloid.x')
prob.model.connect('indeps.y', 'paraboloid.y')

prob.model.connect('indeps.x', 'con.x')
prob.model.connect('indeps.y', 'con.y')

prob.model.add_design_var('indeps.x', lower=-50, upper=50)
prob.model.add_design_var('indeps.y', lower=-50, upper=50)
prob.model.add_objective('paraboloid.f')
prob.model.add_constraint('con.c', lower=0.0)

# Create the ParOpt driver
prob.driver = ParOptDriver()

# Set options for the driver
prob.driver.options['optimizer'] = args.optimizer
prob.driver.options['output_file'] = 'paropt.out'
prob.driver.options['tr_output_file'] = 'tr_paropt.out'

# Run the problem
prob.setup()
prob.run_driver()

# Print the minimum value
print(prob['paraboloid.f'])

# Print the x/y location of the minimum
print(prob['indeps.x'])
print(prob['indeps.y'])
