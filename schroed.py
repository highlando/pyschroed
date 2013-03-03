from dolfin import *
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt


def func_and_propagate(IniVecReal,IntervalBorder):
	"""For now only Explicit Euler"""

	n_cycle = 4
	omega = 0.1
	t, T = 0, 2*pi*n_cycle / omega
	dt = 0.1

	mesh = IntervalMesh(len(IniVecReal)-1, -IntervalBorder, IntervalBorder)

	V = FunctionSpace(mesh,'CG',1)

	(ur, ui) = TrialFunctions(V*V)
	(vr, vi)  = TestFunctions(V*V )

	#ur, ui = split(u)
	#vr, vi = split(v)
	#ExpU0r = Expression('pow(2.71828,-(x[0]*x[0]))')

	# Define the Potential V(x)
	Vx = Expression('V0*(1-1.0/(1+pow(x[0]/Zr,2)))', V0 = 1, Zr = 1)

	# Define the perturbation F = x*F(t)
	F = Expression('x[0]* b_0 *pow(sin( omega*t*0.5/n_cycle ),2) *sin(omega*t +phi)',
			omega = omega, phi = 0, n_cycle = n_cycle, b_0 = 100, t = 0)

	# Initial value
	u0 = Function(V*V)

	u0rAux = Function(V)
	u0rAux.vector().set_local(IniVecReal)

	u0r, u0i = u0.split()
	u0r = u0rAux

	# Form for the explicit Euler time propagation
	a = -ui*vr*dx + ur*vi*dx
	# Form for the rhs
	L = -u0i*vr*dx + u0r*vi*dx \
	  + dt*(0.5*(inner(grad(u0r),grad(vr))*dx 
				+inner(grad(u0i),grad(vi))*dx)
		   + (Vx + F)*(u0r*vr + u0i*vi)*dx
		   )

	# current solution vector
	u1 = Function(V*V)
	NormUVec = []
	while t < T:
		# assemble the system for the current time step
		F.t = t
		A = assemble(a)
		rhs = assemble(L)

		# solve !
		solve(A,u1.vector(),rhs)

		# update the iterate
		u0.assign(u1)
		t += dt
		u1r, u1i = split(u1)
		NormUVec.append(norm(u1.vector()))

	plt.figure(1)
	plt.plot(NormUVec)
	plt.show(block=False)
	plot(u1r)
	interactive()
