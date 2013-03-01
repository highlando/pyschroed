from dolfin import *

t, T = 0, 1
dt = 0.1

mesh = UnitInterval(32)

V = FunctionSpace(mesh,'CG',1)

(ur, ui) = TrialFunctions(V*V)
(vr, vi)  = TestFunctions(V*V )

#ur, ui = split(u)
#vr, vi = split(v)

Vx = Expression('V0*(1-1.0/(1+pow(x[0]/Zr,2)))', V0 = 1, Zr = 1)

F = Expression('x[0]* b_0 *pow(sin( omega*t*0.5/n_cycle ),2) *sin(omega*t +phi)',
		omega = 1, phi = 1, n_cycle = 1, b_0 = 1, t = 0)

u0 = Function(V*V)
u0r, u0i = u0.split()

a = -ui*vr*dx + ur*vi*dx

L = -u0i*vr*dx + u0r*vi*dx \
  + dt*(0.5*(inner(grad(u0r),grad(vr))*dx 
		    +inner(grad(u0i),grad(vi))*dx)
	   + Vx*(u0r*vr + u0i*vi)*dx
	   + F*vr*dx
	   )
u1 = Function(V*V)
while t < T:
	F.t = t
	A = assemble(a)
	rhs = assemble(L)
	solve(A,u1.vector(),rhs)
	#solve(a==L,u1)
	u0.assign(u1)
	t += dt
	u1r, u1i = split(u1)
	plot(u1r)
	interactive()

	

	




