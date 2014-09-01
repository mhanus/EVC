"""
This program demonstrates the use of SLEPc to
find the three smallest magnitude eigenvalues of the Laplacian
with Dirichlet boundary conditions on a square and the corresponding
eigenfunctions.
"""

from dolfin import *
import sys

from dolfin.cpp.common import has_slepc
from dolfin.cpp.mesh import UnitSquareMesh
from dolfin.cpp.la import PETScOptions, SLEPcEigenSolver
from dolfin.cpp.io import File


# Test for PETSc and1 SLEPc
from utils import comm

if not has_linear_algebra_backend("PETSc"):
    print "DOLFIN has not been configured with PETSc. Exiting."
    sys.exit(0)

if not has_slepc():
    print "DOLFIN has not been configured with SLEPc. Exiting."
    sys.exit(0)

parameters.linear_algebra_backend = "PETSc"

#================================================================================================================



def geteig(n, export_eigenfunction):
  """
  Compute the eigenvalues of the Laplacian with Dirichlet boundary conditions on the square.
  Use a mesh of n x n squares, divided into triangles, Lagrange elements of degree deg, and
  request nreq eigenpairs. If export_eigenfunctions=True, write the eigenfunctions to
  PVD files. Return values are the number of converged eigenpairs and the computed eigenvalues.
  """

  # Define basis and bilinear form
  mesh = UnitSquareMesh(n, n)


  # parse PETSc/SLEPc parameters from file -----------------------------

  solver_args = ["arg0"]
  try:
    with open (sys.argv[1], "r") as solver_params_file:
      for l in solver_params_file:
        s = l.strip().split('#')[0]
        if s:
          solver_args.append("--petsc." + s)
  except Exception as e:
    print "Solver parameters could not be read."
    print "Details: " + str(e)
    exit()

  parameters.parse(solver_args)

  # set additional PETSc/SLEPc parameters  -----------------------------

  #PETScOptions.set("eps_view")

  #---------------------------------------------------------------------
  # Create eigensolver
  eigensolver = SLEPcEigenSolver(A,B)

  # use the limited set of SLEPcEigenSolver wrapper paratmeters --------

  # Specify the part of the spectrum desired
  eigensolver.parameters["spectrum"] = "smallest magnitude"
  # Specify the problem type (this can make a big difference)
  eigensolver.parameters["problem_type"] = "gen_hermitian"
  # Use the shift-and-invert spectral transformation
  #eigensolver.parameters["spectral_transform"] = "shift-and-invert"
  # Specify the shift
  #eigensolver.parameters["spectral_shift"] = 1.0e-10

  #---------------------------------------------------------------------
  # Compute the smallest eigenvalue.

  eigensolver.solve(1)

  # Extract the eigenvalues (ignore the imaginary part) and compare with the exact values
  r, _, x, _ = eigensolver.get_eigenpair(0)

  # export the eigenfunctions for external visualization
  if export_eigenfunction:
    eigenfun = Function(V)
    eigenfun.vector().set_local(x)
    File("eigfn_SLEPc.pvd") << eigenfun

  return r


if __name__ == "__main__":
  exact = 2*pi*pi
  n = 25; export_eigenfunction = True
  lam = geteig(n, export_eigenfunction)

  if comm.rank == 0:
    print "Smallest positive eigenvalue and exact: "
      print lam, exact