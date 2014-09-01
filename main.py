"""
This program demonstrates the use of SLEPc to
find the three smallest magnitude eigenvalues of the Laplacian
with Dirichlet boundary conditions on a square and the corresponding
eigenfunctions.
"""

import sys
from dolfin import Function
import numpy

from dolfin.cpp.common import has_slepc, warning, dolfin_error, Timer, timing, parameters, Table, timings, info
from dolfin.cpp.mesh import UnitSquareMesh
from dolfin.cpp.la import PETScOptions, SLEPcEigenSolver, has_linear_algebra_backend
from dolfin.cpp.io import File

# Test for PETSc and1 SLEPc
from evc_solver import EVCEigenvalueSolver
from problem import LaplaceEigenvalueProblem
from utils import comm, MPI_min0, MPI_max0, pid

if not has_linear_algebra_backend("PETSc"):
    print "DOLFIN has not been configured with PETSc. Exiting."
    sys.exit(0)

parameters.linear_algebra_backend = "PETSc"

#=======================================================================================================================

def solve_by_SLEPc(problem_def, tol, max_it):
  """
  
  :param Problem problem_def:
  :param double tol:
  :param int max_it
  :return:
  :rtype: (double, PETScVector, int)
  """

  if not has_slepc():
    warning("DOLFIN has not been configured with SLEPc. SLEPc solution will not be available.")
    return numpy.nan

  timer = Timer("SLEPc solution")

  #---------------------------------------------------------------------
  # Create eigensolver
  #
  type = "hermitian"
  if not problem_def.sym:
    type = "non_"+type

  if problem_def.B_fine.size(0) > 0:
    type = "gen_"+type

    if problem_def.switch_gep_matrices:
      eigensolver = SLEPcEigenSolver(problem_def.B_fine, problem_def.A_fine)
      eigensolver.parameters["spectrum"] = "smallest magnitude"
    else:
      eigensolver = SLEPcEigenSolver(problem_def.A_fine, problem_def.B_fine)
      eigensolver.parameters["spectrum"] = "largest magnitude"

  else:
    eigensolver = SLEPcEigenSolver(problem_def.A_fine)
    eigensolver.parameters["spectrum"] = "smallest magnitude"

  eigensolver.parameters["problem_type"] = type
  eigensolver.parameters["tolerance"] = tol
  eigensolver.parameters["maximum_iterations"] = max_it

  # Use the shift-and-invert spectral transformation
  #eigensolver.parameters["spectral_transform"] = "shift-and-invert"
  # Specify the shift
  #eigensolver.parameters["spectral_shift"] = 1.0e-10

  #---------------------------------------------------------------------
  # Compute the smallest eigenvalue.
  #
  eigensolver.solve(1)

  # Extract the eigenvalues (ignore the imaginary part) and compare with the exact values
  r, _, x, _ = eigensolver.get_eigenpair(0)

  timer.stop()

  sln = Function(problem_def.V_fine)
  sln.vector().set_local(x)
  File("eigfn_SLEPc.pvd") << sln

  return r, sln.vector(), eigensolver.get_iteration_number()

def solve_by_EVC(problem_def, tol, max_it):
  """

  :param Problem problem_def:
  :param double tol:
  :param int max_it
  :return:
  :rtype: (double, PETScVector, int)
  """

  timer = Timer("EVC solution")
  
  eigensolver = EVCEigenvalueSolver(problem_def)
  eigensolver.solve(max_it, tol)
  
  timer.stop()
  
  File("eigfn_EVC.pvd") << eigensolver.sln_fine

  return eigensolver.lam, eigensolver.vec_fine, eigensolver.num_it_fine_smoothing + eigensolver.num_it_coarse


def parse_parameters():
  switch_mat_setting = None

  if len(sys.argv) == 2:
    solver_args = [sys.argv[0]]
    try:
      with open (sys.argv[1], "r") as solver_params_file:
        for l in solver_params_file:
          s = l.strip().split('#')[0]
          if s:
            if s == 'eps_largest_magnitude':
              switch_mat_setting = True
            ss = s.split()
            solver_args.append("--petsc." + ss[0])
            solver_args.extend(ss[1:])

      parameters.parse(solver_args)

    except EnvironmentError as e:
      dolfin_error("main.py",
                   "load PETSc/SLEPc parameters"
                   "Solver parameters could not be read: {}.".format(e))

  elif len(sys.argv) > 2:
    parameters.parse(sys.argv)
    if '--petsc.eps_largest_magnitude' in sys.argv[1:]:
      switch_mat_setting = True

  return switch_mat_setting

if __name__ == "__main__":

  # Command line/file params
  switch_mat_setting_from_params = parse_parameters()

  # Additional PETSc params
  #PETScOptions.set("eps_view")

  # Global solver params
  tol = 1e-12
  max_it = 100

  # Coarse mesh params
  n = 4
  p_coarse = 1
  coarse_mesh = UnitSquareMesh(n, n)

  # Fine mesh params
  nref = 8
  p_fine = 1

  # Problem specification
  problem = LaplaceEigenvalueProblem(coarse_mesh, nref, p_coarse, p_fine)
  if switch_mat_setting_from_params is not None:
    if switch_mat_setting_from_params != problem.switch_gep_matrices:
      warning("Switch matrices setting mismatch. Command-line/file params get precedence.")
      problem.switch_gep_matrices = switch_mat_setting_from_params

  problem_spec_timings = timings(True)

  #---------------------------------------------------------------------
  # SLEPc solution
  #
  lam_slepc, vec_slepc, it_slepc = solve_by_SLEPc(problem, tol, max_it)
  res_slepc = problem.residual_norm(lam_slepc, vec_slepc)

  it_slepc_min = MPI_min0(it_slepc)
  it_slepc_max = MPI_max0(it_slepc)

  t_slepc = timing("SLEPc solution")
  t_slepc_min = MPI_min0(t_slepc)
  t_slepc_max = MPI_max0(t_slepc)

  #---------------------------------------------------------------------
  # EVC solution
  #
  lam_evc, vec_evc, it_evc = solve_by_EVC(problem, tol, max_it)
  res_evc = problem.residual_norm(lam_evc, vec_evc)

  it_evc_min = MPI_min0(it_evc)
  it_evc_max = MPI_max0(it_evc)

  t_evc = timing("EVC solution")
  t_evc_min = MPI_min0(t_evc)
  t_evc_max = MPI_max0(t_evc)


  #---------------------------------------------------------------------
  # Results
  #
  solver_int_timings = timings(True)

  problem_spec_timings_str = pid + "\n\n" + problem_spec_timings.str(True)
  problem_spec_timings_str = comm.gather(   problem_spec_timings_str, root=0)
  problem_spec_timings_str = "\n____________________________________________________\n".join(problem_spec_timings_str)
  
  solver_int_timings_str = pid + "\n\n" + solver_int_timings.str(True)
  solver_int_timings_str = comm.gather(   solver_int_timings_str, root=0)
  solver_int_timings_str = "\n____________________________________________________\n".join(solver_int_timings_str)
  
  if comm.rank == 0:
    res =  Table("Results")

    res.set("Exact", "eigenvalue", 2*numpy.pi*numpy.pi)
    res.set("Exact", "residual", 0)
    res.set("Exact", "time_min", 0)
    res.set("Exact", "time_max", 0)
    res.set("Exact", "iter_min", 0)
    res.set("Exact", "iter_max", 0)

    res.set("SLEPc", "eigenvalue", lam_slepc)
    res.set("SLEPc", "residual",   res_slepc)
    res.set("SLEPc", "time_min",     t_slepc_min)
    res.set("SLEPc", "time_max",     t_slepc_max)
    res.set("SLEPc", "iter_min",    it_slepc_min)
    res.set("SLEPc", "iter_max",    it_slepc_max)

    res.set("EVC", "eigenvalue", lam_evc)
    res.set("EVC", "residual",   res_evc)
    res.set("EVC", "time_min",     t_evc_min)
    res.set("EVC", "time_max",     t_evc_max)
    res.set("EVC", "iter_min",    it_evc_min)
    res.set("EVC", "iter_max",    it_evc_max)
    
    info(res)
    
    print
    print "----------------------------------------------------"
    print "        PROBLEM SPECIFICATION TIMINGS"
    print "____________________________________________________"
    print
    print problem_spec_timings

    print "----------------------------------------------------"
    print "            SOLVER INTERNAL TIMINGS"
    print "____________________________________________________"
    print
    print solver_int_timings
