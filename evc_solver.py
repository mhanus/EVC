from dolfin import Function, norm
from dolfin.cpp.common import warning, Timer
import numpy
from dolfin.cpp.la import PETScVector, PETScMatrix, KrylovSolver, PETScOptions
from dolfin.fem.interpolation import interpolate
from matrix_utils import create_composite_matrix, split_dofs, condense_matrices
from utils import MPI_sum0, comm, MPI_type, print0, pid
from scipy.sparse.linalg import lobpcg, eigs, eigsh

try:
  from pyamg import smoothed_aggregation_solver
  __py_amg_available__ = True
except ImportError:
  __py_amg_available__ = False

__author__ = 'mhanus'

class EVCEigenvalueSolver(object):
  def __init__(self, problem,
               coarse_level_tol=0, coarse_level_maxit=None, coarse_level_num_ritz_vec=None,
               use_lobpcg_on_coarse_level=False, lobpcg_verb=0, precond_lobpcg_by_ml=False, update_lobpcg_prec=False,
               verbosity=1):
    """

    :param Problem problem:
    :param bool use_lobpcg_on_coarse_level:
    :return:
    """

    self.problem = problem
    self.verbosity = verbosity

    self.sln_fine = Function(problem.V_fine)
    self.vec_fine = self.sln_fine.vector()

    self.sln_coarse = Function(problem.V_coarse)
    self.vec_coarse = self.sln_coarse.vector()

    local_ndof = self.vec_coarse.local_size()
    self.local_ndof_all_proc = comm.allgather(local_ndof)

    self.A_fine = problem.A_fine
    self.B_fine = problem.B_fine
    self.A_coarse = problem.A_coarse
    self.B_coarse = problem.B_coarse

    # Prepare dof lists for handling Dirichlet boundary conditions.
    if problem.bc_coarse:
      self.bcdofs, self.indofs = split_dofs(problem.bc_coarse, problem.ndof_coarse)
      # Account for the extension
      self.bcdofs += 1
      self.indofs += 1
    else:
      self.bcdofs = numpy.array([], dtype=numpy.int32)
      self.indofs = numpy.s_[1:problem.ndof_coarse+1]

    if comm.rank == 0:
      # Coarse level solution (without Dirichlet BC dofs). Will be set in each call to `solve_on_coarse_level`.
      self.v_coarse = numpy.empty((problem.ndof_coarse+1,1))

      # Set the final coarse level solution (including the Dirichlet BC dofs, if any)
      self.v_coarse_with_all_dofs = numpy.zeros(problem.ndof_coarse)

    # Eigenvalue to be searched
    self.lam = 0

    # Fine-level solver
    self.fine_solver = KrylovSolver(self.A_fine, "cg" if self.problem.sym else "bicgstab", "ml_amg")
    #PETScOptions.set("pc_type", "ml")

    self.fs_param = self.fine_solver.parameters
    self.fs_param["preconditioner"]["structure"] = "same"
    self.fs_param["nonzero_initial_guess"] = True
    self.fs_param["monitor_convergence"] = False
    self.fs_param["relative_tolerance"] = 1e-6

    # Coarse level solver
    self.use_lobpcg_on_coarse_level = use_lobpcg_on_coarse_level
    self.coarse_level_maxit = coarse_level_maxit
    self.coarse_level_tol = coarse_level_tol

    if use_lobpcg_on_coarse_level:
      self.lobpcg_verb = lobpcg_verb

      if precond_lobpcg_by_ml:
        if not __py_amg_available__:
          warning("PyAMG preconditioning is not be available.")
          precond_lobpcg_by_ml = False

      self.precond_lobpcg_by_ml = precond_lobpcg_by_ml
      self.update_lobpcg_prec = update_lobpcg_prec
      self.M = None

      if not self.problem.sym:
          warning("LOBPCG will be used for non-symmetric eigenproblem.")

      if self.coarse_level_tol == 0:
        self.coarse_level_tol = None
      if self.coarse_level_maxit is None:
        self.coarse_level_tol = 1000
    else:
      self.coarse_level_num_ritz_vec = coarse_level_num_ritz_vec

    # Total iterations counters
    self.num_it_coarse = 0
    self.num_it_fine = 0
    self.num_it_fine_smoothing = 0
    self.num_it = 0

  def get_coarse_level_extensions(self, M_fine):
    if self.verbosity >= 3:
      print0(pid+"    calculating coarse matrices extensions")

    timer = Timer("Coarse matrices extensions")

    Mx_fun = Function(self.problem.V_fine)
    Mx_vec = Mx_fun.vector()
    M_fine.mult(self.vec_fine, Mx_vec)

    # M11
    xtMx = MPI_sum0( numpy.dot(self.vec_fine.get_local(), Mx_vec.get_local()) )

    # Mi1
    PtMx_fun = interpolate(Mx_fun, self.problem.V_coarse)
    PtMx = PtMx_fun.vector().get_local()

    # M1j
    if self.problem.sym:
      PtMtx = PtMx
    else:
      self.A_fine.transpmult(self.vec_fine, Mx_vec)
      PtMx_fun = interpolate(Mx_fun, self.problem.V_coarse)
      PtMtx = PtMx_fun.vector().get_local()

    return xtMx, PtMtx, PtMx

  def update_coarse_level(self):
    if self.verbosity >= 2:
      print0(pid+"  Updating coarse level matrices")

    timer = Timer("Coarse matrices update")

    # Get the coarse level extensions
    A11, A1j, Ai1 = self.get_coarse_level_extensions(self.A_fine)
    B11, B1j, Bi1 = self.get_coarse_level_extensions(self.B_fine)

    if isinstance(self.A_coarse, PETScMatrix):
      # Create the coarse level matrices for the first time

      assert self.A_coarse.size(0) == self.A_coarse.size(1) == self.problem.ndof_coarse
      self.A_coarse = create_composite_matrix(self.A_coarse, A11, A1j, Ai1)

      if self.B_coarse.size() == 0:
        assert self.B_fine.size() == 0
        self.B_coarse = self.problem.identity_at_coarse_level()

      self.B_coarse = create_composite_matrix(self.B_coarse, B11, B1j, Bi1)

      if self.problem.bc_coarse:
        self.A_coarse, self.B_coarse = condense_matrices(self.indofs, self.A_coarse, self.B_coarse)

    else:
      # Update the existing coarse level matrices
      if comm.rank == 0:
        self.A_coarse[0, 0] = A11
        self.A_coarse[0,1:] = A1j[self.indofs]
        self.A_coarse[1:,0] = Ai1[self.indofs]

        self.B_coarse[0, 0] = B11
        self.B_coarse[0,1:] = B1j[self.indofs]
        self.B_coarse[1:,0] = Bi1[self.indofs]

  def solve_on_coarse_level(self):

    if comm.rank == 0:
      if self.verbosity >= 2:
        print pid+"  Solving on coarse level"

      timer = Timer("Coarse level solution")

      if self.problem.switch_matrices_on_coarse_level:
        A = self.B_coarse
        B = self.A_coarse
        largest = True
        which = 'LM'
      else:
        A = self.A_coarse
        B = self.B_coarse
        largest = False
        which = 'SM'

      # Set initial approximation
      self.v_coarse.fill(0.0)
      self.v_coarse[0] = 1.0

      if self.use_lobpcg_on_coarse_level:
        if self.precond_lobpcg_by_ml:
          if self.update_lobpcg_prec or self.M is None:
            if self.verbosity >= 3:
              print0(pid+"    Creating coarse level preconditioner")

            ml = smoothed_aggregation_solver(A)
            self.M = ml.aspreconditioner()

        w, v, h = lobpcg(A, self.v_coarse, B, self.M, tol=self.coarse_level_tol, maxiter=self.coarse_level_maxit,
                         largest=largest, verbosityLevel=self.lobpcg_verb, retResidualNormsHistory=True)
      else:
        if self.problem.sym:
          w, v = eigsh(A, 1, B, which=which, v0=self.v_coarse,
                       ncv=self.coarse_level_num_ritz_vec, maxiter=self.coarse_level_maxit, tol=self.coarse_level_tol)
        else:
          w, v = eigs(A, 1, B, which=which, v0=self.v_coarse,
                      ncv=self.coarse_level_num_ritz_vec, maxiter=self.coarse_level_maxit, tol=self.coarse_level_tol)

      self.lam = w[0]
      self.v_coarse = v[0]

      try:
        self.num_it_coarse += len(h)
      except NameError:
        pass  # There seems to be no way to obtain number of iterations for eigs/eigsh

  def prolongate(self):
    if self.verbosity >= 2:
      print0(pid+"  Prolongating")

    timer = Timer("Prolongating")

    if comm.rank == 0:
      self.v_coarse_with_all_dofs[self.indofs] = self.v_coarse[1:]
      # 0 Dirichlet BC are automatic by the initial setting in the constructor

    x_coarse = self.vec_coarse.get_local()

    comm.Scatterv([self.v_coarse_with_all_dofs, (self.local_ndof_all_proc, None), MPI_type(self.v_coarse_with_all_dofs)],
                  [x_coarse, MPI_type(x_coarse)])

    self.vec_coarse.set_local(x_coarse)
    self.vec_coarse.apply("insert")
    self.vec_coarse.update_ghost_values()

    self.sln_fine = interpolate(self.sln_coarse, self.problem.V_fine)
    v1 = comm.bcast(self.v_coarse[0], root=0)
    self.vec_fine.axpy(v1, self.vec_fine)

  def smooth_on_fine_level(self, smoothing_steps):
    if self.verbosity >= 2:
      print0(pid+"  Smoothing on fine level")

    timer = Timer("Smoothing on fine level")

    for i in xrange(smoothing_steps):
      if self.verbosity >= 3:
        print0(pid+"    iteration {}".format(i))

      if self.B_fine.size() > 0:
        b = self.B_fine.mult(self.vec_fine)
      else:
        b = self.vec_fine

      self.num_it_fine += self.fine_solver.solve(self.vec_fine, b)

  def solve(self, max_it, res_norm_tol, smoothing_steps=None, norm_type='l2'):
    self.num_it_coarse = 0
    self.num_it_fine = 0

    if smoothing_steps is None:
      smoothing_steps = self.problem.beta/2+1

    for i in xrange(max_it):
      if self.verbosity >= 1:
        print0(pid+"Iteration {}".format(i))

      self.update_coarse_level()
      self.solve_on_coarse_level()
      self.prolongate()
      self.smooth_on_fine_level(smoothing_steps)

      self.vec_fine *= 1./norm(self.vec_fine, norm_type)

      if self.problem.residual_norm(norm_type) <= res_norm_tol:
        self.num_it = i+1
        break

    self.lam = self.problem.rayleigh_quotient(self.vec_fine)
    self.num_it_fine_smoothing = self.num_it * smoothing_steps