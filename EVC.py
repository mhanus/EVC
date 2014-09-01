from dolfin import Function, norm
import numpy
from dolfin.cpp.la import PETScVector, PETScMatrix, KrylovSolver, PETScOptions
from dolfin.fem.interpolation import interpolate
from matrix_utils import create_composite_matrix, split_dofs, condense_matrices
from utils import MPI_sum0, comm, MPI_type
from scipy.sparse.linalg import lobpcg, eigs, eigsh

__author__ = 'mhanus'

class EVC(object):
  def __init__(self, problem,
               coarse_level_tol=0, coarse_level_maxit=None, coarse_level_num_ritz_vec=None, coarse_level_verb=0,
               use_lobpcg_if_symmetric=False):
    """

    :param Problem problem:
    :param bool use_lobpcg_if_symmetric:
    :return:
    """

    self.problem = problem

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
      # Coarse level solution (without Dirichlet BC dofs). Will be set in each call to `solve_at_coarse_level`.
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
    self.use_lobpcg_if_sym = use_lobpcg_if_symmetric
    self.coarse_level_maxit = coarse_level_maxit
    self.coarse_level_tol = coarse_level_tol

    if use_lobpcg_if_symmetric:
      self.coarse_level_verb = coarse_level_verb

      if self.coarse_level_tol == 0:
        self.coarse_level_tol = None
      if self.coarse_level_maxit is None:
        self.coarse_level_tol = 1000
    else:
      self.coarse_level_num_ritz_vec = coarse_level_num_ritz_vec

  def get_coarse_level_extensions(self, M_fine):
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


  def solve_at_coarse_level(self):
    if comm.rank == 0:
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

      if self.problem.sym:
        if self.use_lobpcg_if_sym:
          w, v = lobpcg(A, self.v_coarse, B, tol=self.coarse_level_tol, maxiter=self.coarse_level_maxit,
                        largest=largest, verbosityLevel=self.coarse_level_verb)
        else:
          w, v = eigsh(A, 1, B, which=which, v0=self.v_coarse,
                       ncv=self.coarse_level_num_ritz_vec, maxiter=self.coarse_level_maxit, tol=self.coarse_level_tol)
      else:
        w, v = eigs(A, 1, B, which=which, v0=self.v_coarse,
                    ncv=self.coarse_level_num_ritz_vec, maxiter=self.coarse_level_maxit, tol=self.coarse_level_tol)

      self.lam = w[0]
      self.v_coarse = v[0]


  def prolongate(self):
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

  def post_smooth(self, smooth_steps):
    for i in xrange(smooth_steps):
      if self.B_fine.size() > 0:
        b = self.B_fine.mult(self.vec_fine)
      else:
        b = self.vec_fine

      self.fine_solver.solve(self.vec_fine, b)

  def eigenvalue_residual_norm(self, norm_type='l2'):
    r = PETScVector()

    self.A_fine.mult(self.vec_fine, r)

    if self.B_fine.size(0) > 0:
      y = PETScVector()
      self.B_fine.mult(self.vec_fine, y)
    else:
      y = 1

    r -= self.lam*y

    return norm(r,norm_type)

  def solve(self, max_it, smooth_steps, res_norm_tol, norm_type='l2'):
    for i in xrange(max_it):
      self.update_coarse_level()
      self.solve_at_coarse_level()
      self.prolongate()
      self.post_smooth(smooth_steps)

      self.vec_fine *= 1./norm(self.vec_fine, norm_type)

      if self.eigenvalue_residual_norm(norm_type) <= res_norm_tol:
        break
