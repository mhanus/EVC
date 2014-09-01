from dolfin import DirichletBC, assemble_system, assemble, norm, sqr
from dolfin import FunctionSpace
from dolfin.cpp.common import Table, info
from dolfin.cpp.la import PETScMatrix, PETScVector
from dolfin.functions import TrialFunction, TestFunction, Constant
from dolfin.mesh.refinement import refine
from math import log
import numpy
from prettytable import PrettyTable
from ufl import inner, grad, dx
from utils import pid, print0, comm, MPI_sum0, MPI_sum

__author__ = 'mhanus'

class Problem(object):
  def __init__(self, coarse_mesh, nref, p_coarse, p_fine, sym=False):
    """

    :param dolfin.cpp.mesh.Mesh coarse_mesh:
    :param int nref:
    :param int p_coarse:
    :param int p_fine:
    :param bool sym:
    :return:
    """

    print0("Creating approximation spaces")

    self.V_coarse = FunctionSpace(coarse_mesh, "CG", p_coarse)
    self.ndof_coarse = self.V_coarse.dim()

    refined_mesh = coarse_mesh
    for ref in xrange(nref):
      refined_mesh = refine(refined_mesh)   # creates a new Mesh, initial coarse mesh is unchanged

    self.V_fine = FunctionSpace(refined_mesh, "CG", p_fine)
    self.ndof_fine = self.V_fine.dim()

    H = coarse_mesh.hmax()
    h = refined_mesh.hmax()
    self.alpha = log(H)/log(h)
    self.beta = p_fine + 1

    if comm.rank == 0:
      prop = Table("Approximation properties")
      prop.set("ndof", "coarse", self.ndof_coarse)
      prop.set("ndof", "fine", self.ndof_fine)
      prop.set("h", "coarse", H)
      prop.set("h", "fine", h)

      info(prop)

      print "alpha = {}, beta = {}".format(self.alpha, self.beta)

    self.bc_coarse = None

    self.A_fine = PETScMatrix()
    self.B_fine = PETScMatrix()
    self.A_coarse = PETScMatrix()
    self.B_coarse = PETScMatrix()

    self.sym = sym
    self.switch_gep_matrices = False

  def identity_at_coarse_level(self):
    I = PETScMatrix()
    u = TrialFunction(self.V_coarse)
    v = TestFunction(self.V_coarse)
    assemble(Constant(0)*u*v*dx, tensor=I)
    I.ident_zeros()
    return I

  def residual_norm(self, vec, lam, norm_type='l2', A=None, B=None):
    if A is None:
      A = self.A_fine
      B = self.B_fine

    r = PETScVector()
    A.mult(vec, r)

    if B.size(0) > 0:
      y = PETScVector()
      B.mult(vec, y)
    else:
      y = 1

    r -= lam*y

    return norm(r,norm_type)

  def rayleigh_quotient(self, vec, A=None, B=None):
    if A is None:
      A = self.A_fine
      B = self.B_fine

    r = PETScVector()
    A.mult(vec, r)
    nom = MPI_sum( numpy.dot(r, vec) )

    if B.size(0) > 0:
      B.mult(vec, r)
      denom = MPI_sum( numpy.dot(r, vec) )
    else:
      denom = sqr(norm(r, norm_type='l2'))

    return nom/denom


class LaplaceEigenvalueProblem(Problem):
  def __init__(self, coarse_mesh, nref, p_coarse, p_fine):
    super(LaplaceEigenvalueProblem, self).__init__(coarse_mesh, nref, p_coarse, p_fine)

    print0("Assembling fine-mesh problem")

    self.dirichlet_bdry = lambda x,on_boundary: on_boundary

    bc = DirichletBC(self.V_fine, 0.0, self.dirichlet_bdry)
    u = TrialFunction(self.V_fine)
    v = TestFunction(self.V_fine)
    a = inner(grad(u), grad(v))*dx
    m = u*v*dx

    # Assemble the stiffness matrix and the mass matrix.
    b = v*dx # just need this to feed an argument to assemble_system
    assemble_system(a, b, bc, A_tensor=self.A_fine)
    assemble_system(m, b, bc, A_tensor=self.B_fine)
    # set the diagonal elements of M corresponding to boundary nodes to zero to
    # remove spurious eigenvalues.
    bc.zero(self.B_fine)

    print0("Assembling coarse-mesh problem")

    self.bc_coarse = DirichletBC(self.V_coarse, 0.0, self.dirichlet_bdry)
    u = TrialFunction(self.V_coarse)
    v = TestFunction(self.V_coarse)
    a = inner(grad(u), grad(v))*dx
    m = u*v*dx

    # Assemble the stiffness matrix and the mass matrix, without Dirichlet BCs. Dirichlet DOFs will be removed later.
    assemble(a, tensor=self.A_coarse)
    assemble(m, tensor=self.B_coarse)
