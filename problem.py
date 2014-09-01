from dolfin import DirichletBC, assemble_system, assemble
from dolfin import FunctionSpace
from dolfin.cpp.la import PETScMatrix
from dolfin.functions import TrialFunction, TestFunction, Constant
from dolfin.mesh.refinement import refine
from ufl import inner, grad, dx
from utils import pid, print0

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

    self.V_coarse = FunctionSpace(coarse_mesh, "CG", p_coarse)
    self.ndof_coarse = self.V_coarse.dim()

    print0("Refining mesh")

    refined_mesh = coarse_mesh
    for ref in xrange(nref):
      refined_mesh = refine(refined_mesh)   # creates a new Mesh, initial coarse mesh is unchanged

    self.V_fine = FunctionSpace(refined_mesh, "CG", p_fine)
    self.ndof_fine = self.V_fine.dim()

    print pid + "ndof_coarse = {}, ndof_fine = {}".format(self.ndof_coarse, self.ndof_fine)

    self.bc_coarse = None

    self.A_fine = PETScMatrix()
    self.B_fine = PETScMatrix()
    self.A_coarse = PETScMatrix()
    self.B_coarse = PETScMatrix()

    self.sym = sym
    self.switch_matrices_on_coarse_level = False

  def identity_at_coarse_level(self):
    I = PETScMatrix()
    u = TrialFunction(self.V_coarse)
    v = TestFunction(self.V_coarse)
    assemble(Constant(0)*u*v*dx, tensor=I)
    I.ident_zeros()
    return I

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
