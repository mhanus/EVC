import numpy
from dolfin.cpp.common import Timer
from dolfin.cpp.la import as_backend_type, PETScMatrix
import scipy.sparse as sp

from utils import *

__author__ = 'mhanus'

def composite_coo_on_zero(Apq, A11, A1j, Ai1, rows_glob=None, cols_glob=None, vals_glob=None):
  """ COO representation of matrix

              [ A11 |      A1j     ]
              [-----|--------------]
              [     |              ]
       Aij =  [     |              ]
              [ Ai1 |      Apq     ]
              [     |              ]
              [     |              ]

    on rank 0.

    :param PETScMatrix Apq: original matrix to be extended (must be square!)
    :param double A11:  new matrix block; needs to be specified only on rank 0
    :param ndarray A1j: new matrix block
    :param ndarray Ai1: new matrix block
    :param ndarray rows_glob:
    :param ndarray cols_glob:
    :param ndarray vals_glob:

    :return:
      rank 0:        rows, cols, vals
      rank 1,2,... : None, None, None
    :rtype: (ndarray, ndarray, ndarray)
  """

  timer = Timer("COO representation")

  # Remove zeros of the main block if possible

  try: # DOLFIN 1.3-
    A = as_backend_type(Apq.copy().compress())
  except:
    try:  # DOLFIN 1.4+
      A = as_backend_type(Apq.copy().compressed())
    except:
      A = Apq   # Don't compress

  # Create COO representation of the main block

  COO = backend_ext_module.COO(A)

  rows = COO.get_rows()
  cols = COO.get_cols()
  vals = COO.get_vals()
  n = COO.get_local_nrows()   # Apq must be square !!!

  # Create new blocks

  N = 2*n

  new_rows = numpy.empty(N, dtype=rows.dtype)
  """:type: ndarray"""
  new_cols = numpy.empty(N, dtype=cols.dtype)
  """:type: ndarray"""
  new_vals = numpy.empty(N, dtype=vals.dtype)
  """:type: ndarray"""

  # Shift existing rows/cols one down/right
  rows += 1
  cols += 1

  # Add new first col
  new_rows[:n] = numpy.arange(1,n+1,dtype=rows.dtype)
  new_cols[:n] = 0
  new_vals[:n] = Ai1

  # Add new first row
  new_rows[n:] = 0
  new_cols[n:] = numpy.arange(1,n+1,dtype=cols.dtype)
  new_vals[n:] = A1j

  # Extend the existing main block with the new ones
  local_nnz = COO.get_local_nnz() # just for sure (don't know how C++ vector from the ext_module will be transformed to
                                  # ndarray `rows`)
  rows = numpy.append(rows[:local_nnz], new_rows)
  vals = numpy.append(cols[:local_nnz], new_cols)
  cols = numpy.append(vals[:local_nnz], new_vals)

  local_nnz = vals.size

  recv_counts = comm.allgather(local_nnz)

  if comm.rank == 0 and not (rows_glob and cols_glob and vals_glob):
    nnz = int(numpy.sum(recv_counts))
    # Allow space for the new first element
    rows_glob = numpy.empty(nnz + 1, dtype=rows.dtype)
    cols_glob = numpy.empty(nnz + 1, dtype=cols.dtype)
    vals_glob = numpy.empty(nnz + 1, dtype=vals.dtype)

  MPI_IDX_T = MPI_type(rows)
  MPI_VAL_T = MPI_type(vals)

  recv_displs = None
  comm.Gatherv(sendbuf=[rows, MPI_IDX_T],
               recvbuf=[rows_glob, (recv_counts, recv_displs), MPI_IDX_T],
               root=0)
  comm.Gatherv(sendbuf=[cols, MPI_IDX_T],
               recvbuf=[cols_glob, (recv_counts, recv_displs), MPI_IDX_T],
               root=0)
  comm.Gatherv(sendbuf=[vals, MPI_VAL_T],
               recvbuf=[vals_glob, (recv_counts, recv_displs), MPI_VAL_T],
               root=0)

  # Add new first element

  if comm.rank == 0:
    rows_glob[-1] = 0
    cols_glob[-1] = 0
    vals_glob[-1] = A11

  return rows_glob, cols_glob, vals_glob

def condense_matrix(A, bc):
  """
  Condense the input matrix to inner dofs.

  :param scipy.sparse.csr_matrix A: (in/out) Matrix to be condensed.
  :param DirichletBC bc: (in) Dolfin object representing the zero Dirichlet BC.
  :return: rank 0: the condensed matrix, indices of inner and boundary dofs in the original matrix.
           rank 1,2,...: None objects
  :rtype: (scipy.sparse.csr_matrix, ndarray, ndarray) | (None, None, None)
  """

  bcdofs_loc = numpy.fromiter( bc.get_boundary_values().iterkeys(), dtype=numpy.int32 )
  recv_counts = comm.allgather(bcdofs_loc.size)

  if comm.rank == 0:
    bcdofs = numpy.empty(int(numpy.sum(recv_counts)), dtype=bcdofs_loc.dtype)
  else:
    bcdofs = None

  MPI_IDX_T = MPI_type(bcdofs_loc)
  recv_displs = None
  comm.Gatherv(sendbuf=[bcdofs_loc, MPI_IDX_T],
               recvbuf=[bcdofs, (recv_counts, recv_displs), MPI_IDX_T],
               root=0)

  if comm.rank == 0:
    alldofs = numpy.arange(A.shape[0], dtype=numpy.int32)
    indofs = numpy.setdiff1d(alldofs,bcdofs)
    A = A[numpy.ix_(indofs, indofs)]
  else:
    indofs = None
    A = None

  return A, bcdofs, indofs

def create_composite_matrix(Apq, A11, A1j, Ai1):
  """

  :param PETScMatrix Apq:
  :param double A11:
  :param ndarray A1j:
  :param ndarray Ai1:
  :return: Composite matrix on rank 0, None on other ranks
  :rtype: scipy.sparse.csr_matrix | None
  """

  r,c,v = composite_coo_on_zero(Apq, A11, A1j, Ai1)

  if comm.rank == 0:
    nnz = v.size

    ij = numpy.empty((2,nnz), r.dtype)
    ij[0,:] = r
    ij[1,:] = c

    A = sp.csr_matrix((v, ij))
  else:
    A = None

  return A
