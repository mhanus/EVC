import numpy

__author__ = 'mhanus'

import os
from dolfin.cpp.common import dolfin_version

#
# MPI
#
from mpi4py.MPI import __TypeDict__, COMM_WORLD, SUM, MAX, MIN

MPI_type = lambda array: __TypeDict__[array.dtype.char]
comm = COMM_WORLD
pid = "Process " + str(comm.rank) + ": " if comm.size > 1 else ""

def print0(x, end="\n", _comm=COMM_WORLD):
  if _comm.rank == 0:
    print str(x)+end,

def MPI_sum(arg,ax=None):
  if ax:
    r = numpy.atleast_1d(numpy.sum(arg,ax))
  else:
    r = numpy.atleast_1d(numpy.sum(arg))
  rout = numpy.zeros_like(r)

  comm.Allreduce([r, MPI_type(r)], [rout, MPI_type(rout)], op=SUM)

  if rout.size == 1:
    return rout[0]
  else:
    return rout

def MPI_sum0(arg,ax=None):
  if ax:
    r = numpy.atleast_1d(numpy.sum(arg,ax))
  else:
    r = numpy.atleast_1d(numpy.sum(arg))

  if comm.rank == 0:
    rout = numpy.zeros_like(r)
  else:
    rout = None

  comm.Reduce([r, MPI_type(r)], [rout, MPI_type(rout)], op=SUM, root=0)

  if rout.size == 1:
    return rout[0]
  else:
    return rout

def MPI_max0(arg,ax=None):
  if ax:
    r = numpy.atleast_1d(numpy.max(arg,ax))
  else:
    r = numpy.atleast_1d(numpy.max(arg))
  rout = numpy.zeros_like(r)

  comm.Reduce([r, MPI_type(r)], [rout, MPI_type(rout)], op=MAX, root=0)

  if rout.size == 1:
    return rout[0]
  else:
    return rout

def MPI_min0(arg,ax=None):
  if ax:
    r = numpy.atleast_1d(numpy.min(arg,ax))
  else:
    r = numpy.atleast_1d(numpy.min(arg))
  rout = numpy.zeros_like(r)

  comm.Reduce([r, MPI_type(r)], [rout, MPI_type(rout)], op=MIN, root=0)

  if rout.size == 1:
    return rout[0]
  else:
    return rout


#
# EXTENSION MODULES
#
dolfin_version_id = "".join(dolfin_version().split('.')[0:2])

_ext_module_versions = {"12" : "PETSc_utils_12", "13" : "PETSc_utils_12", "14" : "PETSc_utils_14"}

try:
  from dolfin.compilemodules.compilemodule import compile_extension_module

  _src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "cpp", "PETSc"))

  with open(os.path.join(_src_dir, _ext_module_versions[dolfin_version_id]+".h"), "r") as header:
    backend_ext_module = compile_extension_module(header.read(),
                                                  include_dirs=[".", _src_dir],
                                                  source_directory=_src_dir,
                                                  sources=[_ext_module_versions[dolfin_version_id]+".cpp"])

except EnvironmentError as e:
  print "Cannot open source files for PETSc extension module: {}".format(e)
  raise e
except Exception as e:
  print "Cannot initialize PETSc extension module"
  raise e

