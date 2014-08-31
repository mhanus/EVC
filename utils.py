__author__ = 'mhanus'

import os
from dolfin.cpp.common import dolfin_version

#
# MPI
#
from mpi4py.MPI import __TypeDict__, COMM_WORLD

MPI_type = lambda array: __TypeDict__[array.dtype.char]
comm = COMM_WORLD
pid = "Process " + str(comm.rank) + ": " if comm.size > 1 else ""

def print0(x, end="\n", _comm=COMM_WORLD):
  if _comm.rank == 0:
    print str(x)+end,


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

