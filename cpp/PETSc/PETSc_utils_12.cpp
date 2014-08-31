#include "PETSc_utils_12.h"
#include <slepcversion.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/MPI.h>
#include <dolfin/parameter/GlobalParameters.h>
using namespace dolfin;

void PETScMatrixExt::diag(boost::shared_ptr<dolfin::PETScVector> vec) const
{
	boost::shared_ptr<Vec> d = vec->vec();
	assert(d);
	MatGetDiagonal(*_A, *d);
}

std::size_t PETScMatrixExt::local_nnz() const
{
	MatInfo info;
	MatGetInfo(*_A, MAT_LOCAL, &info);
	return std::size_t(info.nz_used);
}

std::size_t PETScMatrixExt::global_nnz() const
{
	MatInfo info;
	MatGetInfo(*_A, MAT_GLOBAL_SUM, &info);
	return std::size_t(info.nz_used);
}

void PETScMatrixExt::mult(const PETScVector& xx, PETScVector& yy) const
{
  dolfin_assert(_A);

  if (PETScBaseMatrix::size(1) != xx.size())
  {
    dolfin_error("PETSc_utils_12.cpp",
                 "compute matrix-vector product with PETSc matrix",
                 "Non-matching dimensions for matrix-vector product");
  }

  // Resize RHS if empty
  if (yy.size() == 0)
  	PETScBaseMatrix::resize(yy, 0);

  if (PETScBaseMatrix::size(0) != yy.size())
  {
    dolfin_error("PETSc_utils_12.cpp",
                 "compute matrix-vector product with PETSc matrix",
                 "Vector for matrix-vector result has wrong size");
  }

  PetscErrorCode ierr = MatMult(*_A, *xx.vec(), *yy.vec());
  if (ierr != 0) PETScObject::petsc_error(ierr, __FILE__, "MatMult");
}

















void COO::init(const PETScMatrix& mat)
{
	_A = mat.mat();

	dolfin_assert(_A);

	row_range = mat.local_range(0);
	local_size = row_range.second - row_range.first;

	MatInfo info;
	MatGetInfo(*_A, MAT_LOCAL, &info);
	local_nnz = std::size_t(info.nz_used);

	rows.reserve(local_nnz);
	columns.reserve(local_nnz);
	values.reserve(local_nnz);
}

COO::COO(const PETScMatrix& mat)
{
	init(mat);

	for (std::size_t r = 0; r < local_size; r++)
		process_row(row_range.first+r);
}

COO::COO(const PETScMatrix& mat, int N, int gto, int gfrom, bool negate)
{
	init(mat);

	for (std::size_t r = 0; r < local_size; r++)
		process_row(row_range.first+r);

	std::transform(rows.begin(), rows.end(), rows.begin(), bind2nd(std::plus<int>(), N*gto));
	std::transform(columns.begin(), columns.end(), columns.begin(), bind2nd(std::plus<int>(), N*gfrom));
	if (negate)
		std::transform(values.begin(), values.end(), values.begin(), std::negate<double>());
}

void COO::process_row(std::size_t row)
{
	PetscErrorCode ierr;
	const PetscInt *cols = 0;
	const double *vals = 0;
	PetscInt ncols = 0;
	ierr = MatGetRow(*_A, row, &ncols, &cols, &vals);
	if (ierr != 0) PETScObject::petsc_error(ierr, "PETSc_utils_12.cpp", "MatGetRow");

	// Insert values to std::vectors
	rows.insert(rows.end(), ncols, row);
	columns.insert(columns.end(), cols, cols + ncols);
	values.insert(values.end(), vals, vals + ncols);

	ierr = MatRestoreRow(*_A, row, &ncols, &cols, &vals);
	if (ierr != 0) PETScObject::petsc_error(ierr, "PETSc_utils_12.cpp", "MatRestoreRow");
}














COO2::COO2(const PETScMatrix& mat, int G, int gto, int gfrom, bool negate)
{
	init(mat);

	for (std::size_t r = 0; r < local_size; r++)
		process_row(row_range.first+r, G, gto, gfrom, negate);
}

void COO2::process_row(std::size_t row, int G, int gto, int gfrom, bool negate)
{
	PetscErrorCode ierr;
	const PetscInt *cols = 0;
	const double *vals = 0;
	PetscInt ncols = 0;
	ierr = MatGetRow(*_A, row, &ncols, &cols, &vals);
	if (ierr != 0) PETScObject::petsc_error(ierr, "PETSc_utils_12.cpp", "MatGetRow");

	// Insert values to std::vectors
	for (int i = 0; i < ncols; i++)
	{
		rows.push_back(row*G + gto);
		columns.push_back(cols[i]*G + gfrom);
		values.push_back(negate ? -vals[i] : vals[i]);
	}

	ierr = MatRestoreRow(*_A, row, &ncols, &cols, &vals);
	if (ierr != 0) PETScObject::petsc_error(ierr, "PETSc_utils_12.cpp", "MatRestoreRow");
}

