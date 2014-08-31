#include "PETSc_utils_14.h"
#include <slepcversion.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/MPI.h>
#include <dolfin/parameter/GlobalParameters.h>
using namespace dolfin;

void COO::init(const PETScMatrix& mat)
{
	_matA = mat.mat();

	dolfin_assert(_matA);

	row_range = mat.local_range(0);
	local_size = row_range.second - row_range.first;

	MatInfo info;
	MatGetInfo(_matA, MAT_LOCAL, &info);
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
	ierr = MatGetRow(_matA, row, &ncols, &cols, &vals);
	if (ierr != 0) PETScObject::petsc_error(ierr, "PETSc_utils.cpp", "MatGetRow");

	// Insert values to std::vectors
	rows.insert(rows.end(), ncols, row);
	columns.insert(columns.end(), cols, cols + ncols);
	values.insert(values.end(), vals, vals + ncols);

	ierr = MatRestoreRow(_matA, row, &ncols, &cols, &vals);
	if (ierr != 0) PETScObject::petsc_error(ierr, "PETSc_utils.cpp", "MatRestoreRow");
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
	ierr = MatGetRow(_matA, row, &ncols, &cols, &vals);
	if (ierr != 0) PETScObject::petsc_error(ierr, "PETSc_utils_14.cpp", "MatGetRow");

	// Insert values to std::vectors
	for (int i = 0; i < ncols; i++)
	{
		rows.push_back(row*G + gto);
		columns.push_back(cols[i]*G + gfrom);
		values.push_back(negate ? -vals[i] : vals[i]);
	}

	ierr = MatRestoreRow(_matA, row, &ncols, &cols, &vals);
	if (ierr != 0) PETScObject::petsc_error(ierr, "PETSc_utils_14.cpp", "MatRestoreRow");
}
