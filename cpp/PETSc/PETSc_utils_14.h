#ifndef _PETSC_UTILS_H	// #pragma once doesn't work here (this whole file will be inserted into .cpp file)
#define _PETSC_UTILS_H

#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/la/SLEPcEigenSolver.h>
#include <dolfin/la/PETScOptions.h>

namespace dolfin {

  class COO
  {
    public:
			COO() {};
      COO(const PETScMatrix& mat);
      COO(const PETScMatrix& mat, int N, int gto, int gfrom, bool negate);

			std::vector<double> get_vals() const { return values; }
      std::vector<int> get_cols() const { return columns; }
      std::vector<int> get_rows() const { return rows; }
      std::size_t get_local_nnz() const { return local_nnz; }
      std::size_t get_local_nrows() const { return local_size; }

    private:
      void process_row(std::size_t row);

    protected:
      Mat _matA;
      std::pair<std::size_t,std::size_t> row_range;
      std::size_t local_size;
      std::size_t local_nnz;

      std::vector<double> values;
      std::vector<int> columns;
      std::vector<int> rows;

      void init(const PETScMatrix& mat);
  };

  class COO2 : public COO
  {
    public:
      COO2(const PETScMatrix& mat, int G, int gto, int gfrom, bool negate);

    private:
      void process_row(std::size_t row, int G, int gto, int gfrom, bool negate);
  };

}

#endif
