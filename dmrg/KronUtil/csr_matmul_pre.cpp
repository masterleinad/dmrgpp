#include "util.h"
#ifdef USE_KOKKOS
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <KokkosKernels_ArithTraits.hpp>
#include <KokkosBatched_Spmv.hpp>
#include <KokkosBatched_CrsMatrix.hpp>
#include <vector>
#include <type_traits>

// helper to pick Kokkos scalar type only when appropriate (file-scope)
template<typename T, bool IsComplex>
struct KokkosScalarType { using type = T; };
template<typename T>
struct KokkosScalarType<T, true> { using type = Kokkos::complex<typename T::value_type>; };
#endif

template <typename ComplexOrRealType>
void csr_matmul_pre(char                                                       trans_A,
                    const PsimagLite::CrsMatrix<ComplexOrRealType>&            a,
                    const int                                                  nrow_Y,
                    const int                                                  ncol_Y,
                    const PsimagLite::MatrixNonOwned<const ComplexOrRealType>& yin,
                    const int                                                  nrow_X,
                    const int                                                  ncol_X,
                    PsimagLite::MatrixNonOwned<ComplexOrRealType>&             xout)
{
	/*
	 * -------------------------------------------------------
	 * A in compressed sparse ROW format
	 *
	 * compute   X +=  op(A) * Y
	 * where op(A) is transpose(A)   if trans_A = 'T' or 't'
	 *       op(A) is A              otherwise
	 *
	 * if need transpose(A) then
	 *   X(nrow_X,ncol_X) +=  tranpose(A(nrow_A,ncol_A))*Y(nrow_Y,ncol_Y)
	 *   requires nrow_X == ncol_A, ncol_X == ncol_Y, nrow_A == nrow_Y
	 *
	 * if need A then
	 *  X(nrow_X,ncol_X) += A(nrow_A,ncol_A) * Y(nrow_Y,ncol_Y)
	 *  requires  nrow_X == nrow_A, ncol_A == nrow_Y, ncol_X == ncol_Y
	 * -------------------------------------------------------
	 */

	const bool is_complex      = PsimagLite::IsComplexNumber<ComplexOrRealType>::True;
	const int  nrow_A          = a.rows();
	int        isTranspose     = (trans_A == 'T') || (trans_A == 't');
	int        isConjTranspose = (trans_A == 'C') || (trans_A == 'c');
	int        isConj          = (trans_A == 'Z') || (trans_A == 'z');

#ifdef USE_KOKKOS
	// Kokkos path: compute X += op(A) * Y.
	using ExecSpace = Kokkos::DefaultExecutionSpace;
	ExecSpace exec;
	using Ordinal = int;

	using KokkosScalar = typename KokkosScalarType<ComplexOrRealType, PsimagLite::IsComplexNumber<ComplexOrRealType>::True>::type;

	const int nnz = a.nonZeros();

	// build host arrays
	std::vector<int> rowptr(nrow_A + 1);
	for (int i = 0; i <= nrow_A; ++i)
		rowptr[i] = a.getRowPtr(i);

	std::vector<int> cols(nnz);
	std::vector<KokkosScalar> vals(nnz);
	for (int k = 0; k < nnz; ++k) {
		cols[k] = a.getCol(k);
		if constexpr (!PsimagLite::IsComplexNumber<ComplexOrRealType>::True) {
			vals[k] = a.getValue(k);
		} else {
			ComplexOrRealType v = a.getValue(k);
			if (is_complex && (isConj || isConjTranspose)) v = PsimagLite::conj(v);
			vals[k] = Kokkos::complex<typename ComplexOrRealType::value_type>(v.real(), v.imag());
		}
	}

	// If KokkosBatched is available, use batched CrsMatrix; otherwise fall back to KokkosSparse spmv
	#if defined(__has_include)
	# if __has_include(<KokkosBatched_CrsMatrix.hpp>)
		// create host rank-2 values view (batch dim = 1), and host int views
		using ValuesHostViewType = Kokkos::View<KokkosScalar**, Kokkos::LayoutLeft, Kokkos::HostSpace>;
		using IntHostViewType = Kokkos::View<int*, Kokkos::HostSpace>;

		ValuesHostViewType h_values("h_values", 1, nnz);
		for (int k = 0; k < nnz; ++k) h_values(0, k) = vals[k];

		IntHostViewType h_rowptr("h_rowptr", nrow_A + 1);
		for (int i = 0; i <= nrow_A; ++i) h_rowptr(i) = rowptr[i];

		IntHostViewType h_cols("h_cols", nnz);
		for (int k = 0; k < nnz; ++k) h_cols(k) = cols[k];

		// copy to execution space
		auto d_values = Kokkos::create_mirror_view_and_copy(ExecSpace(), h_values);
		auto d_rowptr = Kokkos::create_mirror_view_and_copy(ExecSpace(), h_rowptr);
		auto d_cols = Kokkos::create_mirror_view_and_copy(ExecSpace(), h_cols);

		// construct batched CrsMatrix
		using ValuesDeviceViewType = decltype(d_values);
		using IntDeviceViewType = decltype(d_rowptr);
		KokkosBatched::CrsMatrix<ValuesDeviceViewType, IntDeviceViewType> A_crs(d_values, d_rowptr, d_cols);
	# else
		// Fallback: build KokkosSparse CrsMatrix from raw host arrays; constructor will deep-copy to device
		KokkosSparse::CrsMatrix<KokkosScalar, Ordinal, ExecSpace> A_crs("A_crs", nrow_A, (int)a.cols(), nnz,
		                                                              vals.data(), rowptr.data(), cols.data());
	# endif
	#else
		// No __has_include; fallback to KokkosSparse path
		KokkosSparse::CrsMatrix<KokkosScalar, Ordinal, ExecSpace> A_crs("A_crs", nrow_A, (int)a.cols(), nnz,
		                                                              vals.data(), rowptr.data(), cols.data());
	#endif

	// For each column of Y perform spmv: xcol = op(A) * ycol
	const char trans = (isTranspose || isConjTranspose) ? 'T' : 'N';
#endif

	for (int jy = 0; jy < ncol_Y; ++jy) {
		// load ycol (length nrow_Y) from yin (note nrow_Y expected == a.cols() when trans)
		std::vector<KokkosScalar> yhost((size_t)nrow_Y);
		for (int i = 0; i < nrow_Y; ++i) {
			if constexpr (std::is_floating_point<ComplexOrRealType>::value) {
				yhost[i] = yin(i, jy);
			} else {
				auto vv = yin(i, jy);
				if (is_complex && isConj && !(isTranspose || isConjTranspose)) vv = PsimagLite::conj(vv);
				yhost[i] = Kokkos::complex<typename ComplexOrRealType::value_type>(vv.real(), vv.imag());
			}
		}

				// Build rank-2 X and Y host views (batch dim = 1)
				int nX = (trans == 'N') ? static_cast<int>(a.cols()) : nrow_A;
				int nY = (trans == 'N') ? nrow_A : static_cast<int>(a.cols());

				ValuesHostViewType h_X("h_X", 1, nX);
				for (int j = 0; j < nX; ++j) h_X(0, j) = yhost[j];
				ValuesHostViewType h_Y("h_Y", 1, nY);
				for (int j = 0; j < nY; ++j) h_Y(0, j) = KokkosKernels::ArithTraits<KokkosScalar>::zero();

				// copy to device
				auto d_X = Kokkos::create_mirror_view_and_copy(ExecSpace(), h_X);
				auto d_Y = Kokkos::create_mirror_view_and_copy(ExecSpace(), h_Y);

				using MagType = typename KokkosKernels::ArithTraits<KokkosScalar>::mag_type;

				// perform apply (serial)
				if (trans == 'N') {
					A_crs.template apply<KokkosBatched::Trans::NoTranspose>(d_X, d_Y, MagType(1), MagType(0));
				} else {
					A_crs.template apply<KokkosBatched::Trans::Transpose>(d_X, d_Y, MagType(1), MagType(0));
				}

				// copy back and accumulate into xout
				auto h_Y_res = Kokkos::create_mirror_view(d_Y);
				Kokkos::deep_copy(h_Y_res, d_Y);
				exec.fence();

				for (int ix = 0; ix < nY; ++ix) {
					if constexpr (std::is_floating_point<ComplexOrRealType>::value) {
						xout(ix, jy) += h_Y_res(0, ix);
					} else {
						Kokkos::complex<typename ComplexOrRealType::value_type> c = h_Y_res(0, ix);
						xout(ix, jy) += ComplexOrRealType(static_cast<typename ComplexOrRealType::value_type>(c.real()),
											 static_cast<typename ComplexOrRealType::value_type>(c.imag()));
					}
				}
	}

	return;

	// fallback host implementation
	if (isTranspose || isConjTranspose) {
		assert(static_cast<SizeType>(nrow_X) == a.cols());
		assert(nrow_A == nrow_Y && ncol_X == ncol_Y);

		int ia = 0;
		for (ia = 0; ia < nrow_A; ia++) {
			int istart = a.getRowPtr(ia);
			int iend   = a.getRowPtr(ia + 1);
			int k      = 0;
			for (k = istart; k < iend; k++) {
				int               ja   = a.getCol(k);
				ComplexOrRealType aij  = a.getValue(k);
				ComplexOrRealType atji = aij;
				if (is_complex && isConjTranspose) {
					atji = PsimagLite::conj(atji);
				};

				int jy = 0;
				for (jy = 0; jy < ncol_Y; jy++) {
					int ix = ja;
					int jx = jy;
					xout(ix, jx) += (atji * yin(ia, jy));
				}
			}
		}
	} else {
		assert(nrow_X == nrow_A);
		assert(a.cols() == static_cast<SizeType>(nrow_Y) && (ncol_X == ncol_Y));

		int ia = 0;
		for (ia = 0; ia < nrow_A; ia++) {
			int istart = a.getRowPtr(ia);
			int iend   = a.getRowPtr(ia + 1);
			int k      = 0;
			for (k = istart; k < iend; k++) {
				int               ja  = a.getCol(k);
				ComplexOrRealType aij = a.getValue(k);
				if (is_complex && isConj) {
					aij = PsimagLite::conj(aij);
				};

				int jy = 0;

				for (jy = 0; jy < ncol_Y; jy++) {
					int ix = ia;
					int jx = jy;

					xout(ix, jx) += (aij * yin(ja, jy));
				}
			}
		}
	}
}
