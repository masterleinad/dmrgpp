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
struct KokkosScalarTypePost { using type = T; };
template<typename T>
struct KokkosScalarTypePost<T, true> { using type = Kokkos::complex<typename T::value_type>; };
#endif

template <typename ComplexOrRealType>
void csr_matmul_post(char                                                       trans_A,
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
	 * compute   X +=  Y * op(A)
	 * where op(A) is transpose(A)   if trans_A = 'T' or 't'
	 *       op(A) is A              otherwise
	 *
	 * if need transpose(A) then
	 *   X(nrow_X,ncol_X) +=  Y(nrow_Y,ncol_Y) * tranpose(A(nrow_A,ncol_A))
	 *   requires (nrow_X == nrow_Y) && (ncol_Y == ncol_A) && (ncol_X == nrow_A)
	 *
	 * if need A then
	 *  X(nrow_X,ncol_X) +=  Y(nrow_Y,ncol_Y) * A(nrow_A,ncol_A)
	 *  requires  (nrow_X == nrow_Y) && ( ncol_Y == nrow_A) && (ncol_X == ncol_A)
	 * -------------------------------------------------------
	 */
	const bool is_complex      = PsimagLite::IsComplexNumber<ComplexOrRealType>::True;
	const int  nrow_A          = a.rows();
	int        isTranspose     = (trans_A == 'T') || (trans_A == 't');
	int        isConjTranspose = (trans_A == 'C') || (trans_A == 'c');
	int        isConj          = (trans_A == 'Z') || (trans_A == 'z');

#ifdef USE_KOKKOS
	// Use KokkosBatched if available, otherwise fallback to KokkosSparse
	using ExecSpace = Kokkos::DefaultExecutionSpace;
	ExecSpace exec;
	using Ordinal = int;

	using KokkosScalar = typename KokkosScalarTypePost<ComplexOrRealType, PsimagLite::IsComplexNumber<ComplexOrRealType>::True>::type;

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

	#if defined(__has_include)
	# if __has_include(<KokkosBatched_CrsMatrix.hpp>)
		// create host rank-2 values view (batch dim = 1), and host int views
		using ValuesHostViewType = Kokkos::View<KokkosScalar**, Kokkos::LayoutLeft, Kokkos::HostSpace>;
		using IntHostViewType = Kokkos::View<int*, Kokkos::HostSpace>;

		ValuesHostViewType h_values("h_values_p", 1, nnz);
		for (int k = 0; k < nnz; ++k) h_values(0, k) = vals[k];

		IntHostViewType h_rowptr("h_rowptr_p", nrow_A + 1);
		for (int i = 0; i <= nrow_A; ++i) h_rowptr(i) = rowptr[i];

		IntHostViewType h_cols("h_cols_p", nnz);
		for (int k = 0; k < nnz; ++k) h_cols(k) = cols[k];

		// copy to execution space
		auto d_values = Kokkos::create_mirror_view_and_copy(ExecSpace(), h_values);
		auto d_rowptr = Kokkos::create_mirror_view_and_copy(ExecSpace(), h_rowptr);
		auto d_cols = Kokkos::create_mirror_view_and_copy(ExecSpace(), h_cols);

		using ValuesDeviceViewType = decltype(d_values);
		using IntDeviceViewType = decltype(d_rowptr);
		KokkosBatched::CrsMatrix<ValuesDeviceViewType, IntDeviceViewType> A_crs(d_values, d_rowptr, d_cols);

		// For each column of Y perform spmv
		for (int iy = 0; iy < nrow_Y; ++iy) {
			std::vector<KokkosScalar> yhost((size_t)ncol_Y);
			for (int j = 0; j < ncol_Y; ++j) {
				if constexpr (!PsimagLite::IsComplexNumber<ComplexOrRealType>::True) {
					yhost[j] = yin(iy, j);
				} else {
					auto vv = yin(iy, j);
					if (is_complex && isConj && !(isTranspose || isConjTranspose)) vv = PsimagLite::conj(vv);
					yhost[j] = Kokkos::complex<typename ComplexOrRealType::value_type>(vv.real(), vv.imag());
				}
			}

			// Build host/device X/Y as rank-2 (batch dim = 1)
			using ValuesHostViewType = Kokkos::View<KokkosScalar**, Kokkos::LayoutLeft, Kokkos::HostSpace>;
			ValuesHostViewType h_X("h_X_p", 1, ncol_Y);
			for (int j = 0; j < ncol_Y; ++j) h_X(0, j) = yhost[j];
			ValuesHostViewType h_Y("h_Y_p", 1, ncol_X);
			for (int j = 0; j < ncol_X; ++j) h_Y(0, j) = KokkosKernels::ArithTraits<KokkosScalar>::zero();

			auto d_X = Kokkos::create_mirror_view_and_copy(ExecSpace(), h_X);
			auto d_Y = Kokkos::create_mirror_view_and_copy(ExecSpace(), h_Y);

			using MagType = typename KokkosKernels::ArithTraits<KokkosScalar>::mag_type;

			if (isTranspose || isConjTranspose) {
				A_crs.template apply<KokkosBatched::Trans::Transpose>(d_X, d_Y, MagType(1), MagType(0));
			} else {
				A_crs.template apply<KokkosBatched::Trans::NoTranspose>(d_X, d_Y, MagType(1), MagType(0));
			}

			auto h_Y_res = Kokkos::create_mirror_view(d_Y);
			Kokkos::deep_copy(h_Y_res, d_Y);
			exec.fence();

			for (int jx = 0; jx < ncol_X; ++jx) {
				if constexpr (!PsimagLite::IsComplexNumber<ComplexOrRealType>::True) {
					xout(iy, jx) += h_Y_res(0, jx);
				} else {
					Kokkos::complex<typename ComplexOrRealType::value_type> c = h_Y_res(0, jx);
					xout(iy, jx) += ComplexOrRealType(static_cast<typename ComplexOrRealType::value_type>(c.real()),
										 static_cast<typename ComplexOrRealType::value_type>(c.imag()));
				}
			}
		}
	# else
		// Fallback: construct KokkosSparse CrsMatrix from raw host arrays and use KokkosSparse::spmv per-column
		KokkosSparse::CrsMatrix<KokkosScalar, Ordinal, ExecSpace> A_crs("A_crs", nrow_A, (int)a.cols(), nnz,
		                                                              vals.data(), rowptr.data(), cols.data());

		for (int iy = 0; iy < nrow_Y; ++iy) {
			std::vector<KokkosScalar> yhost((size_t)ncol_Y);
			for (int j = 0; j < ncol_Y; ++j) {
				if constexpr (!PsimagLite::IsComplexNumber<ComplexOrRealType>::True) {
					yhost[j] = yin(iy, j);
				} else {
					auto vv = yin(iy, j);
					if (is_complex && isConj && !(isTranspose || isConjTranspose)) vv = PsimagLite::conj(vv);
					yhost[j] = Kokkos::complex<typename ComplexOrRealType::value_type>(vv.real(), vv.imag());
				}
			}

			auto y_dev = Kokkos::create_mirror_view_and_copy(ExecSpace(), Kokkos::View<const KokkosScalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(yhost.data(), ncol_Y));
			auto x_dev_out = Kokkos::View<KokkosScalar*>("x_dev_out", ncol_X);

			if (isTranspose || isConjTranspose) {
				KokkosSparse::spmv("T", (KokkosScalar)1.0, A_crs, y_dev, (KokkosScalar)0.0, x_dev_out);
			} else {
				KokkosSparse::spmv("N", (KokkosScalar)1.0, A_crs, y_dev, (KokkosScalar)0.0, x_dev_out);
			}

			std::vector<KokkosScalar> xhost((size_t)ncol_X);
			Kokkos::View<KokkosScalar*, Kokkos::HostSpace> h_xhost(xhost.data(), ncol_X);
			Kokkos::deep_copy(h_xhost, x_dev_out);
			exec.fence();

			for (int jx = 0; jx < ncol_X; ++jx) {
				if constexpr (!PsimagLite::IsComplexNumber<ComplexOrRealType>::True) {
					xout(iy, jx) += xhost[jx];
				} else {
					Kokkos::complex<typename ComplexOrRealType::value_type> c = xhost[jx];
					xout(iy, jx) += ComplexOrRealType(static_cast<typename ComplexOrRealType::value_type>(c.real()),
										 static_cast<typename ComplexOrRealType::value_type>(c.imag()));
				}
			}
		}
	# endif
	#else
		// No __has_include: fallback to KokkosSparse
		KokkosSparse::CrsMatrix<KokkosScalar, Ordinal, ExecSpace> A_crs("A_crs", nrow_A, (int)a.cols(), nnz,
		                                                              vals.data(), rowptr.data(), cols.data());

		for (int iy = 0; iy < nrow_Y; ++iy) {
			std::vector<KokkosScalar> yhost((size_t)ncol_Y);
			for (int j = 0; j < ncol_Y; ++j) {
				if constexpr (!PsimagLite::IsComplexNumber<ComplexOrRealType>::True) {
					yhost[j] = yin(iy, j);
				} else {
					auto vv = yin(iy, j);
					if (is_complex && isConj && !(isTranspose || isConjTranspose)) vv = PsimagLite::conj(vv);
					yhost[j] = Kokkos::complex<typename ComplexOrRealType::value_type>(vv.real(), vv.imag());
				}
			}

			auto y_dev = Kokkos::create_mirror_view_and_copy(ExecSpace(), Kokkos::View<const KokkosScalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(yhost.data(), ncol_Y));
			auto x_dev_out = Kokkos::View<KokkosScalar*>("x_dev_out", ncol_X);

			if (isTranspose || isConjTranspose) {
				KokkosSparse::spmv("T", (KokkosScalar)1.0, A_crs, y_dev, (KokkosScalar)0.0, x_dev_out);
			} else {
				KokkosSparse::spmv("N", (KokkosScalar)1.0, A_crs, y_dev, (KokkosScalar)0.0, x_dev_out);
			}

			std::vector<KokkosScalar> xhost((size_t)ncol_X);
			Kokkos::View<KokkosScalar*, Kokkos::HostSpace> h_xhost(xhost.data(), ncol_X);
			Kokkos::deep_copy(h_xhost, x_dev_out);
			exec.fence();

			for (int jx = 0; jx < ncol_X; ++jx) {
				if constexpr (!PsimagLite::IsComplexNumber<ComplexOrRealType>::True) {
					xout(iy, jx) += xhost[jx];
				} else {
					Kokkos::complex<typename ComplexOrRealType::value_type> c = xhost[jx];
					xout(iy, jx) += ComplexOrRealType(static_cast<typename ComplexOrRealType::value_type>(c.real()),
										 static_cast<typename ComplexOrRealType::value_type>(c.imag()));
				}
			}
		}
	#endif

	return;
#endif

	// fallback host implementation
	if (isTranspose || isConjTranspose) {
		assert(nrow_X == nrow_Y);
		assert(static_cast<SizeType>(ncol_Y) == a.cols() && (ncol_X == nrow_A));

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

				int iy = 0;
				for (iy = 0; iy < nrow_Y; iy++) {
					int ix = iy;
					int jx = ia;

					xout(ix, jx) += (yin(iy, ja) * atji);
				}
			}
		}
	} else {
		assert(nrow_X == nrow_Y);
		assert(ncol_Y == nrow_A && static_cast<SizeType>(ncol_X) == a.cols());

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

				int iy = 0;
				for (iy = 0; iy < nrow_Y; iy++) {
					int ix = iy;
					int jx = ja;

					xout(ix, jx) += (yin(iy, ia) * aij);
				}
			}
		}
	}
}
