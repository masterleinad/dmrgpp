#include "util.h"
#ifdef USE_KOKKOS
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>
#ifdef USE_KOKKOSBATCHED
#include <KokkosBatched_SerialBlas.hpp>
#endif
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

	// Kokkos path: compute X += op(A) * Y using KokkosSparse::spmv per column of Y
	using HostExec = Kokkos::DefaultExecutionSpace;
	HostExec exec;
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
			vals[k] = Kokkos::complex<typename ComplexOrRealType::value_type>(v.real(), v.imag());
		}
	}

	// build CrsMatrix from raw host arrays; constructor will deep-copy to device
	{
		double maxAbs=0.0;
		for (int i=0;i<nnz;++i){
			double v = 0.0;
			if constexpr (!PsimagLite::IsComplexNumber<ComplexOrRealType>::True) v = std::abs((double)vals[i]);
			else v = std::abs((double)vals[i].real()) + std::abs((double)vals[i].imag());
			if (v>maxAbs) maxAbs=v;
		}
		(void)maxAbs;
	}
	KokkosSparse::CrsMatrix<KokkosScalar, Ordinal, HostExec> A_crs("A_crs", nrow_A, (int)a.cols(), nnz,
	                                                              vals.data(), rowptr.data(), cols.data());

	#ifdef USE_KOKKOSBATCHED
	{
		// Build device views (LayoutLeft so columns are contiguous) and helper arrays
		Kokkos::View<int*, HostExec> rowptr_dev("rowptr", nrow_A+1);
		Kokkos::View<int*, HostExec> cols_dev("cols", nnz);
		Kokkos::View<int*, HostExec> rindex_dev("rindex", nnz);
		Kokkos::View<KokkosScalar*, HostExec> vals_dev("vals", nnz);
		for (int i=0;i<=nrow_A;++i) rowptr_dev(i)=rowptr[i];
		for (int k=0;k<nnz;++k){ cols_dev(k)=cols[k]; vals_dev(k)=vals[k]; }
		// build rindex: row index for each nz
		for (int ia=0, k=0; ia<nrow_A; ++ia){ int ist=rowptr[ia]; int iend=rowptr[ia+1]; for (k=ist;k<iend;++k) rindex_dev(k)=ia; }

		Kokkos::View<KokkosScalar**, Kokkos::LayoutLeft, HostExec> Y_dev("Y_dev", nrow_Y, ncol_Y);
		Kokkos::View<KokkosScalar**, Kokkos::LayoutLeft, HostExec> X_dev("X_dev", nrow_X, ncol_X);
		// copy yin into Y_dev and existing xout into X_dev
		for (int i=0;i<nrow_Y;++i) for (int j=0;j<ncol_Y;++j) {
			if constexpr (std::is_floating_point<ComplexOrRealType>::value) Y_dev(i,j)=yin(i,j);
			else { auto vv=yin(i,j); Y_dev(i,j)=Kokkos::complex<typename ComplexOrRealType::value_type>(vv.real(),vv.imag()); }
		}
		for (int i=0;i<nrow_X;++i) for (int j=0;j<ncol_X;++j) {
			if constexpr (std::is_floating_point<ComplexOrRealType>::value) X_dev(i,j)=xout(i,j);
			else { auto vv=xout(i,j); X_dev(i,j)=Kokkos::complex<typename ComplexOrRealType::value_type>(vv.real(),vv.imag()); }
		}

		// For each nonzero, perform X_dev(row=ia, :) += vals[k] * Y_dev(row=ja, :)
		Kokkos::parallel_for("csr_pre_batched", Kokkos::RangePolicy<HostExec>(0, nnz), KOKKOS_LAMBDA(const int k){
			int ia = rindex_dev(k);
			int ja = cols_dev(k);
			KokkosScalar alpha = vals_dev(k);
			// length = ncol_Y, Y pointer = &Y_dev(ja,0), stride = nrow_Y; X pointer = &X_dev(ia,0), stride = nrow_X
			using SerialAxpy = KokkosBatched::SerialAxpy;
			SerialAxpy::invoke(ncol_Y, alpha, &Y_dev(ja,0), nrow_Y, &X_dev(ia,0), nrow_X);
		});
		exec.fence();

		// copy back X_dev into xout
		for (int i=0;i<nrow_X;++i) for (int j=0;j<ncol_X;++j){
			if constexpr (std::is_floating_point<ComplexOrRealType>::value) xout(i,j) = X_dev(i,j);
			else { auto c = X_dev(i,j); xout(i,j) = ComplexOrRealType(static_cast<typename ComplexOrRealType::value_type>(c.real()), static_cast<typename ComplexOrRealType::value_type>(c.imag())); }
		}
	}
	return;
	#endif

	const char trans = (isTranspose || isConjTranspose) ? 'T' : 'N';

	for (int jy = 0; jy < ncol_Y; ++jy) {
		// load ycol (length nrow_Y) from yin (note nrow_Y expected == a.cols() when trans)
		std::vector<KokkosScalar> yhost((size_t)nrow_Y);
		for (int i = 0; i < nrow_Y; ++i) {
			if constexpr (std::is_floating_point<ComplexOrRealType>::value) {
				yhost[i] = yin(i, jy);
			} else {
				auto vv = yin(i, jy);
				yhost[i] = Kokkos::complex<typename ComplexOrRealType::value_type>(vv.real(), vv.imag());
			}
		}

		auto x_dev_in = Kokkos::create_mirror_view_and_copy(HostExec(),
		                                                   Kokkos::View<const KokkosScalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(yhost.data(), nrow_Y));
			Kokkos::View<KokkosScalar*> x_dev_out("x_dev_out", nrow_X);

			// perform spmv
			if (trans == 'N') {
				KokkosSparse::spmv("N", (KokkosScalar)1.0, A_crs, x_dev_in, (KokkosScalar)0.0, x_dev_out);
			} else {
				KokkosSparse::spmv("T", (KokkosScalar)1.0, A_crs, x_dev_in, (KokkosScalar)0.0, x_dev_out);
			}

			// copy back and accumulate into xout
			std::vector<KokkosScalar> xhost((size_t)nrow_X);
			Kokkos::View<KokkosScalar*, Kokkos::HostSpace> h_xhost(xhost.data(), nrow_X);
			Kokkos::deep_copy(h_xhost, x_dev_out);
			exec.fence();

		for (int ix = 0; ix < nrow_X; ++ix) {
			if constexpr (std::is_floating_point<ComplexOrRealType>::value) {
				xout(ix, jy) += xhost[ix];
			} else {
				Kokkos::complex<typename ComplexOrRealType::value_type> c = xhost[ix];
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
