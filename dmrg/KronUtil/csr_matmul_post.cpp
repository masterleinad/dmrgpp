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

	// Use KokkosSparse::spmv by transposing operations
	using HostExec = Kokkos::DefaultExecutionSpace;
	HostExec exec;
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
		// Build device views and helper arrays
		Kokkos::View<int*, HostExec> rowptr_dev("rowptr", nrow_A+1);
		Kokkos::View<int*, HostExec> cols_dev("cols", nnz);
		Kokkos::View<int*, HostExec> rindex_dev("rindex", nnz);
		Kokkos::View<KokkosScalar*, HostExec> vals_dev("vals", nnz);
		for (int i=0;i<=nrow_A;++i) rowptr_dev(i)=rowptr[i];
		for (int k=0;k<nnz;++k){ cols_dev(k)=cols[k]; vals_dev(k)=vals[k]; }
		for (int ia=0,k=0; ia<nrow_A; ++ia){ int ist=rowptr[ia]; int iend=rowptr[ia+1]; for (k=ist;k<iend;++k) rindex_dev(k)=ia; }

		Kokkos::View<KokkosScalar**, Kokkos::LayoutLeft, HostExec> Y_dev("Y_dev", nrow_Y, ncol_Y);
		Kokkos::View<KokkosScalar**, Kokkos::LayoutLeft, HostExec> X_dev("X_dev", nrow_X, ncol_X);
		// copy yin into Y_dev and existing xout into X_dev
		for (int i=0;i<nrow_Y;++i) for (int j=0;j<ncol_Y;++j){ if constexpr (!PsimagLite::IsComplexNumber<ComplexOrRealType>::True) Y_dev(i,j)=yin(i,j); else { auto vv=yin(i,j); Y_dev(i,j)=Kokkos::complex<typename ComplexOrRealType::value_type>(vv.real(),vv.imag()); } }
		for (int i=0;i<nrow_X;++i) for (int j=0;j<ncol_X;++j){ if constexpr (!PsimagLite::IsComplexNumber<ComplexOrRealType>::True) X_dev(i,j)=xout(i,j); else { auto vv=xout(i,j); X_dev(i,j)=Kokkos::complex<typename ComplexOrRealType::value_type>(vv.real(),vv.imag()); } }

		bool conjTrans = (isConjTranspose!=0);
		bool conjFlag = (isConj!=0);
		bool doTrans = (isTranspose!=0) || (isConjTranspose!=0);

		// For each nonzero: if op(A)==A -> X(:,ja) += aij * Y(:,ia)
		// if op(A)==transpose -> X(:,ia) += atji * Y(:,ja)
		Kokkos::parallel_for("csr_post_batched", Kokkos::RangePolicy<HostExec>(0, nnz), KOKKOS_LAMBDA(const int k){
			int ia = rindex_dev(k);
			int ja = cols_dev(k);
			KokkosScalar a = vals_dev(k);
			if (doTrans) {
			// atji = aij (conjugate if needed for conjTranspose)
			if (conjTrans) {
				if constexpr (PsimagLite::IsComplexNumber<ComplexOrRealType>::True) a = Kokkos::conj(a);
			}
			// X(:,ia) += atji * Y(:,ja)
			using SerialAxpy = KokkosBatched::SerialAxpy;
			SerialAxpy::invoke(nrow_Y, a, &Y_dev(0,ja), 1, &X_dev(0,ia), 1);
			} else {
			// op(A)==A, possibly conjugate aij when isConj
			if (conjFlag) {
				if constexpr (PsimagLite::IsComplexNumber<ComplexOrRealType>::True) a = Kokkos::conj(a);
			}
			// X(:,ja) += aij * Y(:,ia)
			using SerialAxpy = KokkosBatched::SerialAxpy;
			SerialAxpy::invoke(nrow_Y, a, &Y_dev(0,ia), 1, &X_dev(0,ja), 1);
			}
		});
		exec.fence();

		// copy back into xout
		for (int i=0;i<nrow_X;++i) for (int j=0;j<ncol_X;++j){ if constexpr (!PsimagLite::IsComplexNumber<ComplexOrRealType>::True) xout(i,j)=X_dev(i,j); else { auto c=X_dev(i,j); xout(i,j)=ComplexOrRealType(static_cast<typename ComplexOrRealType::value_type>(c.real()), static_cast<typename ComplexOrRealType::value_type>(c.imag())); } }

	}
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
