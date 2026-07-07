#include "util.h"
#ifdef USE_KOKKOS
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>
//#ifdef USE_KOKKOSBATCHED
#include <KokkosBatched_Axpy.hpp>
#include <KokkosBatched_Gemv_Decl.hpp>
#include <KokkosBlas2_serial_gemv.hpp>
//#endif
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
Kokkos::Profiling::ScopedRegion region("matmulpost");
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
			if (is_complex && (isConj || isConjTranspose)) v = PsimagLite::conj(v);
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

#if 0//def USE_KOKKOSBATCHED
		// TeamPolicy batched implementation: build per-target lists to avoid atomics
		bool conjTrans = (isConjTranspose!=0);
		bool conjFlag = (isConj!=0);
		bool doTrans = (isTranspose!=0) || (isConjTranspose!=0);

		// build rindex: row index for each nz
		std::vector<int> rindex(nnz);
		for (int ia=0, kk=0; ia<nrow_A; ++ia) {
			int ist = rowptr[ia]; int iend = rowptr[ia+1];
			for (int k=ist;k<iend;++k) rindex[k]=ia;
		}

		int nTarget = ncol_X;
		std::vector<int> colCounts(nTarget,0);
		for (int k=0;k<nnz;++k) {
			int tgt = doTrans ? rindex[k] : cols[k];
			if (tgt>=0 && tgt < nTarget) ++colCounts[tgt];
		}
		std::vector<int> colPtr(nTarget+1,0);
		for (int i=0;i<nTarget;++i) colPtr[i+1] = colPtr[i] + colCounts[i];
		std::vector<int> idxList(nnz);
		std::vector<int> cur(colPtr.begin(), colPtr.end());
		for (int k=0;k<nnz;++k) {
			int tgt = doTrans ? rindex[k] : cols[k];
			int pos = cur[tgt]++;
			idxList[pos] = k;
		}

		// device views
		Kokkos::View<int*, HostExec> colPtr_dev("colPtr", nTarget+1);
		Kokkos::View<int*, HostExec> idx_dev("idx", nnz);
		Kokkos::View<int*, HostExec> cols_dev("cols_dev", nnz);
		Kokkos::View<int*, HostExec> rindex_dev("rindex", nnz);
		Kokkos::View<KokkosScalar*, HostExec> vals_dev("vals", nnz);
		for (int i=0;i<=nTarget;++i) colPtr_dev(i)=colPtr[i];
		for (int k=0;k<nnz;++k){ idx_dev(k)=idxList[k]; cols_dev(k)=cols[k]; rindex_dev(k)=rindex[k]; vals_dev(k)=vals[k]; }

		Kokkos::View<KokkosScalar**, Kokkos::LayoutLeft, HostExec> Y_dev("Y_dev", nrow_Y, ncol_Y);
		Kokkos::View<KokkosScalar**, Kokkos::LayoutLeft, HostExec> X_dev("X_dev", nrow_X, ncol_X);
		for (int i=0;i<nrow_Y;++i) for (int j=0;j<ncol_Y;++j) Y_dev(i,j) = yin(i,j);
		for (int i=0;i<nrow_X;++i) for (int j=0;j<ncol_X;++j) X_dev(i,j) = xout(i,j);

		// prepare vals_used (conjugated if needed)
		Kokkos::View<KokkosScalar*, HostExec> vals_used("vals_used", nnz);
		for (int k=0;k<nnz;++k) {
			if (doTrans) {
			if (conjTrans) {
				if constexpr (PsimagLite::IsComplexNumber<ComplexOrRealType>::True) vals_used(k) = Kokkos::conj(vals_dev(k));
				else vals_used(k) = vals_dev(k);
			} else vals_used(k) = vals_dev(k);
			} else {
			if (conjFlag) {
				if constexpr (PsimagLite::IsComplexNumber<ComplexOrRealType>::True) vals_used(k) = Kokkos::conj(vals_dev(k));
				else vals_used(k) = vals_dev(k);
			} else vals_used(k) = vals_dev(k);
			}
		}

		using team_policy = Kokkos::TeamPolicy<HostExec>;
		using member_type = typename team_policy::member_type;
		Kokkos::parallel_for("csr_post_batched", team_policy(nTarget, Kokkos::AUTO), KOKKOS_LAMBDA(const member_type &member){
			const int jc = member.league_rank();
			int start = colPtr_dev(jc);
			int end = colPtr_dev(jc+1);
			int m = end - start;
			if (m<=0) return;
			Kokkos::View<KokkosScalar**, Kokkos::LayoutLeft, HostExec> A_mat("A_mat", nrow_Y, m);
			Kokkos::View<KokkosScalar*, HostExec> alpha_vec("alpha", m);
			for (int p=0;p<m;++p) {
			int k = idx_dev(start+p);
			int src = doTrans ? cols_dev(k) : rindex_dev(k);
			alpha_vec(p) = vals_used(k);
			for (int irow=0;irow<nrow_Y;++irow) A_mat(irow,p) = Y_dev(irow, src);
			}
			Kokkos::View<KokkosScalar*, HostExec> y_out("y_out", nrow_Y);
			for (int irow=0;irow<nrow_Y;++irow) y_out(irow) = X_dev(irow, jc);
			KokkosBlas::SerialGemv<KokkosBlas::Trans::NoTranspose, KokkosBlas::Algo::Gemv::Unblocked>::invoke((KokkosScalar)1.0, A_mat, alpha_vec, (KokkosScalar)1.0, y_out);
			for (int irow=0;irow<nrow_Y;++irow) X_dev(irow, jc) = y_out(irow);
		});
		exec.fence();

		// copy back into xout
		for (int i=0;i<nrow_X;++i) for (int j=0;j<ncol_X;++j) xout(i,j) = X_dev(i,j);

		return;
#else
		// For each column of Y perform spmv
		for (int iy = 0; iy < nrow_Y; ++iy) {
			std::vector<KokkosScalar> yhost((size_t)ncol_Y);
			for (int j = 0; j < ncol_Y; ++j) {
			if constexpr (!PsimagLite::IsComplexNumber<ComplexOrRealType>::True) {
				yhost[j] = yin(iy, j);
			} else {
				auto vv = yin(iy, j);
				yhost[j] = Kokkos::complex<typename ComplexOrRealType::value_type>(vv.real(), vv.imag());
			}
			}

			auto y_dev = Kokkos::create_mirror_view_and_copy(HostExec(), Kokkos::View<const KokkosScalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(yhost.data(), ncol_Y));
			auto x_dev_out = Kokkos::View<KokkosScalar*>("x_dev_out", ncol_X);

			if (isTranspose || isConjTranspose) {
			KokkosSparse::spmv("N", (KokkosScalar)1.0, A_crs, y_dev, (KokkosScalar)0.0, x_dev_out);
			} else {
			KokkosSparse::spmv("T", (KokkosScalar)1.0, A_crs, y_dev, (KokkosScalar)0.0, x_dev_out);
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
				xout(iy, jx) += ComplexOrRealType(static_cast<typename ComplexOrRealType::value_type>(c.real()), static_cast<typename ComplexOrRealType::value_type>(c.imag()));
			}
			}
		}

		return;
#endif

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
