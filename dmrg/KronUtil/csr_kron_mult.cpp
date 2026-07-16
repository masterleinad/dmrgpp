#include "util.h"
#include <PsimagLite/KokkosType.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

template <typename ComplexOrRealType>
void csr_to_den(const PsimagLite::CrsMatrix<ComplexOrRealType>& a,
                PsimagLite::Matrix<ComplexOrRealType>&          a_)
{
	int ia = 0;
	int ja = 0;

	int nrow_A = a.row();
	int ncol_A = a.col();

	for (ja = 0; ja < ncol_A; ja++) {
		for (ia = 0; ia < nrow_A; ia++) {
			a_(ia, ja) = 0;
		};
	};

	for (ia = 0; ia < nrow_A; ia++) {
		int istarta = a.getRowPtr(ia);
		int ienda   = a.getRowPtr(ia + 1);
		int ka      = 0;
		for (ka = istarta; ka < ienda; ka++) {
			ComplexOrRealType aij = a.getValue(ka);
			int               ja  = a.getCol(ka);
			a_(ia, ja)            = aij;
		};
	};
}

template <typename ComplexOrRealType>
void csr_kron_mult_method(const int  imethod,
                          const char transA,
                          const char transB,

                          const PsimagLite::CrsMatrix<ComplexOrRealType>& a,

                          const PsimagLite::CrsMatrix<ComplexOrRealType>& b,

                          const PsimagLite::MatrixNonOwned<const ComplexOrRealType>& yin,
                          PsimagLite::MatrixNonOwned<ComplexOrRealType>&             xout)
{
	const bool is_complex   = PsimagLite::IsComplexNumber<ComplexOrRealType>::True;
	const int  isTransA     = (transA == 'T') || (transA == 't');
	const int  isTransB     = (transB == 'T') || (transB == 't');
	const int  isConjTransA = (transA == 'C') || (transA == 'c');
	const int  isConjTransB = (transB == 'C') || (transB == 'c');

	const int nrow_A = a.rows();
	const int ncol_A = a.cols();
	const int nrow_B = b.rows();
	const int ncol_B = b.cols();

	const int nrow_1 = (isTransA || isConjTransA) ? ncol_A : nrow_A;
	const int ncol_1 = (isTransA || isConjTransA) ? nrow_A : ncol_A;
	const int nrow_2 = (isTransB || isConjTransB) ? ncol_B : nrow_B;
	const int ncol_2 = (isTransB || isConjTransB) ? nrow_B : ncol_B;

	const int nrow_X = nrow_2;
	const int ncol_X = nrow_1;
	const int nrow_Y = ncol_2;
	const int ncol_Y = ncol_1;

	assert((imethod == 1) || (imethod == 2) || (imethod == 3));

	bool no_work = (csr_is_zeros(a) || csr_is_zeros(b));
	if (no_work) {
		return;
	};
	/*
	 *   -------------------------------------------------------------
	 *   A and B in compressed sparse ROW format
	 *
	 *   X += kron( op(A), op(B)) * Y
	 *   X +=  op(B) * Y * transpose(op(A))
	 *
	 *   nrow_X = nrow_2,   ncol_X = nrow_1
	 *   nrow_Y = ncol_2,   nrow_Y = nrow_2
	 *
	 *   that can be computed as either
	 *   imethod == 1
	 *
	 *   X(ib,ia) +=  (B(ib,jb) * Y( jb,ja)) * transpose( A(ia,ja))
	 *
	 *   X(ix,jx) +=  (B2(ib2,jb2) * Y(iy,jy)) * transpose(A1(ia1,ja1))
	 *
	 *
	 *
	 *   X(ib,ia) += (B(ib,jb) * Y(jb,ja) ) * transpose(A(ia,ja)   or
	 *               BY(ib,ja) = B(ib,jb)*Y(jb,ja)
	 *               BY is nrow_B by ncol_A, need   2*nnz(B)*ncolA flops
	 *
	 *   X(ib,ia) +=   BY(ib,ja) * transpose(A(ia,ja)) need 2*nnz(A)*nrowB flops
	 *
	 *   imethod == 2
	 *
	 *   X(ib,ia) += B(ib,jb) * (Y(jb,ja) * transpose(A))    or
	 *                YAt(jb,ia) = Y(jb,ja) * transpose(A(ia,ja))
	 *                YAt is ncolB by nrowA, need 2*nnz(A) * ncolB flops
	 *
	 *   X(ib,ia) += B(ib,jb) * YAt(jb,ia)  need nnz(B) * nrowA flops
	 *
	 *   imethod == 3
	 *
	 *   X += kron(A,B) * Y   by visiting all non-zero entries in A, B
	 *
	 *   this is feasible only if A and B are very sparse, need nnz(A)*nnz(B) flops
	 *   -------------------------------------------------------------
	 */

	if (imethod == 1) {

		/*
		 *  --------------------------------------------
		 *  BY(ib,ja) = (B(ib,jb))*Y(jb,ja)
		 *
		 *  X(ib,ia) += BY(ib,ja ) * transpose(A(ia,ja))
		 *  --------------------------------------------
		 */

		int                                                 nrow_BY = nrow_X;
		int                                                 ncol_BY = ncol_Y;
		PsimagLite::Matrix<ComplexOrRealType>               by_(nrow_BY, ncol_BY);
		PsimagLite::MatrixNonOwned<ComplexOrRealType>       byRef(by_);
		PsimagLite::MatrixNonOwned<const ComplexOrRealType> byConstRef(by_);
		/*
		 * ---------------
		 * setup BY
		 * ---------------
		 */

		{
			int iby = 0;
			int jby = 0;

			// not needed FIXME
			for (jby = 0; jby < ncol_BY; jby++) {
				for (iby = 0; iby < nrow_BY; iby++) {
					by_(iby, jby) = 0;
				};
			};
		}

		{
			/*
			 * ------------------------------
			 * BY(ib,ja)  = B(ib,jb)*Y(jb,ja)
			 * ------------------------------
			 */
			// const char trans = (isTransB) ? 'T' : 'N';
			const char trans = transB;
			csr_matmul_pre(trans,
			               b,

			               nrow_Y,
			               ncol_Y,
			               yin,

			               nrow_BY,
			               ncol_BY,
			               byRef);
		}

		{
			/*
			 * -------------------------------------------
			 * X(ib,ia) += BY(ib,ja) * transpose(A(ia,ja))
			 * -------------------------------------------
			 */
			/*
			 * ---------------------------------
			 * note trans = 'Z' mean use conj(A)
			 * ---------------------------------
			 */
			const char trans = isTransA ? 'N' : (isConjTransA ? 'Z' : 'T');
			csr_matmul_post(trans,
			                a,

			                nrow_BY,
			                ncol_BY,
			                byConstRef,

			                nrow_X,
			                ncol_X,
			                xout);
		}
	} else if (imethod == 2) {
		/*
		 * ---------------------
		 * YAt(jb,ia) = Y(jb,ja) * tranpose(A(ia,ja))
		 * X(ib,ia) += B(ib,jb) * YAt(jb,ia)
		 * ---------------------
		 */

		int                                                 nrow_YAt = nrow_Y;
		int                                                 ncol_YAt = ncol_X;
		PsimagLite::Matrix<ComplexOrRealType>               yat_(nrow_YAt, ncol_YAt);
		PsimagLite::MatrixNonOwned<ComplexOrRealType>       yatRef(yat_);
		PsimagLite::MatrixNonOwned<const ComplexOrRealType> yatConstRef(yat_);

		/*
		 * ----------------
		 * setup YAt(jb,ia)
		 * ----------------
		 */

		{
			int iy = 0;
			int jy = 0;

			// not needed FIXME
			for (jy = 0; jy < ncol_YAt; jy++) {
				for (iy = 0; iy < nrow_YAt; iy++) {
					yat_(iy, jy) = 0;
				};
			};
		}

		{
			/*
			 * ---------------------
			 * YAt(jb,ia) = Y(jb,ja) * tranpose(A(ia,ja)
			 * ---------------------
			 */
			/*
			 * ---------------------------------
			 * note trans = 'Z' mean use conj(A)
			 * ---------------------------------
			 */
			const char transa = isTransA ? 'N' : (isConjTransA ? 'Z' : 'T');
			csr_matmul_post(transa,
			                a,

			                nrow_Y,
			                ncol_Y,
			                yin,

			                nrow_YAt,
			                ncol_YAt,
			                yatRef);
		}

		{
			/*
			 * ------------
			 * X(ib,ia) += B(ib,jb) * YAt(jb,ia)
			 * ------------
			 */

			// const char trans = (isTransB) ? 'T' : 'N';
			const char trans = transB;

			csr_matmul_pre(trans,
			               b,

			               nrow_YAt,
			               ncol_YAt,
			               yatConstRef,

			               nrow_X,
			               ncol_X,
			               xout);
		}
	} else if (imethod == 3) {
		/*
		 * ---------------------------------------------
		 * Kokkos-parallel implementation of:
		 * C = kron(A,B)
		 * C([ib,ia], [jb,ja]) = A(ia,ja)*B(ib,jb)
		 * X([ib,ia]) += C([ib,ia],[jb,ja]) * Y([jb,ja])
		 * ---------------------------------------------
		 */

		Kokkos::Profiling::ScopedRegion region("PsimagLite::csr_kron_mult::imethod3");
		using ExecutionSpace = Kokkos::DefaultExecutionSpace;
		ExecutionSpace exec;
		using KokkosScalar = typename PsimagLite::KokkosType<ComplexOrRealType>::type;

		const int nnzA = a.nonZeros();
		const int nnzB = b.nonZeros();

		Kokkos::View<const int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> rowptrA_host(
		    &a.getRowPtr(0), nrow_A + 1);
		Kokkos::View<const int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> colsA_host(
		    &a.getCol(0), nnzA);
		Kokkos::View<const KokkosScalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
		    valsA_host(reinterpret_cast<const KokkosScalar*>(&a.getValue(0)), nnzA);

		Kokkos::View<const int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> rowptrB_host(
		    &b.getRowPtr(0), nrow_B + 1);
		Kokkos::View<const int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> colsB_host(
		    &b.getCol(0), nnzB);
		Kokkos::View<const KokkosScalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
		    valsB_host(reinterpret_cast<const KokkosScalar*>(&b.getValue(0)), nnzB);

		Kokkos::View<const KokkosScalar**,
		             Kokkos::LayoutLeft,
		             Kokkos::HostSpace,
		             Kokkos::MemoryUnmanaged>
		     yin_host(reinterpret_cast<const KokkosScalar*>(&yin(0, 0)), nrow_Y, ncol_Y);
		auto y_dev = Kokkos::create_mirror_view_and_copy(ExecutionSpace {}, yin_host);

		// Copy CSR arrays to device for A
		auto d_rowptrA
		    = Kokkos::create_mirror_view_and_copy(ExecutionSpace {}, rowptrA_host);
		auto d_colsA = Kokkos::create_mirror_view_and_copy(ExecutionSpace {}, colsA_host);
		auto d_valsA = Kokkos::create_mirror_view_and_copy(ExecutionSpace {}, valsA_host);

		// Prepare B arrays: if transpose or conj-transpose requested, form B_t on host
		bool needBtranspose = (isTransB || isConjTransB);
		int  nrow_B_used    = nrow_B;
		int  nnzB_used      = nnzB;

		Kokkos::View<const int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
		    rowptrB_used_host(nullptr, 0);
		Kokkos::View<const int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
		    colsB_used_host(nullptr, 0);
		Kokkos::View<const KokkosScalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
		    valsB_used_host(nullptr, 0);

		std::vector<int>               at_rowptr;
		std::vector<int>               at_col;
		std::vector<ComplexOrRealType> at_val;

		if (needBtranspose) {
			// build transpose of B on host
			nrow_B_used = ncol_B; // rows in B_t
			at_rowptr.resize(nrow_B_used + 1);
			at_col.resize(nnzB);
			at_val.resize(nnzB);

			csr_transpose<ComplexOrRealType>(nrow_B,
			                                 ncol_B,
			                                 &b.getRowPtr(0),
			                                 &b.getCol(0),
			                                 &b.getValue(0),
			                                 at_rowptr.data(),
			                                 at_col.data(),
			                                 at_val.data());

			// apply conjugation if needed
			if (isConjTransB) {
				for (int k = 0; k < nnzB; ++k)
					at_val[k] = PsimagLite::conj(at_val[k]);
			}

			rowptrB_used_host
			    = Kokkos::View<const int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
			        at_rowptr.data(), nrow_B_used + 1);
			colsB_used_host
			    = Kokkos::View<const int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
			        at_col.data(), nnzB);
			valsB_used_host = Kokkos::
			    View<const KokkosScalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
			        reinterpret_cast<const KokkosScalar*>(at_val.data()), nnzB);
		} else {
			rowptrB_used_host
			    = Kokkos::View<const int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
			        &b.getRowPtr(0), nrow_B + 1);
			colsB_used_host
			    = Kokkos::View<const int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
			        &b.getCol(0), nnzB);
			valsB_used_host = Kokkos::
			    View<const KokkosScalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
			        reinterpret_cast<const KokkosScalar*>(&b.getValue(0)), nnzB);
		}

		// create device mirrors for B_used

		auto d_rowptrB
		    = Kokkos::create_mirror_view_and_copy(ExecutionSpace {}, rowptrB_used_host);
		auto d_colsB
		    = Kokkos::create_mirror_view_and_copy(ExecutionSpace {}, colsB_used_host);
		auto d_valsB
		    = Kokkos::create_mirror_view_and_copy(ExecutionSpace {}, valsB_used_host);

		Kokkos::View<KokkosScalar**> x_dev_out("kron_x_dev_out", nrow_X, ncol_X);
		Kokkos::deep_copy(x_dev_out, static_cast<KokkosScalar>(0));

		// Use a TeamPolicy for higher throughput: one team per row of A
		using team_policy = Kokkos::TeamPolicy<ExecutionSpace>;
		using member_type = team_policy::member_type;
		team_policy policy(nrow_A, Kokkos::AUTO);

		Kokkos::parallel_for(
		    "csr_kron_mult::imethod3::team",
		    policy,
		    KOKKOS_LAMBDA(const member_type& team) {
			    const int ia      = team.league_rank();
			    int       istarta = d_rowptrA(ia);
			    int       ienda   = d_rowptrA(ia + 1);

			    bool jx_unique_per_team = !(isTransA || isConjTransA);

			    // For B_used, rows correspond to output ix. Parallelize over B_used
			    // rows (TeamThreadRange)
			    Kokkos::parallel_for(
			        Kokkos::TeamThreadRange(team, nrow_B_used),
			        [&](int r)
			        {
				        int istartb = d_rowptrB(r);
				        int iendb   = d_rowptrB(r + 1);

				        if (jx_unique_per_team) {
					        // jy depends on ka, so compute per-ka
					        // contributions; jx == ia unique per team For each
					        // ka compute contribution over B_used row and
					        // immediately add to x_dev_out(ix, jx)
					        for (int ka = istarta; ka < ienda; ++ka) {
						        int          ja  = d_colsA(ka);
						        KokkosScalar aij = d_valsA(ka);
						        if constexpr (is_complex)
							        if (isConjTransA)
								        aij = Kokkos::conj(aij);

						        KokkosScalar local_sum
						            = static_cast<KokkosScalar>(0);
						        Kokkos::parallel_reduce(
						            Kokkos::ThreadVectorRange(
						                team, istartb, iendb),
						            [&](int kb, KokkosScalar& lsum)
						            {
							            int c = d_colsB(
							                kb); // other index (iy)
							            KokkosScalar bij = d_valsB(kb);
							            KokkosScalar cij = aij * bij;
							            int          iy  = c;
							            int          jy
							                = (isTransA || isConjTransA)
							                ? ia
							                : ja;
							            lsum += cij * y_dev(iy, jy);
						            },
						            local_sum);

						        int ix = r; // since B_used rows map to
						                    // output ix
						        int jx = ia; // unique per team
						        // single writer per team for this jx -> use
						        // team single (no atomic)
						        Kokkos::single(
						            Kokkos::PerTeam(team),
						            [&]()
						            { x_dev_out(ix, jx) += local_sum; });
					        }
				        } else {
					        // general case: jx may be written by multiple teams
					        // -> need atomic per (ka,r)
					        for (int ka = istarta; ka < ienda; ++ka) {
						        int          ja  = d_colsA(ka);
						        KokkosScalar aij = d_valsA(ka);
						        if constexpr (is_complex)
							        if (isConjTransA)
								        aij = Kokkos::conj(aij);

						        KokkosScalar local_sum
						            = static_cast<KokkosScalar>(0);
						        Kokkos::parallel_reduce(
						            Kokkos::ThreadVectorRange(
						                team, istartb, iendb),
						            [&](int kb, KokkosScalar& lsum)
						            {
							            int          c   = d_colsB(kb);
							            KokkosScalar bij = d_valsB(kb);
							            KokkosScalar cij = aij * bij;
							            int          iy  = c;
							            int          jy
							                = (isTransA || isConjTransA)
							                ? ia
							                : ja;
							            lsum += cij * y_dev(iy, jy);
						            },
						            local_sum);

						        int ix = r;
						        int jx
						            = (isTransA || isConjTransA) ? ja : ia;
						        Kokkos::atomic_add(&x_dev_out(ix, jx),
						                           local_sum);
					        }
				        }
			        });
		    });

		auto xhost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {}, x_dev_out);
		for (int ix = 0; ix < nrow_X; ++ix) {
			for (int jx = 0; jx < ncol_X; ++jx)
				xout(ix, jx) += static_cast<ComplexOrRealType>(xhost(ix, jx));
		}
	};
}

template <typename ComplexOrRealType>
void csr_kron_mult_method(const int                                                   imethod,
                          const char                                                  transA,
                          const char                                                  transB,
                          const PsimagLite::CrsMatrix<ComplexOrRealType>&             a,
                          const PsimagLite::CrsMatrix<ComplexOrRealType>&             b,
                          const typename PsimagLite::Vector<ComplexOrRealType>::Type& yin_,
                          SizeType                                                    offsetY,
                          typename PsimagLite::Vector<ComplexOrRealType>::Type&       xout_,
                          SizeType                                                    offsetX)

{
	const int isTransA     = (transA == 'T') || (transA == 't');
	const int isTransB     = (transB == 'T') || (transB == 't');
	const int isConjTransA = (transA == 'C') || (transA == 'c');
	const int isConjTransB = (transB == 'C') || (transB == 'c');

	const int nrow_A = a.rows();
	const int ncol_A = a.cols();
	const int nrow_B = b.rows();
	const int ncol_B = b.cols();

	const int nrow_1 = (isTransA || isConjTransA) ? ncol_A : nrow_A;
	const int ncol_1 = (isTransA || isConjTransA) ? nrow_A : ncol_A;
	const int nrow_2 = (isTransB || isConjTransB) ? ncol_B : nrow_B;
	const int ncol_2 = (isTransB || isConjTransB) ? nrow_B : ncol_B;

	const int                                           nrow_X = nrow_2;
	const int                                           ncol_X = nrow_1;
	const int                                           nrow_Y = ncol_2;
	const int                                           ncol_Y = ncol_1;
	PsimagLite::MatrixNonOwned<const ComplexOrRealType> yin(nrow_Y, ncol_Y, yin_, offsetY);
	PsimagLite::MatrixNonOwned<ComplexOrRealType>       xout(nrow_X, ncol_X, xout_, offsetX);
	csr_kron_mult_method(imethod, transA, transB, a, b, yin, xout);
}

template <typename ComplexOrRealType>
void csr_kron_mult(const char                                                  transA,
                   const char                                                  transB,
                   const PsimagLite::CrsMatrix<ComplexOrRealType>&             a,
                   const PsimagLite::CrsMatrix<ComplexOrRealType>&             b,
                   const typename PsimagLite::Vector<ComplexOrRealType>::Type& yin,
                   SizeType                                                    offsetY,
                   typename PsimagLite::Vector<ComplexOrRealType>::Type&       xout,
                   SizeType                                                    offsetX,
                   const typename PsimagLite::Real<ComplexOrRealType>::Type    denseFlopDiscount)
{
	/*
	 *   -------------------------------------------------------------
	 *   A and B in compressed sparse ROW format
	 *
	 *   X += kron( A, B) * Y
	 *   that can be computed as either
	 *   imethod == 1
	 *
	 *   X(ib,ia) += (B(ib,jb) * Y(jb,ja) ) * transpose(A(ia,ja)   or
	 *               BY(ib,ja) = B(ib,jb)*Y(jb,ja)
	 *               BY is nrow_B by ncol_A, need   2*nnz(B)*ncolA flops
	 *
	 *   X(ib,ia) +=   BY(ib,ja) * transpose(A(ia,ja)) need 2*nnz(A)*nrowB flops
	 *
	 *   imethod == 2
	 *
	 *   X(ib,ia) += B(ib,jb) * (Y(jb,ja) * transpose(A))    or
	 *                YAt(jb,ia) = Y(jb,ja) * transpose(A(ia,ja))
	 *                YAt is ncolB by nrowA, need 2*nnz(A) * ncolB flops
	 *
	 *   X(ib,ia) += B(ib,jb) * YAt(jb,ia)  need nnz(B) * nrowA flops
	 *
	 *   imethod == 3
	 *
	 *   X += kron(A,B) * Y   by visiting all non-zero entries in A, B
	 *
	 *   this is feasible only if A and B are very sparse, need nnz(A)*nnz(B) flops
	 *   -------------------------------------------------------------
	 */
	int nnz_A = csr_nnz(a);
	int nnz_B = csr_nnz(b);

	bool no_work = (csr_is_zeros(a) || csr_is_zeros(b));
	if (no_work) {
		return;
	};

	ComplexOrRealType kron_nnz   = 0;
	ComplexOrRealType kron_flops = 0;
	int               imethod    = 1;

	const int isTransA     = (transA == 'T') || (transA == 't');
	const int isTransB     = (transB == 'T') || (transB == 't');
	const int isConjTransA = (transA == 'C') || (transA == 'c');
	const int isConjTransB = (transB == 'C') || (transB == 'c');

	const int nrow_A = a.rows();
	const int ncol_A = a.cols();
	const int nrow_B = b.rows();
	const int ncol_B = b.cols();

	// -----------------------------------
	// both A and B are considered sparse
	// -----------------------------------

	const int nrow_1 = (isTransA || isConjTransA) ? ncol_A : nrow_A;
	const int ncol_1 = (isTransA || isConjTransA) ? nrow_A : ncol_A;

	const int nrow_2 = (isTransB || isConjTransB) ? ncol_B : nrow_B;
	const int ncol_2 = (isTransB || isConjTransB) ? nrow_B : ncol_B;

	estimate_kron_cost(nrow_1,
	                   ncol_1,
	                   nnz_A,
	                   nrow_2,
	                   ncol_2,
	                   nnz_B,
	                   &kron_nnz,
	                   &kron_flops,
	                   &imethod,
	                   denseFlopDiscount);

	csr_kron_mult_method(imethod, transA, transB, a, b, yin, offsetY, xout, offsetX);
}
