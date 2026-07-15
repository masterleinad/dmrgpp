#include "util.h"

#include "PsimagLiteConfig.h"
#ifdef PSIMAGLITE_USE_KOKKOS
#include <KokkosBatched_Axpy.hpp>
#include <KokkosBatched_Gemv_Decl.hpp>
#include <KokkosBlas2_serial_gemv.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>
#include <type_traits>
#include <vector>

// helper to pick Kokkos scalar type only when appropriate (file-scope)
template <typename T, bool IsComplex> struct KokkosScalarTypePost {
	using type = T;
};
template <typename T> struct KokkosScalarTypePost<T, true> {
	using type = Kokkos::complex<typename T::value_type>;
};
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
#ifdef PSIMAGLITE_USE_KOKKOS
	Kokkos::Profiling::ScopedRegion region("PsimagLite::csr_matmul_post");
#endif
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

#ifdef PSIMAGLITE_USE_KOKKOS
	// Use KokkosSparse::spmv by transposing operations
	using ExecutionSpace = Kokkos::DefaultExecutionSpace;
	ExecutionSpace exec;
	using Ordinal = int;

	using KokkosScalar = typename KokkosScalarTypePost<
	    ComplexOrRealType,
	    PsimagLite::IsComplexNumber<ComplexOrRealType>::True>::type;

	const int nnz = a.nonZeros();

	// build host arrays
  Kokkos::View<const int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> rowptr_host(&a.getRowPtr(0), nrow_A + 1);
  Kokkos::View<const int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> cols_host(&a.getCol(0), nnz);
  Kokkos::View<KokkosScalar*, Kokkos::HostSpace> vals_host(Kokkos::view_alloc("vals_host", Kokkos::WithoutInitializing), nnz);
	for (int k = 0; k < nnz; ++k) {
			ComplexOrRealType v = a.getValue(k);
			if (is_complex && (isConj || isConjTranspose))
				v = PsimagLite::conj(v);
			vals_host[k] = v;
	}

    Kokkos::View<KokkosScalar**> x_dev_out("x_dev_out", nrow_Y, ncol_X);          
    Kokkos::View<const KokkosScalar**, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> yin_host(reinterpret_cast<const KokkosScalar*>(&yin(0,0)), nrow_Y, ncol_Y);
    auto y_dev = Kokkos::create_mirror_view_and_copy(ExecutionSpace{}, yin_host);

  // Copy CSR arrays to device so the TeamPolicy kernel can access them
  auto d_rowptr = Kokkos::create_mirror_view_and_copy(ExecutionSpace{}, rowptr_host);
  auto d_cols = Kokkos::create_mirror_view_and_copy(ExecutionSpace{}, cols_host);
  auto d_vals = Kokkos::create_mirror_view_and_copy(ExecutionSpace{}, vals_host);

{
#ifdef PSIMAGLITE_USE_KOKKOS
  Kokkos::Profiling::ScopedRegion region("PsimagLite::csr_matmul_post::kernel");
#endif

  using team_policy = Kokkos::TeamPolicy<ExecutionSpace>;
  using member_type = team_policy::member_type;

  // One team per output row iy; let Kokkos pick team/vector sizes
  team_policy policy(nrow_Y, Kokkos::AUTO);
  if (isTranspose || isConjTranspose) {
  Kokkos::parallel_for("csr_matmul_post::team_transpose", policy, KOKKOS_LAMBDA(const member_type& team) {
    const int iy = team.league_rank();

    // create a subview for the current Y row for faster access
    auto yrow = Kokkos::subview(y_dev, iy, Kokkos::ALL);

      // For transpose: X(iy,ia) += sum_j Y(iy,ja)*A(ia,ja)
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nrow_A), [&](int ia) {
        int istart = d_rowptr(ia);
        int iend = d_rowptr(ia + 1);
        KokkosScalar local_sum;
        // reduce contributions across the nonzeros of row ia
        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, istart, iend), [&](int k, KokkosScalar& lsum) {
          int ja = d_cols(k);
          KokkosScalar aij = d_vals(k);
          lsum += yrow(ja) * aij;
        }, local_sum);
        Kokkos::single(Kokkos::PerThread(team), [&] () { 
          x_dev_out(iy, ia) += local_sum;
        });
      });
    }); 
    }else {
     Kokkos::parallel_for("csr_matmul_post::team_no_transpose", policy, KOKKOS_LAMBDA(const member_type& team) {
      const int iy = team.league_rank();

     // create a subview for the current Y row for faster access
      auto yrow = Kokkos::subview(y_dev, iy, Kokkos::ALL);

      // Non-transpose: X(iy,ja) += sum_ia Y(iy,ia)*A(ia,ja)
      // Parallelize over ia (TeamThreadRange), then vectorize over nonzeros and use atomics for updates
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nrow_A), [&](int ia) {
        int istart = d_rowptr(ia);
        int iend = d_rowptr(ia + 1);
        KokkosScalar yval = yrow(ia);
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, istart, iend), [&](int k) {
          int ja = d_cols(k);
          KokkosScalar aij = d_vals(k);
          KokkosScalar prod = yval * aij;
            Kokkos::atomic_add(&x_dev_out(iy, ja), prod);
        });
      });
    });
  }

}
    auto xhost = Kokkos::create_mirror_view_and_copy(x_dev_out);
  for (int iy = 0; iy < nrow_Y; ++iy) {
    for (int jx = 0; jx < ncol_X; ++jx)
        xout(iy, jx) += static_cast<ComplexOrRealType>(xhost(iy, jx));
  }

	return;
#endif
	if (isTranspose || isConjTranspose) {
		/*
		 *   ----------------------------------------------------------
		 *   X(nrow_X,ncol_X) +=  Y(nrow_Y,ncol_Y) * transpose(A(nrow_A,ncol_A))
		 *   X(ix,jx) +=  Y(iy,jy) * transpose( A(ia,ja) )
		 *   X(ix,jx) += sum( Y(iy, ja) * At(ja,ia), over ja )
		 *   ----------------------------------------------------------
		 */

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
		/*
		 * ---------------------------------------------
		 * X(nrow_X,ncol_X) += Y(nrow_Y,ncol_Y) * A(nrow_A,ncol_A)
		 * X(ix,jx) += sum( Y(iy,ia) * A(ia,ja), over ia )
		 * ---------------------------------------------
		 */

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
