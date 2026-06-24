#ifdef ENABLE_KOKKOS_GEMM
#include <Kokkos_Core.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <complex>
#include <cstdlib>
#include <cstdio>

extern "C" {
// Fortran-style dgemm/cgemm wrappers (simplified signatures matching BLAS.h)
void dgemm_(const char* transa,
            const char* transb,
            const int* m,
            const int* n,
            const int* k,
            const double* alpha,
            const double* A,
            const int* lda,
            const double* B,
            const int* ldb,
            const double* beta,
            double* C,
            const int* ldc)
{
    int M = *m; int N = *n; int K = *k;
    double a = *alpha; double b = *beta;
    int ldaVal = *lda, ldbVal = *ldb, ldcVal = *ldc;

    // Determine trans flags (Fortran char may not be null-terminated)
    char ta = (transa && transa[0]) ? transa[0] : 'N';
    char tb = (transb && transb[0]) ? transb[0] : 'N';

    // Basic validation: leading dimensions depend on whether matrices are transposed
    int req_lda = (ta == 'N' || ta == 'n') ? std::max(1, M) : std::max(1, K);
    int req_ldb = (tb == 'N' || tb == 'n') ? std::max(1, K) : std::max(1, N);
    int req_ldc = std::max(1, M);
    if (ldaVal < req_lda || ldbVal < req_ldb || ldcVal < req_ldc) {
        fprintf(stderr, "kokkos_gemm: invalid leading dimension M=%d N=%d K=%d lda=%d(req %d) ldb=%d(req %d) ldc=%d(req %d)\n",
                M, N, K, ldaVal, req_lda, ldbVal, req_ldb, ldcVal, req_ldc);
        throw std::runtime_error("kokkos_gemm: invalid leading dimension");
    }

    // We'll materialize op(A) and op(B) into host LayoutLeft views so we can call KokkosBlas::gemm with 'N','N'.
    // If ta == 'N' then op(A) is MxK stored as-is; else op(A) is MxK obtained by transposing stored A (which is KxM logically).

    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> Aview_op("Aview", M, K);
    if (ta == 'N' || ta == 'n') {
        for (int i = 0; i < M; ++i)
            for (int kk = 0; kk < K; ++kk)
                Aview_op(i, kk) = A[i + kk * ldaVal];
    } else {
        // op(A) = A^T or A^H -> fill Aview_op(i,kk) = A[kk + i*lda]
        for (int i = 0; i < M; ++i)
            for (int kk = 0; kk < K; ++kk)
                Aview_op(i, kk) = A[kk + i * ldaVal];
    }

    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> Bview_op("Bview", K, N);
    if (tb == 'N' || tb == 'n') {
        for (int kk = 0; kk < K; ++kk)
            for (int j = 0; j < N; ++j)
                Bview_op(kk, j) = B[kk + j * ldbVal];
    } else {
        for (int kk = 0; kk < K; ++kk)
            for (int j = 0; j < N; ++j)
                Bview_op(kk, j) = B[j + kk * ldbVal];
    }

    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> Cview("Cview", M, N);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            Cview(i, j) = C[i + j * ldcVal];

    // Call KokkosBlas::gemm with non-transposed op(A)/op(B)
    const char transNN[2] = {'N', '\0'};
    KokkosBlas::gemm(transNN, transNN, a, Aview_op, Bview_op, b, Cview);
    Kokkos::fence();

    // Copy result back to Fortran-style column-major C
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            C[i + j * ldcVal] = Cview(i, j);
}

void zgemm_(const char* transa,
            const char* transb,
            const int* m,
            const int* n,
            const int* k,
            const std::complex<double>* alpha,
            const std::complex<double>* A,
            const int* lda,
            const std::complex<double>* B,
            const int* ldb,
            const std::complex<double>* beta,
            std::complex<double>* C,
            const int* ldc)
{
    int M = *m; int N = *n; int K = *k;
    std::complex<double> a = *alpha; std::complex<double> b = *beta;
    int ldaVal = *lda, ldbVal = *ldb, ldcVal = *ldc;

    // Determine trans flags early
    char ta = (transa && transa[0]) ? transa[0] : 'N';
    char tb = (transb && transb[0]) ? transb[0] : 'N';
    if (ta >= 'a' && ta <= 'z') ta = ta - 'a' + 'A';
    if (tb >= 'a' && tb <= 'z') tb = tb - 'a' + 'A';

    // Basic validation
    int req_lda = (ta == 'N') ? std::max(1, M) : std::max(1, K);
    int req_ldb = (tb == 'N') ? std::max(1, K) : std::max(1, N);
    int req_ldc = std::max(1, M);
    if (ldaVal < req_lda || ldbVal < req_ldb || ldcVal < req_ldc) {
        fprintf(stderr, "kokkos_gemm: invalid leading dimension z M=%d N=%d K=%d lda=%d(req %d) ldb=%d(req %d) ldc=%d(req %d)\n",
                M, N, K, ldaVal, req_lda, ldbVal, req_ldb, ldcVal, req_ldc);
        throw std::runtime_error("kokkos_zgemm: invalid leading dimension");
    }

    using KokkosC = Kokkos::complex<double>;

    // Materialize op(A) into MxK LayoutLeft view, handling conjugation if requested
    Kokkos::View<KokkosC**, Kokkos::LayoutLeft, Kokkos::HostSpace> Aview_op("Aview_z", M, K);
    if (ta == 'N' || ta == 'n') {
        for (int i = 0; i < M; ++i)
            for (int kk = 0; kk < K; ++kk) {
                std::complex<double> val = A[i + kk * ldaVal];
                Aview_op(i, kk) = KokkosC(val.real(), val.imag());
            }
    } else if (ta == 'T' || ta == 't') {
        for (int i = 0; i < M; ++i)
            for (int kk = 0; kk < K; ++kk) {
                std::complex<double> val = A[kk + i * ldaVal];
                Aview_op(i, kk) = KokkosC(val.real(), val.imag());
            }
    } else { // 'C' or conjugate-transpose
        for (int i = 0; i < M; ++i)
            for (int kk = 0; kk < K; ++kk) {
                std::complex<double> val = A[kk + i * ldaVal];
                std::complex<double> conjv = std::conj(val);
                Aview_op(i, kk) = KokkosC(conjv.real(), conjv.imag());
            }
    }

    Kokkos::View<KokkosC**, Kokkos::LayoutLeft, Kokkos::HostSpace> Bview_op("Bview_z", K, N);
    if (tb == 'N' || tb == 'n') {
        for (int kk = 0; kk < K; ++kk)
            for (int j = 0; j < N; ++j) {
                std::complex<double> val = B[kk + j * ldbVal];
                Bview_op(kk, j) = KokkosC(val.real(), val.imag());
            }
    } else if (tb == 'T' || tb == 't') {
        for (int kk = 0; kk < K; ++kk)
            for (int j = 0; j < N; ++j) {
                std::complex<double> val = B[j + kk * ldbVal];
                Bview_op(kk, j) = KokkosC(val.real(), val.imag());
            }
    } else {
        for (int kk = 0; kk < K; ++kk)
            for (int j = 0; j < N; ++j) {
                std::complex<double> val = B[j + kk * ldbVal];
                std::complex<double> conjv = std::conj(val);
                Bview_op(kk, j) = KokkosC(conjv.real(), conjv.imag());
            }
    }

    Kokkos::View<KokkosC**, Kokkos::LayoutLeft, Kokkos::HostSpace> Cview("Cview_z", M, N);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            std::complex<double> val = C[i + j * ldcVal];
            Cview(i, j) = KokkosC(val.real(), val.imag());
        }

    const char transNN[2] = {'N', '\0'};
    KokkosC alphaC(alpha->real(), alpha->imag());
    KokkosC betaC(beta->real(), beta->imag());

    KokkosBlas::gemm(transNN, transNN, alphaC, Aview_op, Bview_op, betaC, Cview);

    Kokkos::fence();

    // Copy result back to Fortran-style column-major C
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            KokkosC v = Cview(i, j);
            C[i + j * ldcVal] = std::complex<double>(v.real(), v.imag());
        }
}

// Lightweight float/complex wrappers forwarding to double versions (simple cast)
void sgemm_(const char* transa,const char* transb,const int* m,const int* n,const int* k,const float* alpha,const float* A,const int* lda,const float* B,const int* ldb,const float* beta,float* C,const int* ldc)
{
    // fallback to naive CPU version using doubles for simplicity
    int M=*m, N=*n, K=*k; double a=*alpha, b=*beta;
    for (int i=0;i<M;i++) for (int j=0;j<N;j++){
        double sum=0.0;
        for (int kk=0;kk<K;kk++) sum += (double)A[i + kk * (*lda)] * (double)B[kk + j * (*ldb)];
        C[i + j * (*ldc)] = (float)(a*sum + b*C[i + j * (*ldc)]);
    }
}

void cgemm_(const char* transa,const char* transb,const int* m,const int* n,const int* k,const std::complex<float>* alpha,const std::complex<float>* A,const int* lda,const std::complex<float>* B,const int* ldb,const std::complex<float>* beta,std::complex<float>* C,const int* ldc)
{
    int M=*m, N=*n, K=*k;
    for (int i=0;i<M;i++) for (int j=0;j<N;j++){
        std::complex<float> sum(0.0f,0.0f);
        for (int kk=0;kk<K;kk++) sum += A[i + kk * (*lda)] * B[kk + j * (*ldb)];
        C[i + j * (*ldc)] = (*alpha) * sum + (*beta) * C[i + j * (*ldc)];
    }
}

} // extern "C"
#endif // ENABLE_KOKKOS_GEMM
