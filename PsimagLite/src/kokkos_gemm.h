#ifndef PSIMAG_KOKKOS_GEMM_H
#define PSIMAG_KOKKOS_GEMM_H

#ifdef ENABLE_KOKKOS_GEMM

#include <Kokkos_Core.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <complex>
#include <stdexcept>

// This header is intended to be included inside namespace psimag::BLAS
// It provides inline kokkos_gemm overloads that mirror the GEMM
// signatures used by BLAS.h. IntegerForBlasType is expected to be
// defined by the including header.

// Double precision
template <typename IntegerForBlasType>
inline void kokkos_gemm(char transa,
                        char transb,
                        IntegerForBlasType m,
                        IntegerForBlasType n,
                        IntegerForBlasType k,
                        const double& alpha,
                        const double* A,
                        IntegerForBlasType lda,
                        const double* B,
                        IntegerForBlasType ldb,
                        const double& beta,
                        double* C,
                        IntegerForBlasType ldc)
{
    int M = static_cast<int>(m);
    int N = static_cast<int>(n);
    int K = static_cast<int>(k);
    double a = alpha;
    double b = beta;
    int ldaVal = static_cast<int>(lda);
    int ldbVal = static_cast<int>(ldb);
    int ldcVal = static_cast<int>(ldc);

    char ta = (transa) ? transa : 'N';
    char tb = (transb) ? transb : 'N';

    int req_lda = (ta == 'N' || ta == 'n') ? std::max(1, M) : std::max(1, K);
    int req_ldb = (tb == 'N' || tb == 'n') ? std::max(1, K) : std::max(1, N);
    int req_ldc = std::max(1, M);
    if (ldaVal < req_lda || ldbVal < req_ldb || ldcVal < req_ldc) {
        throw std::runtime_error("kokkos_gemm: invalid leading dimension");
    }

    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> Aview_op("Aview", M, K);
    if (ta == 'N' || ta == 'n') {
        for (int i = 0; i < M; ++i)
            for (int kk = 0; kk < K; ++kk)
                Aview_op(i, kk) = A[i + kk * ldaVal];
    } else {
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

    const char transNN[2] = {'N', '\0'};
    KokkosBlas::gemm(transNN, transNN, a, Aview_op, Bview_op, b, Cview);
    Kokkos::fence();

    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            C[i + j * ldcVal] = Cview(i, j);
}

// Complex double
template <typename IntegerForBlasType>
inline void kokkos_gemm(char transa,
                        char transb,
                        IntegerForBlasType m,
                        IntegerForBlasType n,
                        IntegerForBlasType k,
                        const std::complex<double>& alpha,
                        const std::complex<double>* A,
                        IntegerForBlasType lda,
                        const std::complex<double>* B,
                        IntegerForBlasType ldb,
                        const std::complex<double>& beta,
                        std::complex<double>* C,
                        IntegerForBlasType ldc)
{
    int M = static_cast<int>(m);
    int N = static_cast<int>(n);
    int K = static_cast<int>(k);
    int ldaVal = static_cast<int>(lda);
    int ldbVal = static_cast<int>(ldb);
    int ldcVal = static_cast<int>(ldc);

    char ta = (transa) ? transa : 'N';
    char tb = (transb) ? transb : 'N';
    if (ta >= 'a' && ta <= 'z') ta = ta - 'a' + 'A';
    if (tb >= 'a' && tb <= 'z') tb = tb - 'a' + 'A';

    int req_lda = (ta == 'N') ? std::max(1, M) : std::max(1, K);
    int req_ldb = (tb == 'N') ? std::max(1, K) : std::max(1, N);
    int req_ldc = std::max(1, M);
    if (ldaVal < req_lda || ldbVal < req_ldb || ldcVal < req_ldc) {
        throw std::runtime_error("kokkos_gemm: invalid leading dimension (complex)");
    }

    using KokkosC = Kokkos::complex<double>;
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
    } else {
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
    KokkosC alphaC(alpha.real(), alpha.imag());
    KokkosC betaC(beta.real(), beta.imag());

    KokkosBlas::gemm(transNN, transNN, alphaC, Aview_op, Bview_op, betaC, Cview);
    Kokkos::fence();

    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            KokkosC v = Cview(i, j);
            C[i + j * ldcVal] = std::complex<double>(v.real(), v.imag());
        }
}

// Float and complex-float fallbacks (simple CPU implementations)
template <typename IntegerForBlasType>
inline void kokkos_gemm(char transa,
                        char transb,
                        IntegerForBlasType m,
                        IntegerForBlasType n,
                        IntegerForBlasType k,
                        const float& alpha,
                        const float* A,
                        IntegerForBlasType lda,
                        const float* B,
                        IntegerForBlasType ldb,
                        const float& beta,
                        float* C,
                        IntegerForBlasType ldc)
{
    int M = static_cast<int>(m), N = static_cast<int>(n), K = static_cast<int>(k);
    int ldaVal = static_cast<int>(lda), ldbVal = static_cast<int>(ldb), ldcVal = static_cast<int>(ldc);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int kk = 0; kk < K; ++kk)
                sum += (double)A[i + kk * ldaVal] * (double)B[kk + j * ldbVal];
            C[i + j * ldcVal] = (float)(alpha * sum + beta * C[i + j * ldcVal]);
        }
}

template <typename IntegerForBlasType>
inline void kokkos_gemm(char transa,
                        char transb,
                        IntegerForBlasType m,
                        IntegerForBlasType n,
                        IntegerForBlasType k,
                        const std::complex<float>& alpha,
                        const std::complex<float>* A,
                        IntegerForBlasType lda,
                        const std::complex<float>* B,
                        IntegerForBlasType ldb,
                        const std::complex<float>& beta,
                        std::complex<float>* C,
                        IntegerForBlasType ldc)
{
    int M = static_cast<int>(m), N = static_cast<int>(n), K = static_cast<int>(k);
    int ldaVal = static_cast<int>(lda), ldbVal = static_cast<int>(ldb), ldcVal = static_cast<int>(ldc);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            std::complex<float> sum(0.0f, 0.0f);
            for (int kk = 0; kk < K; ++kk)
                sum += A[i + kk * ldaVal] * B[kk + j * ldbVal];
            C[i + j * ldcVal] = alpha * sum + beta * C[i + j * ldcVal];
        }
}

#endif // ENABLE_KOKKOS_GEMM

#endif // PSIMAG_KOKKOS_GEMM_H
