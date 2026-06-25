#ifndef PSIMAG_KOKKOS_GEMM_H
#define PSIMAG_KOKKOS_GEMM_H

#include <Kokkos_Core.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <complex>
#include <stdexcept>
#include <type_traits>

// This header is intended to be included inside namespace psimag::BLAS
// It provides a single templated kokkos_gemm that handles real and
// complex scalar types (float,double,std::complex<float>,std::complex<double>).
// IntegerForBlasType is expected to be defined by the including header.

// Helper to detect std::complex and extract underlying real type
template <typename> struct is_std_complex : std::false_type {};
template <typename U> struct is_std_complex<std::complex<U>> : std::true_type { using value_type = U; };

template <typename T, typename IntegerForBlasType>
inline void kokkos_gemm(char transa,
                        char transb,
                        IntegerForBlasType m,
                        IntegerForBlasType n,
                        IntegerForBlasType k,
                        const T& alpha,
                        const T* A,
                        IntegerForBlasType lda,
                        const T* B,
                        IntegerForBlasType ldb,
                        const T& beta,
                        T* C,
                        IntegerForBlasType ldc)
{
    using Scalar = T;
    constexpr bool isComplex = is_std_complex<Scalar>::value;

    int M = static_cast<int>(m);
    int N = static_cast<int>(n);
    int K = static_cast<int>(k);
    int ldaVal = static_cast<int>(lda);
    int ldbVal = static_cast<int>(ldb);
    int ldcVal = static_cast<int>(ldc);

    // Normalize trans flags
    char ta = transa ? transa : 'N';
    char tb = transb ? transb : 'N';
    if (ta >= 'a' && ta <= 'z') ta = char(ta - 'a' + 'A');
    if (tb >= 'a' && tb <= 'z') tb = char(tb - 'a' + 'A');

    int req_lda = (ta == 'N') ? std::max(1, M) : std::max(1, K);
    int req_ldb = (tb == 'N') ? std::max(1, K) : std::max(1, N);
    int req_ldc = std::max(1, M);
    if (ldaVal < req_lda || ldbVal < req_ldb || ldcVal < req_ldc) {
        throw std::runtime_error("kokkos_gemm: invalid leading dimension");
    }

    // Determine Kokkos scalar type
    if constexpr (!isComplex) {
        using KokkosScalar = Scalar;

        Kokkos::View<KokkosScalar**, Kokkos::LayoutLeft, Kokkos::HostSpace> Aview_op("Aview", M, K);
        if (ta == 'N') {
            for (int i = 0; i < M; ++i)
                for (int kk = 0; kk < K; ++kk)
                    Aview_op(i, kk) = A[i + kk * ldaVal];
        } else {
            for (int i = 0; i < M; ++i)
                for (int kk = 0; kk < K; ++kk)
                    Aview_op(i, kk) = A[kk + i * ldaVal];
        }

        Kokkos::View<KokkosScalar**, Kokkos::LayoutLeft, Kokkos::HostSpace> Bview_op("Bview", K, N);
        if (tb == 'N') {
            for (int kk = 0; kk < K; ++kk)
                for (int j = 0; j < N; ++j)
                    Bview_op(kk, j) = B[kk + j * ldbVal];
        } else {
            for (int kk = 0; kk < K; ++kk)
                for (int j = 0; j < N; ++j)
                    Bview_op(kk, j) = B[j + kk * ldbVal];
        }

        Kokkos::View<KokkosScalar**, Kokkos::LayoutLeft, Kokkos::HostSpace> Cview("Cview", M, N);
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                Cview(i, j) = C[i + j * ldcVal];

        // KokkosBlas::gemm expects char arguments for transpose; use 'N' since data was pre-transposed
        const char transNN[2] = {'N', '\0'};
        KokkosBlas::gemm(transNN, transNN, alpha, Aview_op, Bview_op, beta, Cview);
        Kokkos::fence();

        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                C[i + j * ldcVal] = Cview(i, j);

    } else {
        // Complex path
        using Real = typename is_std_complex<Scalar>::value_type;
        using KokkosC = Kokkos::complex<Real>;

        Kokkos::View<KokkosC**, Kokkos::LayoutLeft, Kokkos::HostSpace> Aview_op("Aview_z", M, K);
        if (ta == 'N') {
            for (int i = 0; i < M; ++i)
                for (int kk = 0; kk < K; ++kk) {
                    auto val = A[i + kk * ldaVal];
                    Aview_op(i, kk) = KokkosC(std::real(val), std::imag(val));
                }
        } else if (ta == 'T') {
            for (int i = 0; i < M; ++i)
                for (int kk = 0; kk < K; ++kk) {
                    auto val = A[kk + i * ldaVal];
                    Aview_op(i, kk) = KokkosC(std::real(val), std::imag(val));
                }
        } else { // 'C' conjugate transpose
            for (int i = 0; i < M; ++i)
                for (int kk = 0; kk < K; ++kk) {
                    auto val = A[kk + i * ldaVal];
                    auto conjv = std::conj(val);
                    Aview_op(i, kk) = KokkosC(std::real(conjv), std::imag(conjv));
                }
        }

        Kokkos::View<KokkosC**, Kokkos::LayoutLeft, Kokkos::HostSpace> Bview_op("Bview_z", K, N);
        if (tb == 'N') {
            for (int kk = 0; kk < K; ++kk)
                for (int j = 0; j < N; ++j) {
                    auto val = B[kk + j * ldbVal];
                    Bview_op(kk, j) = KokkosC(std::real(val), std::imag(val));
                }
        } else if (tb == 'T') {
            for (int kk = 0; kk < K; ++kk)
                for (int j = 0; j < N; ++j) {
                    auto val = B[j + kk * ldbVal];
                    Bview_op(kk, j) = KokkosC(std::real(val), std::imag(val));
                }
        } else { // 'C'
            for (int kk = 0; kk < K; ++kk)
                for (int j = 0; j < N; ++j) {
                    auto val = B[j + kk * ldbVal];
                    auto conjv = std::conj(val);
                    Bview_op(kk, j) = KokkosC(std::real(conjv), std::imag(conjv));
                }
        }

        Kokkos::View<KokkosC**, Kokkos::LayoutLeft, Kokkos::HostSpace> Cview("Cview_z", M, N);
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                auto val = C[i + j * ldcVal];
                Cview(i, j) = KokkosC(std::real(val), std::imag(val));
            }

        KokkosC alphaC(std::real(alpha), std::imag(alpha));
        KokkosC betaC(std::real(beta), std::imag(beta));
        const char transNN[2] = {'N', '\0'};
        KokkosBlas::gemm(transNN, transNN, alphaC, Aview_op, Bview_op, betaC, Cview);
        Kokkos::fence();

        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                KokkosC v = Cview(i, j);
                C[i + j * ldcVal] = Scalar(v.real(), v.imag());
            }
    }
}

#endif // PSIMAG_KOKKOS_GEMM_H
