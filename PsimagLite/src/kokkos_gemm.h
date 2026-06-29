#ifndef PSIMAG_KOKKOS_GEMM_H
#define PSIMAG_KOKKOS_GEMM_H

#include <Kokkos_Core.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <complex>
#include <stdexcept>
#include <type_traits>

template <typename T>
struct KokkosType {
  using type = T;
};

template <typename T> requires (!std::is_floating_point_v<T>)
struct KokkosType<T> {
  using type = Kokkos::complex<typename T::value_type>;
};

template <typename Scalar, typename IntegerForBlasType>
inline void kokkos_gemm(char transa,
                        char transb,
                        IntegerForBlasType m,
                        IntegerForBlasType n,
                        IntegerForBlasType k,
                        const Scalar& alpha,
                        const Scalar* A,
                        IntegerForBlasType lda,
                        const Scalar* B,
                        IntegerForBlasType ldb,
                        const Scalar& beta,
                        Scalar* C,
                        IntegerForBlasType ldc)
{
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
        using KokkosScalar = KokkosType<Scalar>::type;

Kokkos::Serial exec;
Kokkos::HostSpace mem;

        Kokkos::View<KokkosScalar**, Kokkos::LayoutLeft, Kokkos::HostSpace> Aview_op("Aview", M, K);
        if (ta == 'N') {
            for (int i = 0; i < M; ++i)
                for (int kk = 0; kk < K; ++kk) {
                    auto val = A[i + kk * ldaVal];
                    Aview_op(i, kk) = val;
                }
        } else if (ta == 'T') {
            for (int i = 0; i < M; ++i)
                for (int kk = 0; kk < K; ++kk) {
                    auto val = A[kk + i * ldaVal];
                    Aview_op(i, kk) = val;
                }
        } else {
            KOKKOS_ASSERT(ta == 'C');
            for (int i = 0; i < M; ++i)
                for (int kk = 0; kk < K; ++kk) {
                    auto val = A[kk + i * ldaVal];
                    if constexpr(!std::is_floating_point_v<Scalar>) {
                      auto conjv = std::conj(val);
                      Aview_op(i, kk) = conjv;
                    } else {
                      Aview_op(i, kk) = val;
                    }
                }
        }
        auto Aview_op_device = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(exec, mem), Aview_op);

        Kokkos::View<KokkosScalar**, Kokkos::LayoutLeft, Kokkos::HostSpace> Bview_op("Bview", K, N);
        if (tb == 'N') {
            for (int kk = 0; kk < K; ++kk)
                for (int j = 0; j < N; ++j) {
                    auto val = B[kk + j * ldbVal];
                    Bview_op(kk, j) = val;
                }
        } else if (tb == 'T') {
            for (int kk = 0; kk < K; ++kk)
                for (int j = 0; j < N; ++j) {
                    auto val = B[j + kk * ldbVal];
                    Bview_op(kk, j) = val;
                }
        } else {
            KOKKOS_ASSERT(ta == 'C');
            for (int kk = 0; kk < K; ++kk)
                for (int j = 0; j < N; ++j) {
                    auto val = B[j + kk * ldbVal];
                    if constexpr(!std::is_floating_point_v<Scalar>) {
                      auto conjv = std::conj(val);
                      Bview_op(kk, j) = conjv;
                    } else {
                      Bview_op(kk, j) = val;
                    }
                }
        }
        auto Bview_op_device = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(exec, mem), Bview_op);

        Kokkos::View<KokkosScalar**, Kokkos::LayoutLeft, Kokkos::HostSpace> Cview("Cview", M, N);
/*        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                auto val = C[i + j * ldcVal];
                Cview(i, j) = val;
            }*/
        auto Cview_op_device = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(exec, mem), Cview_op);


        //KokkosScalar alphaC(std::real(alpha), std::imag(alpha));
        //KokkosScalar betaC(std::real(beta), std::imag(beta));
        const char transNN[2] = {'N', '\0'};
        KokkosBlas::gemm(exec, transNN, transNN, alpha, Aview_op_device, Bview_op_device, beta, Cview_device);
        Kokkos::deep_copy(exec, Cview, Cview_device);
        exec.fence();

        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                C[i + j * ldcVal] = Cview(i, j);

}

#endif // PSIMAG_KOKKOS_GEMM_H
