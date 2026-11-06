#include "execution.h"
#include "tensor.h"
#include "tensor_ops.h"

#include <cassert>
#include <cmath>
#include <vector>

void assertClose(
    const Tensor& left,
    const Tensor& right
) {
    assert(left.getShape() == right.getShape());

    for (int i = 0; i < left.size(); i++) {
        assert(std::abs(left.flat(i) - right.flat(i)) < 1e-12);
    }
}

void fill(Tensor& tensor, double offset) {
    for (int i = 0; i < tensor.size(); i++) {
        tensor.flat(i) = offset + 0.1 * (i + 1);
    }
}

void compareAll(
    const Tensor& A,
    const Tensor& B
) {
    execution::setParallelThreshold(1000000);
    Tensor serial = matmul(A, B);
    execution::setParallelThreshold(1);
    Tensor parallel = matmul(A, B);
    assertClose(serial, parallel);
}

int main() {
    Tensor A({3, 4});
    Tensor B({4, 5});
    fill(A, 0.0);
    fill(B, 1.0);
    compareAll(A, B);

    Tensor right_A({3, 4});
    Tensor right_B({5, 4});
    fill(right_A, 2.0);
    fill(right_B, 3.0);

    execution::setParallelThreshold(1000000);
    Tensor right_serial = matmul_transpose_right(right_A, right_B);
    execution::setParallelThreshold(1);
    Tensor right_parallel = matmul_transpose_right(right_A, right_B);
    assertClose(right_serial, right_parallel);

    Tensor left_A({4, 3});
    Tensor left_B({4, 5});
    fill(left_A, 4.0);
    fill(left_B, 5.0);

    execution::setParallelThreshold(1000000);
    Tensor left_serial = matmul_transpose_left(left_A, left_B);
    execution::setParallelThreshold(1);
    Tensor left_parallel = matmul_transpose_left(left_A, left_B);
    assertClose(left_serial, left_parallel);

    Tensor base_A({4, 3});
    Tensor base_B({5, 4});
    fill(base_A, 6.0);
    fill(base_B, 7.0);

    Tensor view_A = transpose(base_A);
    Tensor view_B = transpose(base_B);

    execution::setParallelThreshold(1000000);
    Tensor view_serial = matmul(view_A, view_B);
    execution::setParallelThreshold(1);
    Tensor view_parallel = matmul(view_A, view_B);
    assertClose(view_serial, view_parallel);

    Tensor right_base_A({4, 3});
    Tensor right_base_B({4, 5});
    fill(right_base_A, 8.0);
    fill(right_base_B, 9.0);
    Tensor right_view_A = transpose(right_base_A);
    Tensor right_view_B = transpose(right_base_B);

    execution::setParallelThreshold(1000000);
    Tensor right_view_serial =
        matmul_transpose_right(right_view_A, right_view_B);
    execution::setParallelThreshold(1);
    Tensor right_view_parallel =
        matmul_transpose_right(right_view_A, right_view_B);
    assertClose(right_view_serial, right_view_parallel);

    Tensor left_base_A({3, 4});
    Tensor left_base_B({5, 4});
    fill(left_base_A, 10.0);
    fill(left_base_B, 11.0);
    Tensor left_view_A = transpose(left_base_A);
    Tensor left_view_B = transpose(left_base_B);

    execution::setParallelThreshold(1000000);
    Tensor left_view_serial =
        matmul_transpose_left(left_view_A, left_view_B);
    execution::setParallelThreshold(1);
    Tensor left_view_parallel =
        matmul_transpose_left(left_view_A, left_view_B);
    assertClose(left_view_serial, left_view_parallel);

    execution::setParallelThreshold(16);
    Tensor small_serial = matmul(A, B);
    execution::setParallelThreshold(1);
    Tensor small_parallel = matmul(A, B);
    assertClose(small_serial, small_parallel);
}
