#include <iostream>
#include <vector>
#include <cassert>

class Matrix {
private:
	int rows;
	int cols;
	std::vector<double> data;

public:
	// constructor
	Matrix(int rows, int cols)
		: rows(rows), cols(cols), data(rows * cols, 0.0) {}

	double& operator()(int row, int col){
		return data[row * cols + col];
	}

	double operator()(int row, int col) const {
		return data[row * cols + col];
	}

	int getRows() const {
		return rows;
	}

	int getCols() const {
		return cols;
	}

	void print()  const {
		for (int i = 0; i < rows;  i++){
			for (int j = 0; j < cols;  j++){
				std::cout << (*this)(i , j ) << " ";
			}
			std::cout <<'\n';
		}
	}
};


Matrix matmul(const Matrix& A,  const Matrix& B) {

	assert(A.getCols() == B.getRows());

	Matrix result(A.getRows(),  B.getCols());

	for (int i = 0; i < A.getRows(); i++) {
		for (int j = 0; j < B.getCols(); j++) {
			for (int k = 0; k < A.getCols(); k++){
				result(i, j) += A(i, k) * B(k, j);
			}
		}
	}

	return result;
}



int main() {

    Matrix A(2, 3);
    Matrix B(3, 2);

    A(0, 0) = 1;
    A(0, 1) = 2;
    A(0, 2) = 3;

    A(1, 0) = 4;
    A(1, 1) = 5;
    A(1, 2) = 6;


    B(0, 0) = 7;
    B(0, 1) = 8;

    B(1, 0) = 9;
    B(1, 1) = 10;

    B(2, 0) = 11;
    B(2, 1) = 12;


    Matrix C = matmul(A, B);

    C.print();

    return 0;
}

