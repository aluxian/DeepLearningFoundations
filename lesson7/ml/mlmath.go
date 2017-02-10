package ml

import (
	"math"
)

// Sigmoid calculates 1/(1 + e^x).
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// SigmoidPrime calculates f​`(h)=f(h)(1−f(h)).
func SigmoidPrime(x float64) float64 {
	sigx := Sigmoid(x)
	return sigx * (1.0 - sigx)
}

// SigmoidM returns a new matrix with sigmoid applied to every element of the given matrix.
func SigmoidM(m [][]float64) [][]float64 {
	result := make([][]float64, len(m))
	for i, row := range m {
		result[i] = make([]float64, len(row))
		for j, x := range row {
			result[i][j] = Sigmoid(x)
		}
	}
	return result
}

// SigmoidPrimeM returns a new matrix with sigmoidPrime applied to every element of the given matrix.
func SigmoidPrimeM(m [][]float64) [][]float64 {
	result := make([][]float64, len(m))
	for i, row := range m {
		result[i] = make([]float64, len(row))
		for j, x := range row {
			result[i][j] = SigmoidPrime(x)
		}
	}
	return result
}

// Sum adds up all the elements in the array.
func Sum(xs []float64) float64 {
	sum := float64(0)
	for _, e := range xs {
		sum += e
	}
	return sum
}

// Mean computes the arithmetic mean of the array.
func Mean(xs []float64) float64 {
	return Sum(xs) / float64(len(xs))
}

// MeanM computes the arithmetic mean of the matrix.
func MeanM(m [][]float64) float64 {
	sum := float64(0)
	num := float64(0)

	for _, row := range m {
		for _, x := range row {
			sum += x
			num++
		}
	}

	return sum / num
}

// Std computes the standard deviation of the array, std = sqrt(mean(abs(x-x.mean()) ** 2)).
func Std(xs []float64) float64 {
	xmean := Mean(xs)
	ys := make([]float64, len(xs))
	for i, x := range xs {
		ys[i] = math.Pow(math.Abs(x-xmean), 2)
	}
	return math.Sqrt(Mean(ys))
}

// Standardize uses mean() and std() to return a new, standardized array.
func Standardize(xs []float64) []float64 {
	mean := Mean(xs)
	std := Std(xs)
	ys := make([]float64, len(xs))
	for i, x := range xs {
		ys[i] = (x - mean) / std
	}
	return ys
}

// Dot calculates the dot product of the two matrices and returns the resulting matrix.
func Dot(m1, m2 [][]float64) [][]float64 {
	m2 = T(m2) // transpose to make it easier
	result := make([][]float64, len(m1))

	for i, row := range m1 {
		result[i] = make([]float64, len(m2))
		for j, column := range m2 {
			result[i][j] = 0
			for _, s := range ArrayProduct(row, column) {
				result[i][j] += s
			}
		}
	}

	return result
}

// Add returns the sum between the two matrices.
func Add(m1, m2 [][]float64) [][]float64 {
	result := make([][]float64, len(m1))

	for i, row := range m1 {
		result[i] = make([]float64, len(row))
		for j, x := range row {
			result[i][j] = x + m2[i][j]
		}
	}

	return result
}

// Sub returns the difference between the two matrices.
func Sub(m1, m2 [][]float64) [][]float64 {
	result := make([][]float64, len(m1))

	for i, row := range m1 {
		result[i] = make([]float64, len(row))
		for j, x := range row {
			result[i][j] = x - m2[i][j]
		}
	}

	return result
}

// Mul does element-wise multiplication and returns a new matrix.
func Mul(m1, m2 [][]float64) [][]float64 {
	result := make([][]float64, len(m1))

	for i, row := range m1 {
		result[i] = make([]float64, len(row))
		for j, x := range row {
			result[i][j] = x * m2[i][j]
		}
	}

	return result
}

// Scale multiplies every element of the matrix with a scalar and returns a new matrix.
func Scale(m [][]float64, scalar float64) [][]float64 {
	result := make([][]float64, len(m))

	for i, row := range m {
		result[i] = make([]float64, len(row))
		for j, x := range row {
			result[i][j] = x * scalar
		}
	}

	return result
}
