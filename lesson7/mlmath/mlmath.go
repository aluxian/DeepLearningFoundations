package mlmath

import "math"

// Sigmoid calculates 1/(1 + e^x).
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// SigmoidArray returns a new array with sigmoid applied to every element of the given array.
func SigmoidArray(xs []float64) []float64 {
	result := make([]float64, len(xs))
	for i, x := range xs {
		result[i] = Sigmoid(x)
	}
	return result
}

// SigmoidMatrix returns a new matrix with sigmoid applied to every element of the given matrix.
func SigmoidMatrix(xss [][]float64) [][]float64 {
	result := make([][]float64, len(xss))
	for i, xs := range xss {
		result[i] = SigmoidArray(xs)
	}
	return result
}

// SigmoidGradient calculates f​`(h)=f(h)(1−f(h)).
func SigmoidGradient(x float64) float64 {
	sigx := Sigmoid(x)
	return sigx * (1.0 - sigx)
}

// Sum adds up all the elements in the array.
func Sum(xs []float64) float64 {
	var sum float64

	// add them up
	for _, e := range xs {
		sum += e
	}

	return sum
}

// Mean computes the arithmetic mean of the array.
func Mean(xs []float64) float64 {
	return Sum(xs) / float64(len(xs))
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

// Standardize uses mean() and std() to standardize the array.
// It returns a new array without modifying the existing one.
func Standardize(xs []float64) []float64 {
	mean := Mean(xs)
	std := Std(xs)
	ys := make([]float64, len(xs))
	for i, x := range xs {
		ys[i] = (x - mean) / std
	}
	return ys
}
