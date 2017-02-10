package mlutils

import (
	"strconv"

	"../mlmath"
)

// ArrayEquals checks if two arrays are equal.
func ArrayEquals(array1, array2 []float64) bool {
	if len(array1) != len(array2) {
		return false
	}

	// check every element
	for i := 0; i < len(array1); i++ {
		if array1[i] != array2[i] {
			return false
		}
	}

	return true
}

// MatrixEquals checks if two matrices are equal.
func MatrixEquals(matrix1, matrix2 [][]float64) bool {
	if len(matrix1) != len(matrix2) {
		return false
	}

	// check every row
	for i := 0; i < len(matrix1); i++ {
		if !ArrayEquals(matrix1[i], matrix2[i]) {
			return false
		}
	}

	return true
}

// MatrixParseFloat takes a matrix of strings and returns a new matrix of floats.
func MatrixParseFloat(matrix [][]string) ([][]float64, error) {
	parsed := make([][]float64, len(matrix))

	for i, row := range matrix {
		parsed[i] = make([]float64, len(row))
		for j, s := range row {
			fl, err := strconv.ParseFloat(s, 64)
			if err != nil {
				return nil, err
			}
			parsed[i][j] = fl
		}
	}

	return parsed, nil
}

// FillArray sets every element of the array to the given value.
func FillArray(array []float64, value float64) {
	for i := range array {
		array[i] = value
	}
}

// FilledArray returns a new array where every element is set to the given value.
func FilledArray(length int, value float64) []float64 {
	array := make([]float64, length)
	FillArray(array, value)
	return array
}

// FilledMatrix returns a new matrix where every element is set to the given value.
func FilledMatrix(numRows, numCols int, value float64) [][]float64 {
	matrix := make([][]float64, numRows)

	for i := 0; i < numRows; i++ {
		matrix[i] = FilledArray(numCols, value)
	}

	return matrix
}

// ExtractColumnFrom2D extracts a column from a matrix to a vector.
func ExtractColumnFrom2D(matrix [][]float64, icol int) []float64 {
	column := make([]float64, len(matrix))

	for i, row := range matrix {
		column[i] = row[icol]
	}

	return column
}

// T returns the transpose of the given matrix (cols <-> rows).
func T(matrix [][]float64) [][]float64 {
	numCols := len(matrix[0]) // hopefully it's not empty
	transpose := make([][]float64, numCols)

	for icol := 0; icol < numCols; icol++ {
		transpose[icol] = make([]float64, len(matrix))
		for i, row := range matrix {
			transpose[icol][i] = row[icol]
		}
	}

	return transpose
}

// SplitByValues takes all the discrete values in the given array and splits them in a matrix
// where for each value `a` of `array`, `row_i` of `matrix` = `1` if `i` == `a`, or `0` otherwise.
func SplitByValues(array []float64) [][]float64 {
	maxv := 0 // it is assumed that the discrete values are from 1 through maxv

	// calculate maxv
	for _, elem := range array {
		intElem := int(elem)
		if intElem > maxv {
			maxv = intElem
		}
	}

	// populate final matrix
	matrix := make([][]float64, maxv)
	for i := 0; i < maxv; i++ {
		matrix[i] = make([]float64, len(array))
	}
	for i, dv := range array {
		// assign all to 0
		for j := 0; j < maxv; j++ {
			matrix[j][i] = 0.0
		}

		// assign the correct one to 1
		index := int(dv) - 1 // subtract 1 because the values start from 1
		matrix[index][i] = 1.0
	}

	return matrix
}

// SplitArray returns the first `percent`% and the remaining elements as separate arrays.
func SplitArray(array []float64, percent float32) ([]float64, []float64) {
	itemsNum := int(float32(len(array)) * percent) // the number of elements in the first part

	part1 := array[:itemsNum]
	part2 := array[itemsNum:]

	return part1, part2
}

// SplitMatrix returns the first `percent`% and the remaining columns as separate matrices.
func SplitMatrix(matrix [][]float64, percent float32) ([][]float64, [][]float64) {
	itemsNum := int(float32(len(matrix)) * percent) // the number of rows in the first part

	part1 := matrix[:itemsNum]
	part2 := matrix[itemsNum:]

	return part1, part2
}

// SplitMatrixHoriz returns the first `percent`% and the remaining columns as separate matrices.
func SplitMatrixHoriz(matrix [][]float64, percent float32) ([][]float64, [][]float64) {
	totalNumItems := len(matrix[0])                   // hopefully it's not empty
	itemsNum := int(float32(totalNumItems) * percent) // the number of columns in the first part

	part1 := make([][]float64, len(matrix))
	part2 := make([][]float64, len(matrix))

	for i, row := range matrix {
		part1[i] = row[:itemsNum]
		part2[i] = row[itemsNum:]
	}

	return part1, part2
}

// ArrayProduct calculates the element-wise product of the two arrays and returns the resulting array.
func ArrayProduct(a1, a2 []float64) []float64 {
	result := make([]float64, len(a1))
	for i, a := range a1 {
		result[i] = a * a2[i]
	}
	return result
}

// MatrixProduct calculates the dot product of the two matrices and returns the resulting matrix.
func MatrixProduct(m1, m2 [][]float64) [][]float64 {
	m2 = T(m2) // transpose to make it easier
	result := make([][]float64, len(m1))

	for i, r1 := range m1 {
		result[i] = make([]float64, len(m2))
		for j, r2 := range m2 {
			result[i][j] = mlmath.Sum(ArrayProduct(r1, r2))
		}
	}

	return result
}

// AMatrixProduct calculates the dot product of the two matrices and returns the resulting matrix.
func AMatrixProduct(a1 []float64, m2 [][]float64) [][]float64 {
	if len(m2) == len(a1) {
		return MatrixProduct(ArrayToRowMatrix(a1), m2)
	}
	return MatrixProduct(ArrayToColumnMatrix(a1), m2)
}

// ArrayToRowMatrix takes an array and returns a matrix with 1 row.
func ArrayToRowMatrix(array []float64) [][]float64 {
	result := make([][]float64, 1)
	result[0] = make([]float64, len(array))
	copy(result[0], array)
	return result
}

// ArrayToColumnMatrix takes an array and returns a matrix with 1 column.
func ArrayToColumnMatrix(array []float64) [][]float64 {
	result := make([][]float64, len(array))
	for i, a := range array {
		result[i] = []float64{a}
	}
	return result
}
