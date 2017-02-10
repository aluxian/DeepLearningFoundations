package ml

import (
	"strconv"
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

// MatrixParseFloat takes a matrix of strings and returns a matrix of floats.
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

// FilledArray returns a new array where every element is set to the given value.
func FilledArray(length int, value float64) []float64 {
	array := make([]float64, length)
	for i := range array {
		array[i] = value
	}
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

// T returns the transpose of the given matrix (cols <-> rows).
func T(matrix [][]float64) [][]float64 {
	if len(matrix) == 0 {
		return [][]float64{}
	}

	numCols := len(matrix[0])
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

// SplitMatrix returns the first `percent`% and the remaining rows as separate matrices.
func SplitMatrix(matrix [][]float64, percent float32) ([][]float64, [][]float64) {
	itemsNum := int(float32(len(matrix)) * percent) // the number of rows in the first part
	part1 := matrix[:itemsNum]
	part2 := matrix[itemsNum:]
	return part1, part2
}

// ArrayProduct calculates the element-wise product of the two arrays and returns the result.
func ArrayProduct(a1, a2 []float64) []float64 {
	result := make([]float64, len(a1))
	for i, a := range a1 {
		result[i] = a * a2[i]
	}
	return result
}

// BinarySquash maps every element greater than the midpoint as 1, and the rest as 0.
func BinarySquash(m [][]float64, midpoint float64) [][]float64 {
	result := make([][]float64, len(m))
	for i, row := range m {
		result[i] = make([]float64, len(row))
		for j, x := range row {
			if x > midpoint {
				result[i][j] = float64(1)
			} else {
				result[i][j] = float64(0)
			}
		}
	}
	return result
}

// BinaryMatch compares every 2 elements from the matrices and returns 1 if they match, or 0 otherwise.
func BinaryMatch(m1, m2 [][]float64) [][]float64 {
	result := make([][]float64, len(m1))
	for i, row := range m1 {
		result[i] = make([]float64, len(row))
		for j, x := range row {
			if x == m2[i][j] {
				result[i][j] = float64(1)
			} else {
				result[i][j] = float64(0)
			}
		}
	}
	return result
}
