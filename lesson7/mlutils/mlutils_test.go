package mlutils_test

import (
	"testing"

	"."
)

func TestArrayEquals(t *testing.T) {
	input := []float64{3, 4, 5}

	// same array
	input2 := input
	expected := true
	actual := mlutils.ArrayEquals(input, input2)
	if expected != actual {
		t.Errorf("ArrayEquals(%v, %v): expected %v, actual %v", input, input2, expected, actual)
	}

	// different array, same length
	input2 = []float64{3, 4, 7}
	expected = false
	actual = mlutils.ArrayEquals(input, input2)
	if expected != actual {
		t.Errorf("ArrayEquals(%v, %v): expected %v, actual %v", input, input2, expected, actual)
	}

	// different array, different length
	input2 = []float64{3, 4}
	expected = false
	actual = mlutils.ArrayEquals(input, input2)
	if expected != actual {
		t.Errorf("ArrayEquals(%v, %v): expected %v, actual %v", input, input2, expected, actual)
	}
}

func TestMatrixEquals(t *testing.T) {
	input := [][]float64{{1, 2}, {3, 4}}

	// same matrix
	input2 := input
	expected := true
	actual := mlutils.MatrixEquals(input, input2)
	if expected != actual {
		t.Errorf("MatrixEquals(%v, %v): expected %v, actual %v", input, input2, expected, actual)
	}

	// different array, same length
	input2 = [][]float64{{1, 2}, {3, 5}}
	expected = false
	actual = mlutils.MatrixEquals(input, input2)
	if expected != actual {
		t.Errorf("MatrixEquals(%v, %v): expected %v, actual %v", input, input2, expected, actual)
	}

	// different array, different length
	input2 = [][]float64{{1, 2}}
	expected = false
	actual = mlutils.MatrixEquals(input, input2)
	if expected != actual {
		t.Errorf("MatrixEquals(%v, %v): expected %v, actual %v", input, input2, expected, actual)
	}
}

func TestMatrixParseFloat(t *testing.T) {
	input := [][]string{{"3", "4"}, {"6", "7"}}
	expected := [][]float64{{3, 4}, {6, 7}}
	actual, err := mlutils.MatrixParseFloat(input)
	if err != nil {
		t.Error("MatrixParseFloat unexpected error:", err)
	}
	if !mlutils.MatrixEquals(expected, actual) {
		t.Errorf("MatrixParseFloat(%v): expected %v, actual %v", input, expected, actual)
	}

	// expect error
	input = [][]string{{"3", "4"}, {"6", "z"}}
	_, err = mlutils.MatrixParseFloat(input)
	if err == nil {
		t.Errorf("MatrixParseFloat(%v): expected err != nil", input)
	}
}

func TestFillArray(t *testing.T) {
	input := make([]float64, 5)
	expected := []float64{1, 1, 1, 1, 1}
	mlutils.FillArray(input, 1.0)
	actual := input
	if !mlutils.ArrayEquals(expected, actual) {
		t.Errorf("FillArray(%v, 1): expected %v, actual %v", input, expected, actual)
	}
}

func TestFilledArray(t *testing.T) {
	length := 3
	value := float64(5)
	expected := []float64{value, value, value}
	actual := mlutils.FilledArray(length, value)
	if !mlutils.ArrayEquals(expected, actual) {
		t.Errorf("FilledArray(%v, %v): expected %v, actual %v", length, value, expected, actual)
	}
}

func TestFilledMatrix(t *testing.T) {
	numRows := 2
	numCols := 3
	value := float64(5)
	expected := [][]float64{{value, value, value}, {value, value, value}}
	actual := mlutils.FilledMatrix(numRows, numCols, value)
	if !mlutils.MatrixEquals(expected, actual) {
		t.Errorf("FilledMatrix(%v, %v, %v): expected %v, actual %v", numRows, numCols, value, expected, actual)
	}
}

func TestExtractColumnFrom2D(t *testing.T) {
	input := [][]float64{{3, 4, 5}, {6, 7, 8}, {90, 100, 110}}
	expected := []float64{4, 7, 100}
	actual := mlutils.ExtractColumnFrom2D(input, 1)
	if !mlutils.ArrayEquals(expected, actual) {
		t.Errorf("ExtractColumnFrom2D(%v, 1): expected %v, actual %v", input, expected, actual)
	}
}

func TestT(t *testing.T) {
	input := [][]float64{{3, 4, 5}, {6, 7, 8}, {90, 100, 110}}
	expected := [][]float64{{3, 6, 90}, {4, 7, 100}, {5, 8, 110}}
	actual := mlutils.T(input)
	if !mlutils.MatrixEquals(expected, actual) {
		t.Errorf("T(%v): expected %v, actual %v", input, expected, actual)
	}
}

func TestSplitByValues(t *testing.T) {
	input := []float64{1, 2, 1}
	expected := [][]float64{
		{1, 0, 1},
		{0, 1, 0},
	}
	actual := mlutils.SplitByValues(input)
	if !mlutils.MatrixEquals(expected, actual) {
		t.Errorf("SplitByValues(%v): expected %v, actual %v", input, expected, actual)
	}
}

func TestSplitArray(t *testing.T) {
	input := []float64{1, 2, 3, 4}
	expected1 := []float64{1, 2}
	expected2 := []float64{3, 4}
	actual1, actual2 := mlutils.SplitArray(input, 0.50) // 50%
	if !mlutils.ArrayEquals(expected1, actual1) {
		t.Errorf("SplitArray(%v, 0.50): expected %v, actual %v", input, expected1, actual1)
	}
	if !mlutils.ArrayEquals(expected2, actual2) {
		t.Errorf("SplitArray(%v, 0.50): expected %v, actual %v", input, expected2, actual2)
	}
}

func TestSplitMatrix(t *testing.T) {
	input := [][]float64{{1, 2}, {3, 4}, {5, 6}, {7, 8}}
	expected1 := [][]float64{{1, 2}, {3, 4}}
	expected2 := [][]float64{{5, 6}, {7, 8}}
	actual1, actual2 := mlutils.SplitMatrix(input, 0.50) // 50%
	if !mlutils.MatrixEquals(expected1, actual1) {
		t.Errorf("SplitMatrix(%v, 0.50): expected %v, actual %v", input, expected1, actual1)
	}
	if !mlutils.MatrixEquals(expected2, actual2) {
		t.Errorf("SplitMatrix(%v, 0.50): expected %v, actual %v", input, expected2, actual2)
	}
}

func TestSplitMatrixHoriz(t *testing.T) {
	input := [][]float64{{1, 2, 3, 4}, {5, 6, 7, 8}}
	expected1 := [][]float64{{1, 2}, {5, 6}}
	expected2 := [][]float64{{3, 4}, {7, 8}}
	actual1, actual2 := mlutils.SplitMatrixHoriz(input, 0.50) // 50%
	if !mlutils.MatrixEquals(expected1, actual1) {
		t.Errorf("SplitMatrixHoriz(%v, 0.50): expected %v, actual %v", input, expected1, actual1)
	}
	if !mlutils.MatrixEquals(expected2, actual2) {
		t.Errorf("SplitMatrixHoriz(%v, 0.50): expected %v, actual %v", input, expected2, actual2)
	}
}

func TestArrayProduct(t *testing.T) {
	input1 := []float64{1, 2, 3}
	input2 := []float64{4, 5, 6}
	expected := []float64{4, 10, 18}
	actual := mlutils.ArrayProduct(input1, input2)
	if !mlutils.ArrayEquals(expected, actual) {
		t.Errorf("ArrayProduct(%v, %v): expected %v, actual %v", input1, input2, expected, actual)
	}
}

func TestMatrixProduct(t *testing.T) {
	input1 := [][]float64{{1, 1, -1}, {4, 0, 2}, {1, 0, 0}}
	input2 := [][]float64{{2, -1}, {3, -2}, {0, 1}}
	expected := [][]float64{{5, -4}, {8, -2}, {2, -1}}
	actual := mlutils.MatrixProduct(input1, input2)
	if !mlutils.MatrixEquals(expected, actual) {
		t.Errorf("MatrixProduct(%v, %v): expected %v, actual %v", input1, input2, expected, actual)
	}
}

func TestAMatrixProduct(t *testing.T) {
	// row matrix
	input1 := []float64{2, 4}
	input2 := [][]float64{{2, 1, 3}, {3, 2, 1}}
	expected := [][]float64{{16, 10, 10}}
	actual := mlutils.AMatrixProduct(input1, input2)
	if !mlutils.MatrixEquals(expected, actual) {
		t.Errorf("AMatrixProduct(%v, %v): expected %v, actual %v", input1, input2, expected, actual)
	}

	// column matrix
	input1 = []float64{2, 4}
	input2 = [][]float64{{2, 1, 3}}
	expected = [][]float64{{4, 2, 6}, {8, 4, 12}}
	actual = mlutils.AMatrixProduct(input1, input2)
	if !mlutils.MatrixEquals(expected, actual) {
		t.Errorf("AMatrixProduct(%v, %v): expected %v, actual %v", input1, input2, expected, actual)
	}
}

func TestArrayToRowMatrix(t *testing.T) {
	input := []float64{1, 2, 3}
	expected := [][]float64{{1, 2, 3}}
	actual := mlutils.ArrayToRowMatrix(input)
	if !mlutils.MatrixEquals(expected, actual) {
		t.Errorf("ArrayToRowMatrix(%v): expected %v, actual %v", input, expected, actual)
	}
}

func TestArrayToColumnMatrix(t *testing.T) {
	input := []float64{1, 2, 3}
	expected := [][]float64{{1}, {2}, {3}}
	actual := mlutils.ArrayToColumnMatrix(input)
	if !mlutils.MatrixEquals(expected, actual) {
		t.Errorf("ArrayToColumnMatrix(%v): expected %v, actual %v", input, expected, actual)
	}
}
