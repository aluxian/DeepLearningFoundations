package ml_test

import (
	"testing"

	"."
)

func TestArrayEquals(t *testing.T) {
	input := []float64{3, 4, 5}

	// same array
	input2 := input
	expected := true
	actual := ml.ArrayEquals(input, input2)
	if expected != actual {
		t.Errorf("ArrayEquals(%v, %v): expected %v, actual %v", input, input2, expected, actual)
	}

	// different array, same length
	input2 = []float64{3, 4, 7}
	expected = false
	actual = ml.ArrayEquals(input, input2)
	if expected != actual {
		t.Errorf("ArrayEquals(%v, %v): expected %v, actual %v", input, input2, expected, actual)
	}

	// different array, different length
	input2 = []float64{3, 4}
	expected = false
	actual = ml.ArrayEquals(input, input2)
	if expected != actual {
		t.Errorf("ArrayEquals(%v, %v): expected %v, actual %v", input, input2, expected, actual)
	}
}

func TestMatrixEquals(t *testing.T) {
	input := [][]float64{{1, 2}, {3, 4}}

	// same matrix
	input2 := input
	expected := true
	actual := ml.MatrixEquals(input, input2)
	if expected != actual {
		t.Errorf("MatrixEquals(%v, %v): expected %v, actual %v", input, input2, expected, actual)
	}

	// different array, same length
	input2 = [][]float64{{1, 2}, {3, 5}}
	expected = false
	actual = ml.MatrixEquals(input, input2)
	if expected != actual {
		t.Errorf("MatrixEquals(%v, %v): expected %v, actual %v", input, input2, expected, actual)
	}

	// different array, different length
	input2 = [][]float64{{1, 2}}
	expected = false
	actual = ml.MatrixEquals(input, input2)
	if expected != actual {
		t.Errorf("MatrixEquals(%v, %v): expected %v, actual %v", input, input2, expected, actual)
	}
}

func TestMatrixParseFloat(t *testing.T) {
	input := [][]string{{"3", "4"}, {"6", "7"}}
	expected := [][]float64{{3, 4}, {6, 7}}
	actual, err := ml.MatrixParseFloat(input)
	if err != nil {
		t.Error("MatrixParseFloat unexpected error:", err)
	}
	if !ml.MatrixEquals(expected, actual) {
		t.Errorf("MatrixParseFloat(%v): expected %v, actual %v", input, expected, actual)
	}

	// expect error
	input = [][]string{{"3", "4"}, {"6", "z"}}
	_, err = ml.MatrixParseFloat(input)
	if err == nil {
		t.Errorf("MatrixParseFloat(%v): expected err != nil", input)
	}
}

func TestFilledArray(t *testing.T) {
	length := 3
	value := float64(5)
	expected := []float64{value, value, value}
	actual := ml.FilledArray(length, value)
	if !ml.ArrayEquals(expected, actual) {
		t.Errorf("FilledArray(%v, %v): expected %v, actual %v", length, value, expected, actual)
	}
}

func TestFilledMatrix(t *testing.T) {
	numRows := 2
	numCols := 3
	value := float64(5)
	expected := [][]float64{{value, value, value}, {value, value, value}}
	actual := ml.FilledMatrix(numRows, numCols, value)
	if !ml.MatrixEquals(expected, actual) {
		t.Errorf("FilledMatrix(%v, %v, %v): expected %v, actual %v", numRows, numCols, value, expected, actual)
	}
}

func TestT(t *testing.T) {
	input := [][]float64{{3, 4, 5}, {6, 7, 8}, {90, 100, 110}}
	expected := [][]float64{{3, 6, 90}, {4, 7, 100}, {5, 8, 110}}
	actual := ml.T(input)
	if !ml.MatrixEquals(expected, actual) {
		t.Errorf("T(%v): expected %v, actual %v", input, expected, actual)
	}

	// test case with zero rows
	input = [][]float64{}
	expected = [][]float64{}
	actual = ml.T(input)
	if !ml.MatrixEquals(expected, actual) {
		t.Errorf("T(%v): expected %v, actual %v", input, expected, actual)
	}
}

func TestSplitByValues(t *testing.T) {
	input := []float64{1, 2, 1}
	expected := [][]float64{
		{1, 0, 1},
		{0, 1, 0},
	}
	actual := ml.SplitByValues(input)
	if !ml.MatrixEquals(expected, actual) {
		t.Errorf("SplitByValues(%v): expected %v, actual %v", input, expected, actual)
	}
}

func TestSplitMatrix(t *testing.T) {
	input := [][]float64{{1, 2}, {3, 4}, {5, 6}, {7, 8}}
	expected1 := [][]float64{{1, 2}, {3, 4}}
	expected2 := [][]float64{{5, 6}, {7, 8}}
	actual1, actual2 := ml.SplitMatrix(input, 0.50) // 50%
	if !ml.MatrixEquals(expected1, actual1) {
		t.Errorf("SplitMatrix(%v, 0.50): expected %v, actual %v", input, expected1, actual1)
	}
	if !ml.MatrixEquals(expected2, actual2) {
		t.Errorf("SplitMatrix(%v, 0.50): expected %v, actual %v", input, expected2, actual2)
	}
}

func TestArrayProduct(t *testing.T) {
	input1 := []float64{1, 2, 3}
	input2 := []float64{4, 5, 6}
	expected := []float64{4, 10, 18}
	actual := ml.ArrayProduct(input1, input2)
	if !ml.ArrayEquals(expected, actual) {
		t.Errorf("ArrayProduct(%v, %v): expected %v, actual %v", input1, input2, expected, actual)
	}
}

func TestBinarySquash(t *testing.T) {
	input1 := [][]float64{{0.2, 0.6}, {-5, 5}}
	input2 := float64(0.5)
	expected := [][]float64{{0, 1}, {0, 1}}
	actual := ml.BinarySquash(input1, input2)
	if !ml.MatrixEquals(expected, actual) {
		t.Errorf("BinarySquash(%v, %v): expected %v, actual %v", input1, input2, expected, actual)
	}
}

func TestBinaryMatch(t *testing.T) {
	input1 := [][]float64{{0.2, 0.6}, {-5, 5}}
	input2 := [][]float64{{0.2, 0.5}, {-5, 0}}
	expected := [][]float64{{1, 0}, {1, 0}}
	actual := ml.BinaryMatch(input1, input2)
	if !ml.MatrixEquals(expected, actual) {
		t.Errorf("BinaryMatch(%v, %v): expected %v, actual %v", input1, input2, expected, actual)
	}
}
