package ml_test

import (
	"testing"

	"."
)

func TestSigmoid(t *testing.T) {
	var tests = []struct {
		input    float64 // input
		expected float64 // expected result
	}{
		{0, 0.5},
		{1, 0.7310585786300049},
		{2, 0.8807970779778823},
		{-2, 0.11920292202211755},
	}

	for _, test := range tests {
		actual := ml.Sigmoid(test.input)
		if actual != test.expected {
			t.Errorf("Sigmoid(%v): expected %v, actual %v", test.input, test.expected, actual)
		}
	}
}

func TestSigmoidPrime(t *testing.T) {
	var tests = []struct {
		input    float64 // input
		expected float64 // expected result
	}{
		{0, 0.25},
		{1, 0.19661193324148185},
		{2, 0.10499358540350662},
		{-2, 0.10499358540350651},
	}

	for _, test := range tests {
		actual := ml.SigmoidPrime(test.input)
		if actual != test.expected {
			t.Errorf("SigmoidPrime(%v): expected %v, actual %v", test.input, test.expected, actual)
		}
	}
}

func TestSigmoidM(t *testing.T) {
	input := [][]float64{{0, 1}, {2, -2}}
	expected := [][]float64{
		{0.5, 0.7310585786300049},
		{0.8807970779778823, 0.11920292202211755},
	}
	actual := ml.SigmoidM(input)
	if !ml.MatrixEquals(expected, actual) {
		t.Errorf("SigmoidM(%v): expected %v, actual %v", input, expected, actual)
	}
}

func TestSigmoidPrimeM(t *testing.T) {
	input := [][]float64{{0, 1}, {2, -2}}
	expected := [][]float64{
		{0.25, 0.19661193324148185},
		{0.10499358540350662, 0.10499358540350651},
	}
	actual := ml.SigmoidPrimeM(input)
	if !ml.MatrixEquals(expected, actual) {
		t.Errorf("SigmoidPrimeM(%v): expected %v, actual %v", input, expected, actual)
	}
}

func TestSum(t *testing.T) {
	input := []float64{1, 2, 3}
	expected := float64(6)
	actual := ml.Sum(input)
	if actual != expected {
		t.Errorf("Sum(%v): expected %v, actual %v", input, expected, actual)
	}
}

func TestMean(t *testing.T) {
	input := []float64{1, 6}
	expected := float64(3.5)
	actual := ml.Mean(input)
	if actual != expected {
		t.Errorf("Mean(%v): expected %v, actual %v", input, expected, actual)
	}
}

func TestMeanM(t *testing.T) {
	input := [][]float64{{1, 7}, {1}}
	expected := float64(3)
	actual := ml.MeanM(input)
	if actual != expected {
		t.Errorf("MeanM(%v): expected %v, actual %v", input, expected, actual)
	}
}

func TestStd(t *testing.T) {
	input := []float64{2, 6, 9, 10}
	expected := float64(3.1124748994971831)
	actual := ml.Std(input)
	if actual != expected {
		t.Errorf("Std(%v): expected %v, actual %v", input, expected, actual)
	}
}

func TestStandardize(t *testing.T) {
	input := []float64{1, 2, 3, 7, 15, -5}
	expected := []float64{
		-0.46285285459973063,
		-0.2994930235645316,
		-0.13613319252933256,
		+0.5173061316114635,
		+1.8241847798930557,
		-1.443011840810925,
	}
	actual := ml.Standardize(input)
	if !ml.ArrayEquals(expected, actual) {
		t.Errorf("Standardize(%v): expected %v, actual %v", input, expected, actual)
	}
}

func TestDot(t *testing.T) {
	input1 := [][]float64{{1, 1, -1}, {4, 0, 2}, {1, 0, 0}}
	input2 := [][]float64{{2, -1}, {3, -2}, {0, 1}}
	expected := [][]float64{{5, -4}, {8, -2}, {2, -1}}
	actual := ml.Dot(input1, input2)
	if !ml.MatrixEquals(expected, actual) {
		t.Errorf("Dot(%v, %v): expected %v, actual %v", input1, input2, expected, actual)
	}
}

func TestAdd(t *testing.T) {
	input1 := [][]float64{{5}, {10, 12}}
	input2 := [][]float64{{1}, {2, 4}}
	expected := [][]float64{{6}, {12, 16}}
	actual := ml.Add(input1, input2)
	if !ml.MatrixEquals(expected, actual) {
		t.Errorf("Add(%v, %v): expected %v, actual %v", input1, input2, expected, actual)
	}
}

func TestSub(t *testing.T) {
	input1 := [][]float64{{5}, {10, 12}}
	input2 := [][]float64{{1}, {2, 4}}
	expected := [][]float64{{4}, {8, 8}}
	actual := ml.Sub(input1, input2)
	if !ml.MatrixEquals(expected, actual) {
		t.Errorf("Sub(%v, %v): expected %v, actual %v", input1, input2, expected, actual)
	}
}

func TestMul(t *testing.T) {
	input1 := [][]float64{{5}, {10, 12}}
	input2 := [][]float64{{1}, {2, 4}}
	expected := [][]float64{{5}, {20, 48}}
	actual := ml.Mul(input1, input2)
	if !ml.MatrixEquals(expected, actual) {
		t.Errorf("Mul(%v, %v): expected %v, actual %v", input1, input2, expected, actual)
	}
}

func TestScale(t *testing.T) {
	input1 := [][]float64{{5}, {10, 12}}
	input2 := float64(2)
	expected := [][]float64{{10}, {20, 24}}
	actual := ml.Scale(input1, input2)
	if !ml.MatrixEquals(expected, actual) {
		t.Errorf("Scale(%v, %v): expected %v, actual %v", input1, input2, expected, actual)
	}
}
