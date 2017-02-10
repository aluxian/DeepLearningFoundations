package mlmath_test

import (
	"testing"

	"."
	"../mlutils"
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
		actual := mlmath.Sigmoid(test.input)
		if actual != test.expected {
			t.Errorf("Sigmoid(%v): expected %v, actual %v", test.input, test.expected, actual)
		}
	}
}

func TestSigmoidArray(t *testing.T) {
	input := []float64{0, 1, 2, -2}
	expected := []float64{
		0.5,
		0.7310585786300049,
		0.8807970779778823,
		0.11920292202211755,
	}
	actual := mlmath.SigmoidArray(input)
	if !mlutils.ArrayEquals(expected, actual) {
		t.Errorf("SigmoidArray(%v): expected %v, actual %v", input, expected, actual)
	}
}

func TestSigmoidMatrix(t *testing.T) {
	input := [][]float64{{0, 1}, {2, -2}}
	expected := [][]float64{
		{0.5, 0.7310585786300049},
		{0.8807970779778823, 0.11920292202211755},
	}
	actual := mlmath.SigmoidMatrix(input)
	if !mlutils.MatrixEquals(expected, actual) {
		t.Errorf("SigmoidMatrix(%v): expected %v, actual %v", input, expected, actual)
	}
}

func TestSigmoidGradient(t *testing.T) {
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
		actual := mlmath.SigmoidGradient(test.input)
		if actual != test.expected {
			t.Errorf("SigmoidGradient(%v): expected %v, actual %v", test.input, test.expected, actual)
		}
	}
}

func TestSum(t *testing.T) {
	input := []float64{1, 2, 3}
	expected := float64(6)
	actual := mlmath.Sum(input)
	if actual != expected {
		t.Errorf("Sum(%v): expected %v, actual %v", input, expected, actual)
	}
}

func TestMean(t *testing.T) {
	input := []float64{1, 6}
	expected := float64(3.5)
	actual := mlmath.Mean(input)
	if actual != expected {
		t.Errorf("Mean(%v): expected %v, actual %v", input, expected, actual)
	}
}

func TestStd(t *testing.T) {
	input := []float64{2, 6, 9, 10}
	expected := float64(3.1124748994971831)
	actual := mlmath.Std(input)
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
	actual := mlmath.Standardize(input)
	if !mlutils.ArrayEquals(expected, actual) {
		t.Errorf("Standardize(%v): expected %v, actual %v", input, expected, actual)
	}
}
