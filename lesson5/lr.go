package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

type point struct {
	x, y float64
}

func readCsv(filename string, delimiter string) ([]point, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// parse each line as a data point
	var points []point
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.Split(scanner.Text(), delimiter)

		// parse x as float
		x, err := strconv.ParseFloat(line[0], 64)
		if err != nil {
			return nil, err
		}

		// parse y as float
		y, err := strconv.ParseFloat(line[1], 64)
		if err != nil {
			return nil, err
		}

		// create a new point
		pt := point{x, y}
		points = append(points, pt)
	}

	return points, scanner.Err()
}

func computeError(b, m float64, points []point) float64 {
	var regError float64

	// sum all the errors
	for i := 0; i < len(points); i++ {
		initialX := points[i].x
		initialY := points[i].y
		predictedY := m*initialX + b
		regError += math.Pow(initialY-predictedY, 2)
	}

	// return the average error
	return regError / float64(len(points))
}

func gradientDescent(points []point, b, m float64, learningRate float64, numIterations int) (newB, newM float64) {
	for i := 0; i < numIterations; i++ {
		// update b and m with better values
		b, m = stepGradient(b, m, points, learningRate)
	}
	return b, m
}

func stepGradient(b, m float64, points []point, learningRate float64) (newB, newM float64) {
	var n = float64(len(points))

	var gradientB float64
	var gradientM float64

	// compute direction with respect to b and m
	for i := 0; i < len(points); i++ {
		x := points[i].x
		y := points[i].y
		calculatedY := m*x + b
		// partial derivatives of the error function (whatever that is)
		gradientB += -(2.0 / n) * (y - calculatedY)
		gradientM += -(2.0 / n) * x * (y - calculatedY)
	}

	// new values
	newB = b - learningRate*gradientB
	newM = m - learningRate*gradientM

	return
}

func main() {
	// collect data
	points, err := readCsv("data.csv", ",")
	if err != nil {
		panic(err)
	}

	// hyperparameters
	var learningRate = 0.0001
	var numIterations = 1000
	var initialB float64
	var initialM float64

	// train model
	initialError := computeError(initialB, initialM, points)
	fmt.Printf("starting gradient descent at b=%v m=%v error=%v\n", initialB, initialM, initialError)
	b, m := gradientDescent(points, initialB, initialM, learningRate, numIterations)
	finalError := computeError(b, m, points)
	fmt.Printf("ending point at b=%v m=%v error=%v after %v iterations\n", b, m, finalError, numIterations)

	// evaluate model
	testValues := []float64{2, 10, 20, 30, 50, 60, 80, 100, 200}
	for i := 0; i < len(testValues); i++ {
		x := testValues[i]
		fmt.Printf("%vhours => %v\n", x, m*x+b)
	}
}
