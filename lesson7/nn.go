package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"math"
	"os"

	"./mlmath"
	"./mlutils"
)

func readData(fileName string) ([][]float64, [][]float64) {
	f, err := os.Open(fileName)
	if err != nil {
		panic(err)
	}

	// read all the liens
	r := csv.NewReader(bufio.NewReader(f))
	rawRecords, err := r.ReadAll() // records look like [[admit gre gpa rank], ...]
	if err != nil {
		panic(err)
	}

	// parse data then transpose the matrix to make accessing columns easier
	rawRecords = rawRecords[1:] // exclude the csv header row
	records, err := mlutils.MatrixParseFloat(rawRecords)
	if err != nil {
		panic(err)
	}
	records = mlutils.T(records)

	// split rank into different features
	ranks := records[3]                                        // save the ranks row separately
	records = records[:3]                                      // remove the ranks row
	records = append(records, mlutils.SplitByValues(ranks)...) // append the new features

	// standardize
	records[1] = mlmath.Standardize(records[1]) // GRE
	records[2] = mlmath.Standardize(records[2]) // GPA

	// extract x and y
	x := records[1:7] // x has 6 features: gre, gpa, rank1, rank2, rank3, rank4
	y := records[0:1] // y has 1 target: admit

	// transpose x and y so rows are columns again
	x = mlutils.T(x)
	y = mlutils.T(y)

	// done
	return x, y
}

func main() {
	// read data from the csv
	features, targets := readData("binary.csv")

	// split dataset
	xTrain, xTest := mlutils.SplitMatrix(features, 0.90) // 90%
	yTrain, yTest := mlutils.SplitArray(targets, 0.90)   // 90%

	// hyperparameters
	numHidden := 4
	numEpochs := 10000
	learnRate := 0.005

	// counts
	numRecords := len(targets)
	numFeatures := len(features)

	// initialize weights
	W1 := mlutils.FilledMatrix(numFeatures, numHidden, 0.0)
	W2 := mlutils.FilledMatrix(numHidden, 1, 0.0)

	// track the loss
	lastLoss := float64(0)

	// start training
	for epoch := 0; epoch < numEpochs; epoch++ {
		var x []float64

		// fill delta matrices with zeros
		deltaW1 := mlutils.FilledMatrix(numFeatures, numHidden, 0.0)
		deltaW2 := mlutils.FilledMatrix(numHidden, 1, 0.0)

		// one pass for each record
		for iy, y := range yTrain {
			x = xTrain[iy]

			// forward pass
			hiddenActivations := mlmath.SigmoidMatrix(mlutils.MatrixProduct(x, W1))
			output := mlmath.Sigmoid(mlutils.AMatrixProduct(hiddenActivations, W2)[0][0])

			// backward pass
			yError := y - output
			outputError := yError * output * (1 - output)
			hiddenErrors := make([]float64, len(hiddenActivations))
			for i, ha := range hiddenActivations {
				hiddenErrors[i] = outputError * W2[i][0] * ha * (1 - ha)
			}

			// update the change in weights
			for i, ha := range hiddenActivations {
				deltaW2[i][0] += outputError * ha
			}
			for i, featureValue := range x {
				for j, hiddenError := range hiddenErrors {
					deltaW1[i][j] += hiddenError * featureValue
				}
			}
		}

		// update weights
		for i := 0; i < numFeatures; i++ {
			for j := 0; j < numHidden; j++ {
				W1[i][j] += learnRate * deltaW1[i][j] / float64(numRecords)
			}
		}
		for i := 0; i < numHidden; i++ {
			W2[i][0] += learnRate * deltaW2[i][0] / float64(numRecords)
		}

		// print out the mean square error on the training set
		if epoch%(numEpochs/10) == 0 {
			hiddenActivations := mlmath.SigmoidArray(mlutils.AMatrixProduct(x, W1)[0])
			output := mlmath.Sigmoid(mlutils.AMatrixProduct(hiddenActivations, W2)[0][0])
			yDiffs := make([]float64, len(targets))
			for i := range yDiffs {
				yDiffs[i] = math.Pow(output-targets[i], 2)
			}
			loss := mlmath.Mean(yDiffs)

			if lastLoss != 0 && lastLoss < loss {
				fmt.Printf("Train loss: %v WARNING - Loss Increasing\n", loss)
			} else {
				fmt.Printf("Train loss: %v\n", loss)
			}

			lastLoss = loss
		}
	}

	// calculate accuracy on test data
	hiddenActivations := mlmath.SigmoidMatrix(mlutils.MatrixProduct(xTest, W1))
	outputsMatrix := mlutils.MatrixProduct(hiddenActivations, W2)
	outputs := make([]float64, len(outputsMatrix))
	for i, row := range outputsMatrix {
		outputs[i] = mlmath.Sigmoid(row[0])
	}
	predictions := make([]float64, len(outputs))
	for i, output := range outputs {
		if output > 0.5 {
			predictions[i] = float64(1)
		} else {
			predictions[i] = float64(0)
		}
	}
	matches := make([]float64, len(predictions))
	for i, prediction := range predictions {
		if prediction == yTest[i] {
			matches[i] = float64(1)
		} else {
			matches[i] = float64(0)
		}
	}
	accuracy := mlmath.Mean(matches)
	fmt.Printf("Prediction accuracy: %v\n", accuracy)
}
