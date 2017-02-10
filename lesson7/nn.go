package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"os"

	"./ml"
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

	// parse data
	rawRecords = rawRecords[1:] // exclude the csv header row
	records, err := ml.MatrixParseFloat(rawRecords)
	if err != nil {
		panic(err)
	}

	// transpose the matrix to make accessing columns easier
	records = ml.T(records)

	// split rank into dummy features
	records = append(records[:3], ml.SplitByValues(records[3])...)

	// standardize
	records[1] = ml.Standardize(records[1]) // GRE
	records[2] = ml.Standardize(records[2]) // GPA

	// extract x and y
	x := records[1:7] // x has 6 features: gre, gpa, rank1, rank2, rank3, rank4
	y := records[0:1] // y has 1 target: admit

	// transpose x and y so rows are columns again
	x = ml.T(x)
	y = ml.T(y)

	// done
	return x, y
}

func main() {
	// read data from the csv
	features, targets := readData("binary.csv")

	// split dataset
	xTrain, xTest := ml.SplitMatrix(features, 0.90) // 90%
	yTrain, yTest := ml.SplitMatrix(targets, 0.90)  // 90%

	// hyperparameters
	numHidden := 4
	numEpochs := 10000
	learnRate := 0.005

	// counts
	numFeatures := len(xTrain[0])
	numOutputs := len(yTrain[0])

	// initialize weights
	W1 := ml.FilledMatrix(numFeatures, numHidden, 0.0)
	W2 := ml.FilledMatrix(numHidden, numOutputs, 0.0)

	// track the loss
	lastLoss := float64(0)

	// start training
	for epoch := 0; epoch < numEpochs; epoch++ {
		// forward pass
		z2 := ml.Dot(xTrain, W1)
		a2 := ml.SigmoidM(z2)

		z3 := ml.Dot(a2, W2)
		yHat := ml.SigmoidM(z3)

		// backward pass
		yError := ml.Sub(yTrain, yHat)

		delta3 := ml.Mul(yError, ml.SigmoidPrimeM(z3))
		dJdW2 := ml.Dot(ml.T(a2), delta3)

		delta2 := ml.Mul(ml.Dot(delta3, ml.T(W2)), ml.SigmoidPrimeM(z2))
		dJdW1 := ml.Dot(ml.T(xTrain), delta2)

		// update weights
		W1 = ml.Add(W1, ml.Scale(dJdW1, learnRate))
		W2 = ml.Add(W2, ml.Scale(dJdW2, learnRate))

		// print out the mean squared error on the training set
		if epoch%(numEpochs/10) == 0 {
			a2 := ml.SigmoidM(ml.Dot(xTrain, W1))
			yHat := ml.SigmoidM(ml.Dot(a2, W2))
			yError := ml.Sub(yTrain, yHat)  // y - yHat
			yError = ml.Mul(yError, yError) // square each element
			loss := ml.MeanM(yError)

			if lastLoss != 0 && lastLoss < loss {
				fmt.Printf("Train loss: %v WARNING - Loss Increasing\n", loss)
			} else {
				fmt.Printf("Train loss: %v\n", loss)
			}

			lastLoss = loss
		}
	}

	// calculate accuracy on test data
	a2 := ml.SigmoidM(ml.Dot(xTest, W1))
	yHat := ml.SigmoidM(ml.Dot(a2, W2))
	predictions := ml.BinarySquash(yHat, 0.5)
	matches := ml.BinaryMatch(predictions, yTest)
	accuracy := ml.MeanM(matches)
	fmt.Printf("Prediction accuracy: %v\n", accuracy)
}
