import numpy as np
from data_prep import features, targets, features_test, targets_test

np.random.seed(21)

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


# Hyperparameters
n_hidden = 2  # number of hidden units
epochs = 900
learnrate = 0.005

n_records, n_features = features.shape
last_loss = None
# Initialize weights
W1 = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_features, n_hidden))
W2 = np.random.normal(scale=1 / n_features ** .5,
                                         size=n_hidden)

for e in range(epochs):
    del_w_input_hidden = np.zeros(W1.shape)
    del_w_hidden_output = np.zeros(W2.shape)
    for x, y in zip(features.values, targets):
        ## Forward pass ##
        # TODO: Calculate the output
        z2 = np.dot(x, W1)
        hidden_activations = sigmoid(z2)

        output = np.dot(hidden_activations, W2)

        ## Backward pass ##
        # TODO: Calculate the error
        error = y - sigmoid(output)

        # TODO: Calculate error gradient in output unit
        output_error = error * sigmoidPrime(output) # delta1

        # TODO: propagate errors to hidden layer
        hidden_error = np.dot(output_error, W2) * sigmoidPrime(z2) # delta2

        # TODO: Update the change in weights
        del_w_hidden_output += output_error * hidden_activations
        del_w_input_hidden += hidden_error * x[:, None]

    # TODO: Update weights
    W1 += learnrate * del_w_input_hidden / n_records
    W2 += learnrate * del_w_hidden_output / n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_activations = sigmoid(np.dot(x, W1))
        out = sigmoid(np.dot(hidden_activations,
                             W2))
        loss = np.mean((out - targets) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
hidden = sigmoid(np.dot(features_test, W1))
out = sigmoid(np.dot(hidden, W2))
predictions = out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
