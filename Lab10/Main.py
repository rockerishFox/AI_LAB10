import random
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import neural_network

from Load import *


def main():
    np.random.seed(5)
    inputs = []
    outputs = []

    inputs, outputs = loadSepia(outputs, inputs)
    inputs, outputs = loadNormal(outputs, inputs)

    c = list(zip(inputs, outputs))
    random.shuffle(c)
    inputs, outputs = zip(*c)

    indexes = [i for i in range(len(inputs))]
    train_sample_indexes = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    test_sample_indexes = [i for i in indexes if i not in train_sample_indexes]

    train_inputs = [(inputs[i]) for i in train_sample_indexes]
    train_outputs = [outputs[i] for i in train_sample_indexes]

    test_inputs = [inputs[i] for i in test_sample_indexes]
    test_outputs = [outputs[i] for i in test_sample_indexes]

    train_input_flattened = [flatten(image) for image in train_inputs]
    test_input_flattened = [flatten(image) for image in test_inputs]


    # normalizare folosind z = (x - u) / s unde
    # x = input
    # u = mean of training samples
    # s = standard deviation
    scaler = StandardScaler()
    scaler.fit(train_input_flattened)
    normalised_training = scaler.transform(train_input_flattened)
    normalised_test = scaler.transform(test_input_flattened)

    # functia de activare f(x) = max(0, x)
    classifier = neural_network.MLPClassifier(hidden_layer_sizes=(5, 5,), max_iter=100, solver='sgd', verbose=10,
                                              random_state=1, learning_rate_init=.05)
    classifier.fit(normalised_training, train_outputs)

    predicted_outputs = classifier.predict(normalised_test)

    labels = ["Sepia", "Normal"]
    labelNames = labels
    matrix = confusion_matrix(test_outputs, predicted_outputs)
    acc = sum([matrix[i][i] for i in range(len(labelNames))]) / len(test_outputs)
    print()
    print("Accuracy using confusion matrix: ", acc)

    error = accuracy_score(test_outputs, predicted_outputs)
    print("Accuracy tool: ", acc)


main()
