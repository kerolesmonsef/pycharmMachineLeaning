# Make Predictions with Naive Bayes On The Iris Dataset
from csv import reader
from math import sqrt
from math import exp
from math import pi
import numpy as np


class OwnNaiveBayes:
    DataSet = [];

    def __init__(self, filename: str):
        self.DataSet = self.__load_csv(filename)
        self.__str_column_to_int(len(self.DataSet[0]) - 1)
        for i in range(len(self.DataSet[0]) - 1):
            self.__str_column_to_float(i)

    # Load a CSV file
    def __load_csv(self, filename):
        dataset = list()
        with open(filename, 'r') as file:
            csv_reader = reader(file)
            for ROW in csv_reader:
                if not ROW:
                    continue
                dataset.append(ROW)
        return dataset

    # Convert string column to float
    def __str_column_to_float(self, column):
        for row in self.DataSet:
            row[column] = float(row[column].strip())

    # Convert string column to integer
    def __str_column_to_int(self, column):
        class_values = [row[column] for row in self.DataSet]
        unique = set(class_values)
        unique = sorted(unique)
        lookup = dict()
        for i, value in enumerate(unique):
            lookup[value] = i
            print('[%s] => %d' % (value, i))
        for ROW in self.DataSet:
            ROW[column] = lookup[ROW[column]]
        return lookup

    # Split the dataset by class values, returns a dictionary
    def __separate_by_class(self):
        separated = dict()
        for i in range(len(self.DataSet)):
            vector = self.DataSet[i]
            class_value = vector[-1]
            if class_value not in separated:
                separated[class_value] = list()
            separated[class_value].append(vector)
        return separated

    # Calculate the mean of a list of numbers
    def __mean(self, numbers):
        return sum(numbers) / float(len(numbers))

    # Calculate the standard deviation of a list of numbers
    def __stdev(self, numbers):
        avg = self.__mean(numbers)
        variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
        return sqrt(variance)

    # Calculate the mean, stdev and count for each column in a dataset
    def __summarize_dataset(self, data):
        summaries = [(self.__mean(column), self.__stdev(column), len(column)) for column in zip(*data)]
        del (summaries[-1])
        return summaries

    # Split dataset by class then calculate statistics(mean , std , len) for each row
    def __summarize_by_class(self):
        separated = self.__separate_by_class()
        summaries = dict()
        for class_value, rows in separated.items():
            summaries[class_value] = self.__summarize_dataset(rows)
        return summaries

    # Calculate the Gaussian probability distribution function for x
    def __calculate_probability(self, x, mean, stdev):
        exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return (1 / (sqrt(2 * pi) * stdev)) * exponent

    # Calculate the probabilities of predicting each class for a given row
    def __calculate_class_probabilities(self, summaries, row):
        total_rows = sum([summaries[label][0][2] for label in summaries])
        probabilities = dict()
        for class_value, class_summaries in summaries.items():
            probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
            for i in range(len(class_summaries)):
                mean, stdev, _ = class_summaries[i]
                probabilities[class_value] *= self.__calculate_probability(row[i], mean, stdev)
        return probabilities

    # Predict the class for a given row
    def predict(self, new_row, show_print=False):
        summaries = self.__summarize_by_class()
        probabilities = self.__calculate_class_probabilities(summaries, new_row)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if show_print:
                print("class : ", class_value, " => ", probability)
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label

    def score(self):
        X = np.array(self.DataSet)
        real_y = X[:, -1]
        X = X[:, :-1]
        pred_y = [self.predict(list(xi)) for xi in X]
        score_array = [int(com == real_y[i]) for i, com in enumerate(pred_y)]
        return sum(score_array) / len(self.DataSet)


if __name__ == "__main__":
    filename = '../../Data/Classification/naive_bayes.data'
    NB = OwnNaiveBayes(filename)
    row = [5.7, 2.9, 4.2, 1.3]
    # predict the label
    label = NB.predict(row, show_print=True)
    print('Data=%s, Predicted: %s' % (row, label))
    print("Score : ", NB.score())
