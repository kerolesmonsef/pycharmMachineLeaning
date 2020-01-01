import pandas as pd
from math import sqrt

dataset = pd.read_csv("../../Data/Classification/naive_bayes.data")


def separate_by_class(dataset: pd):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset.iloc[i]
        class_value = vector[-1]
        if class_value not in separated:
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated


def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
    return sqrt(variance)


# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column)) for column in dataset.columns.drop('class')]
    return summaries


summarize_dataset(dataset)
