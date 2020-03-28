from typing import List

import O_Scripts.ThinkBayes.thinkbayes as thinkbayes
import thinkplot
import numpy
import csv


def ReadCSV(filename='amr_db_d_win.csv') -> List:
    fp = open(filename)
    reader = csv.reader(fp)
    res = []

    for t in reader:
        try:
            _heading = t[0]
            the_float = float(_heading)
            res.append(int(the_float))
        except ValueError:
            pass
    fp.close()
    return res


def ReadData(filename='showcases.2011.csv'):
    """Reads a CSV file of data.

    Args:
      filename: string filename

    Returns: sequence of (price1 price2 bid1 bid2 diff1 diff2) tuples
    """
    fp = open(filename)
    reader = csv.reader(fp)
    res = []

    for t in reader:
        _heading = t[0]
        data = t[1:]
        try:
            # data = [int(x) for x in data]
            # print heading, data[0], len(data)
            # res.append(data)
            [res.append(int(x)) for x in data]
        except ValueError:
            pass

    fp.close()
    return res


def myfunc():
    prices = ReadCSV()
    pdf = thinkbayes.EstimatedPdf(prices)
    low, high = min(prices), max(prices)
    n = 101
    xs = numpy.linspace(low, high, n)
    pmf = pdf.MakePmf(xs)
    thinkplot.Pmf(pmf)
    pmf.Print()


def bookfuc():
    prices = ReadData()
    pdf = thinkbayes.EstimatedPdf(prices)
    low, high = 0, 75000
    n = 101
    xs = numpy.linspace(low, high, n)
    pmf = pdf.MakePmf(xs)
    cdf = pmf.MakeCdf()
    thinkplot.Pmf(pmf)


if __name__ == '__main__':
    # bookfuc()
    myfunc()