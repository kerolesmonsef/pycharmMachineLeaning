from O_Scripts.ThinkBayes.Estimation_3.Dice import Dice
from O_Scripts.DChart import DChart


class Train(Dice):
    """Represents hypotheses about how many trains the company has.

    The likelihood function for the train problem is the same as
    for the Dice problem.
    """


if __name__ == '__main__':
    hypos = range(1, 1001)
    suite = Train(hypos)

    # suite.Update(60)
    # print(suite.Mean()) # this is the expectation
    # suite.Print()
    for data in [60, 30, 90]:
        suite.Update(data)
    print(suite.Mean())  # this is the expectation

    # dchart = DChart([x for x, _ in suite.Items()], [y for _, y in suite.Items()])
    # dchart.Draw()
