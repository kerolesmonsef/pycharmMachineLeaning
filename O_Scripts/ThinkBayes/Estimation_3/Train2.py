from O_Scripts.ThinkBayes.Estimation_3.Dice import Dice
from O_Scripts.DChart import DChart
from thinkbayes import Pmf, Percentile


class Train2(Dice):
    """Represents hypotheses about how many trains the company has.

    The likelihood function for the train problem is the same as
    for the Dice problem.
    """

    def __init__(self, hypos, alpha=1.0):
        Pmf.__init__(self)

        for hypo in hypos:
            self.Set(hypo, hypo ** (-alpha))
        self.Normalize()


if __name__ == '__main__':
    hypos = range(1, 1001)
    suite = Train2(hypos=hypos)

    # suite.Update(60)
    # print(suite.Mean()) # this is the expectation
    # suite.Print()
    for data in [60, 30, 90]:
        suite.Update(data)
    print("expectation = ", suite.Mean())  # this is the expectation
    interval = Percentile(suite, 5), Percentile(suite, 95)
    print("interval = ", interval)  # this is the expectation

    cdf = suite.MakeCdf()
    interval = cdf.Percentile(5), cdf.Percentile(95)
    print("interval = ", interval)  # this is the expectation
    # dchart = DChart([x for x, _ in suite.Items()], [y for _, y in suite.Items()])
    # dchart.Draw()
