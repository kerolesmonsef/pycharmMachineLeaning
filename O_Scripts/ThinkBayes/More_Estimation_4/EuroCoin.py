from O_Scripts.ThinkBayes.thinkbayes import Suite
from O_Scripts.DChart import DChart
from O_Scripts.ThinkBayes import thinkbayes


class Euro(Suite):
    # def Likelihood(self, data, hypo):
    #     x = hypo
    #     if data == 'H':
    #         return x / 100.0
    #     else:
    #         return 1 - (x / 100.0)

    def Likelihood(self, data, hypo):
        prob_head = hypo / 100.0
        prob_tail = 1 - prob_head
        heads, tails = data
        like = prob_head ** heads * prob_tail ** tails
        return like


def TrianglePrior() -> Suite:
    suite = Euro()
    for x in range(0, 51):
        suite.Set(x, x)
    for x in range(51, 101):
        suite.Set(x, 100 - x)
    suite.Normalize()
    return suite


if __name__ == "__main__":
    # suite = Euro(range(0, 101))  # list of all heads probabilities from 0 to 100
    # suite = TrianglePrior()
    # dataset = 'H' * 140 + 'T' * 110
    # for data in dataset:
    #     # in this loop if it is head then increase the probability of high head probability ( 95 , 96 , ... , 100)
    #     # if tail then increase the probability of of high tail like ( 0 , 3 , .... 5 )
    #     # if 50 tail and 50 head
    #     # then
    #     # first will increase the first 50 and decrease the second 50
    #     # second will decrease the first 50 and increase the second 50
    #     # but this will increase the middle of 100
    #     suite.Update(data)
    # suite.UpdateSet(dataset)
    beta = thinkbayes.Beta()
    beta.Update((140, 110))
    print(beta.Mean())
    print(beta.EvalPdf(0.1))
    # suite = Euro(range(0, 101))
    # suite.Print()
    # suite.Update((140, 110))
    # print('Mean = ', suite.Mean())
    # print('Median = ', thinkbayes.Percentile(suite, 50))
    # print('CI = ', thinkbayes.CredibleInterval(suite, 90))
    #
    # dchart = DChart([x for x, _ in suite.Items()], [y for _, y in suite.Items()], typo='', xLabel="Heads",
    #                 yLabel='Probability')
    # dchart.Draw()
