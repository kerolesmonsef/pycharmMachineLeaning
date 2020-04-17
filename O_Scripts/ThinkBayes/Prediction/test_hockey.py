from O_Scripts.ThinkBayes import thinkplot, thinkbayes

USE_SUMMARY_DATA = True


class Hockey(thinkbayes.Suite):
    """Represents hypotheses about the scoring rate for a team."""

    def __init__(self, name: str = None, mu: float = 2.8, sigma: float = 0.3):
        """Initializes the Hockey object.

        label: string
        """

        pmf = thinkbayes.MakeNormalPmf(mu, sigma, 4)
        thinkbayes.Suite.__init__(self, pmf, name=name)

    def Likelihood(self, data: float, hypo: float) -> float:
        """Computes the likelihood of the data under the hypothesis.

        Evaluates the Poisson PMF for lambda and k.

        hypo: goal scoring rate in goals per game
        data: goals scored in one game
        """
        lam = hypo
        k = data
        like = thinkbayes.EvalPoissonPmf(k, lam)
        return like


def MakeGoalPmf(suite: thinkbayes.Suite) -> thinkbayes.Pmf:
    metapmf = thinkbayes.Pmf()
    for lam, prob in suite.Items():  # loop throw the goals after update
        pmf = thinkbayes.MakePoissonPmf(lam, 10)  # give me a distribution of mean = lam ;
        # then what is the probability of 0 -> 10 given the mean is lam_i
        metapmf.Set(pmf, prob)
    mix = thinkbayes.MakeMixture(metapmf)
    return mix


if __name__ == "__main__":
    suite1 = Hockey(name='bruins')
    suite2 = Hockey(name='canucks')

    suite1.UpdateSet([0, 2, 8, 4])
    suite2.UpdateSet([1, 3, 1, 0])
    # this is the probablility after updating the goals
    # here is the question : what the probability of each goal in the list if the current goal is

    goal_dist1 = MakeGoalPmf(suite1)
    goal_dist2 = MakeGoalPmf(suite2)

    diff = goal_dist1 - goal_dist2

    thinkplot.Pmf(diff)
