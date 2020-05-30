"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from O_Scripts.ThinkBayes.thinkbayes import Suite


class M_and_M(Suite):
    """Map from hypothesis (A or B) to probability."""

    mix94 = dict(brown=30,
                 yellow=20,
                 red=20,
                 green=10,
                 orange=10,
                 tan=10,
                 blue=0)

    mix96 = dict(blue=24,
                 green=20,
                 orange=16,
                 yellow=14,
                 red=13,
                 brown=13,
                 tan=0)

    hypoA = dict(bag1=mix94, bag2=mix96)
    hypoB = dict(bag1=mix96, bag2=mix94)

    hypotheses = dict(A=hypoA, B=hypoB)

    def Likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: string hypothesis (A or B)
        data: tuple of string bag, string color
        """
        bag, color = data
        mix = self.hypotheses[hypo][bag]
        like = mix[color]
        return like


if __name__ == '__main__':
    suite = M_and_M('AB')
    suite.Update(('bag1', 'brown'))
    # suite.Update(('bag1', 'green'))
    # suite.Update(('bag1', 'green'))
    # suite.Update(('bag1', 'green'))
    suite.Print()

    # suite = M_and_M('BA')
    # suite.Update(('bag1', 'green'))
    # suite.Update(('bag2', 'yellow'))
    # suite.Print()