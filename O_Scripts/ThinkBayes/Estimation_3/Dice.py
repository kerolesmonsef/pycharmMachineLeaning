"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

# from O_Scripts.ThinkBayes.thinkbayes import Suite
import thinkbayes
from thinkbayes2 import Suite
import thinkplot


class Dice(Suite):
    """Represents hypotheses about which die was rolled."""

    def Likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: integer number of sides on the die
        data: integer die roll
        """
        if hypo < data:
            return 0
        else:
            return 1.0 / hypo


# in this stupid example we choose a die at random and then after choosing this die ( without replacement )
# and then we keep rolling this die

if __name__ == '__main__':
    suite = Dice([4, 6, 8, 12, 20])
    suite.Print()
    suite.Update(6)
    print('After one 6')
    suite.Print()

    for roll in [6, 8, 7, 7, 5, 4]:
        suite.Update(roll)

    print('After more rolls')
    suite.Print()
    thinkplot.Pdf(suite)
