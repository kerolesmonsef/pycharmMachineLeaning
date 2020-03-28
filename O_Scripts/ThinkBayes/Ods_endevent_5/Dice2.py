import thinkbayes2 as thinkbayes
from thinkbayes2 import Pmf
import thinkplot


class Dice2(Pmf):
    def __init__(self, sides):
        Pmf.__init__(self)
        for x in range(1, sides + 1):
            self.Set(x, 1)
        self.Normalize()


if __name__ == "__main__":
    d6 = Dice2(6)
    dices = [d6] * 6
    three = thinkbayes.SampleSum(dices, 1000)
    thinkplot.Pmf(three)
