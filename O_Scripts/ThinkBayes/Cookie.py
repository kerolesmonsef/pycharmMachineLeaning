from O_Scripts.ThinkBayes.thinkbayes import Pmf


class Cookie(Pmf):
    mixes = {
        'Bowl 1': dict(vanilla=0.75, chocolate=0.25),
        'Bowl 2': dict(vanilla=0.5, chocolate=0.5),
    }

    def Likelihood(self, data, hypo):
        mix = self.mixes[hypo]
        like = mix[data]
        return like

    def __init__(self, hypos):
        Pmf.__init__(self)
        for hypo in hypos:
            self.Set(hypo, 1)
        self.Normalize()

    def Update(self, data):
        for hypo in self.Values():
            like = self.Likelihood(data, hypo)
            self.Mult(hypo, like)
            self.Mult(hypo, (5/8))
        self.Normalize()


if __name__ == "__main__":
    hypos = ['Bowl 1', 'Bowl 2']
    pmf = Cookie(hypos)
    pmf.Update('vanilla')
