from thinkbayes2 import Hist, Pmf, Suite, Beta
import thinkplot

beta = Beta(140, 110)
redditor = beta.MakePmf(110)
thinkplot.Pdf(redditor)
# thinkplot.decorate(xlabel='Reliability (R)',
#                    ylabel='PMF')
mean_r = redditor.Mean()


beta = Beta(1, 1)
item = beta.MakePmf(11)
# thinkplot.Pdf(item)
# thinkplot.decorate(xlabel='Quality (Q))',
#                    ylabel='PMF')

mean_q = item.Mean()

d = {}
for r, p1 in redditor.Items():
    for q, p2 in item.Items():
        d[q, r] = p1 * p2

suite = Pmf(d);

