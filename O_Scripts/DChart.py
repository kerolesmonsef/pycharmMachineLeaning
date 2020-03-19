import matplotlib.pyplot as plt


class DChart:
    # typo = '--' | 'go--' | '.' | ''
    def __init__(self, xs, ys, xLabel: str = "xLabel", yLabel: str = "yLabel", title: str = "title", typo: str = ''):
        self.xs = xs
        self.ys = ys
        self.xLabel = xLabel
        self.yLabel = yLabel
        self.title = title
        self.typo = typo

    def Draw(self) -> None:
        fig, ax = plt.subplots()
        ax.plot(self.xs, self.ys, self.typo)
        ax.plot()
        ax.set(xlabel=self.xLabel, ylabel=self.yLabel, title=self.title)
        ax.grid()
        # fig.savefig("test.png")
        plt.show()
