import matplotlib.pyplot as plt


class Plot():
    def __init__(self):
        pass

    def drawLine(self, line, color=None):
        plt.plot(line, color=color)

    def drawPoints(self, points, color=None):
        for point in points:
            if len(point) > 0:
                plt.scatter(point[0], point[1], color=color)

    def show(self):
        plt.show()
