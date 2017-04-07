import matplotlib.pyplot as plt
from PyQt4 import QtCore

class RoiPoint(object):
    ''' Class for getting a mouse drawn rectangle
    Based on the example from:
    http://matplotlib.org/users/event_handling.html#draggable-rectangle-exercise
    Note that:

    * It takes several input mouse clicks and creates two lists of x and y
    coordinates for these points.

    '''
    def __init__(self):
        self.ax = plt.gca()
        self.x0 = []
        self.y0 = []
        self.visible = False
        self.set = False
        self.end = False
        self.plt_style = 'r+'
        self.ax.figure.canvas.setFocusPolicy( QtCore.Qt.ClickFocus )
        self.ax.figure.canvas.setFocus()
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.ax.figure.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.ax.figure.canvas.mpl_connect('key_release_event', self.on_key_release)

    def on_press(self, event):
        if not self.set and not self.end:
            self.x0.append(event.xdata)
            self.y0.append(event.ydata)

            self.draw()

    def on_release(self, event):
        if not self.set and not self.end:
            self.ax.figure.canvas.draw()
            self.ax.figure.canvas.mpl_disconnect('button_press_event')
            self.ax.figure.canvas.mpl_disconnect('button_release_event')

    def on_key_release(self, event):
        if not self.set:
            print('end')
            self.set = True
            self.ax.figure.canvas.mpl_disconnect('key_press_event')
            self.ax.figure.canvas.mpl_disconnect('key_release_event')
            self.end = True
            return self.x0, self.y0

    def on_key_press(self):
        if not self.set:
            self.set = True

    def draw(self):
        if not self.visible:
            return self.ax.plot(self.x0, self.y0, self.plt_style)
