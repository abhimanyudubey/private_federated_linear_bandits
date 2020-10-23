import numpy as np
import matplotlib

def normalize(v, p=2):
    ''' project vector on to unit L-p ball. '''
    norm=np.linalg.norm(v, ord=p)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm


def matplotlib_init():
    ''' Initialize matplotlib parameters for pretty figures. '''
    matplotlib.rcParams['lines.linewidth'] = 2
    matplotlib.rcParams['axes.grid'] = True
    matplotlib.rcParams['grid.linestyle'] = '--'
    matplotlib.rcParams['grid.color'] = '#aaaaaa'
    matplotlib.rcParams['xtick.major.size'] = 0
    matplotlib.rcParams['ytick.major.size'] = 0
    matplotlib.rcParams['xtick.labelsize'] = 12
    matplotlib.rcParams['ytick.labelsize'] = 12
    matplotlib.rcParams['axes.labelsize'] = 12
    matplotlib.rcParams['axes.titlesize'] = 12
    matplotlib.rcParams['legend.fontsize'] = 14
    # matplotlib.rcParams['legend.frameon'] = False
    matplotlib.rcParams['figure.subplot.top'] = 0.85
    matplotlib.rcParams['axes.facecolor'] = 'white'
    matplotlib.rcParams['axes.linewidth'] = 0.8
    matplotlib.rcParams['figure.dpi'] = 600