#copy the following to the top of my jupyter notebooks for this repository
#12.5.2020
#Tim Tyree

# %matplotlib inline
# from lib.utils.my_initialization import *
# %autocall 1
# %load_ext autoreload
# %autoreload 2

darkmode = False

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

from PIL import Image
import trimesh, tetgen, pyvista as pv

import heapq

#not needed
from queue import PriorityQueue

#automate the boring stuff
from IPython import utils
import time, os, sys, re, shutil
beep = lambda x: os.system("echo -n '\\a';sleep 0.2;" * x)
if not 'nb_dir' in globals():
    nb_dir = os.getcwd()

# #load the libraries
from . import *
# from lib import *



if darkmode is True:
	#enter darkmode for jupyter notebooks
	# !jt -t monokai -f fira -fs 13 -nf ptsans -nfs 11 -N -kl -cursw 5 -cursc r -cellw 95% -T

	#Hack for images to obey darkmode
	import seaborn as sns

	from jupyterthemes import jtplot
	jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)