import numpy as np, pandas as pd, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

from PIL import Image
import trimesh, tetgen, pyvista as pv

from IPython.utils import io



def plot_mesh(vertices, faces=None, input_file_name=None, darkmode = True, background_color = 'k', text=None,
	window_size = [1280,720], font_color = 'w',
	cpos = [(3.77, 3.77, 3.77),(0.0069, -0.0045, 0.0),(0.0, 0.0, 1.0)]
	):
	'''visualize the mesh surface using vtk and return an img.  
	vertices is a numpy array of vertices with faces.
	faces is a numpy array of indices specifying the faces of the surface of the mesh using ^those vertices.
	input_file_name is a string giving the path to a file containing ^those faces (e.g. *.ply or *.stl).
	an example text=f'time={tme:.1f}'
	window_size = [1280,720] is the standard aspect ratio for youtube,
	'''
	#visualize the mesh surface
	pv.set_plot_theme('document')

	#get the vtk object (wrapped by pyvista from withing tetgen.  Faces recorded by trimesh))
	if faces is None:
		if input_file_name is None:
			Exception('either faces or input_file_name must be specified')
		mesh_trimesh = trimesh.load(input_file_name)
		faces = mesh_trimesh.faces

	tet = tetgen.TetGen(vertices, faces)
	# #fault tolerant tetrahedralization
	vertices_tet, elements_tet = tet.tetrahedralize(order=1, mindihedral=0., minratio=10., nobisect=False, steinerleft=100000)#
	tet.make_manifold()
	grid = tet.grid

	# advanced plotting
	plotter = pv.Plotter()
	if darkmode:
		plotter.set_background(background_color)
		plotter.add_mesh(grid, 'lightgrey', lighting=True)
		#looks like tron plotter.add_mesh(grid, 'r', 'wireframe')
	else:
		plotter.add_mesh(grid, 'lightgrey', lighting=True)
		font_color = 'k'
	if text is not None:	
		plotter.add_text(
		    text,
		    position='upper_left',
		    font_size=24,
		    color=font_color,
		    font='times')
	#font options
	# FONT_KEYS = {'arial': vtk.VTK_ARIAL,
	#              'courier': vtk.VTK_COURIER,
	#              'times': vtk.VTK_TIMES}

	#cpos is (camera position, focal point, and view up)
	#for movies, just set the camera position to some constant value.

	_cpos, img = plotter.show(title=None, return_img=True, cpos=cpos, window_size=window_size, use_ipyvtk = False, interactive=False, auto_close=True);
	plotter.deep_clean()
	del plotter
	return img

def get_img_of_system(vertices, input_file_name, **kwargs):
    with io.capture_output() as captured:
        img = plot_mesh(vertices, input_file_name=input_file_name, **kwargs);
    return img