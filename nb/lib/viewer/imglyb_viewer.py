import imglyb
from imglyb import util
import numpy as numpy

#Example Usage
img = np.random.rand( 300, 200, 100 )*2**16
wrapped = util.to_imglib( img )
util.BdvFunctions.show( wrapped, "wrapped image" )

rgba = np.random.randint( 
	2**32, size = ( 300, 200, 100),
	dtype=np.uint32 )
wrapped_rgba = util.to_imglib_argb( rgba )
util.BdvFunctions.show( wrapped_rgba, "wrapped rgba image" )