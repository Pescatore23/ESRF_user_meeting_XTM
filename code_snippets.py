# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 17:21:26 2026

@author: ROFISCHE

MIT License

Copyright (c) 2026 Robert Fischer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""


import os
import skimage
import numpy as np
import h5py
import dask.array
import scipy as sp

'''
Chapter 1: load image data
'''

# load a single tiff 2D or 3D
im = skimage.io.imread('path/to/tiff')

# load a 3D stack of many 2D tiffs
folder = 'path/to/folder'
files = os.listdir(folder) #folder containing only many 2D tiffs
files.sort()
testim = skimage.io.imread(os.path.join(folder, files[0]))
shp = testim.shape
Stack = np.zeros(shp+(len(files),), dtype=testim.dtype)
for i in range(len(files)):
    Stack[:,:,i] = skimage.io.imread(os.path.join(folder, files[i]))
    
# load data within a hdf5 file with dask
# assuming you know the hdf5 structure
# advanced, allows lazy loading for larger-than-RAM
file = h5py.File('path/to/hdf5')
da = dask.array.from_array(file['hdf5_entry_name']) 
da = da.compute() # optional, loads data into RAM

'''
Chapter 2: registration
'''

'''
2.1 StackReg
'''
from pystackreg import StackReg

def register_2D_time_series(Tstack):
    '''
    Tstack - 16bit uint 3D stack [time,x,y]
    outStack - registered 16bit uint 3D stack [time,x,y]
    trans_matrix - 3D numpy array containing the transformation matrix
    of each time step
    
    '''
    sr = StackReg(StackReg.RIGID_BODY)
    trans_matrix = sr.register_stack(Tstack, reference='first')
    outStack = sr.transform_stack(Tstack)
    outStack = np.uint16(outStack)
    return outStack, trans_matrix


from skimage import transform as tf

def apply_transformation(Tstack, trans_matrix):
    outStack=np.zeros(Tstack.shape)
    for t in range(Tstack.shape[0]):
        tform = tf.AffineTransform(matrix=trans_matrix[t,:,:])
        outStack[t,:,:]=tf.warp(Tstack[t,:,:],tform)
    outStack=np.uint16(outStack*2**16)
    return outStack

'''
2.2 simpleElastix
'''

import SimpleITK as sitk   # assuming simpleITK has been built with Elastix

##### Parameters for the registration   ################
parameterMap = sitk.GetDefaultParameterMap("rigid") #get default settins for rigid registration

# modify some parameters
parameterMap["FixedImagePyramid"] = ["FixedShrinkingImagePyramid"]
parameterMap["MovingImagePyramid"] = ["MovingShrinkingImagePyramid"]
parameterMap["NumberOfResolutions"] = ["2"]
parameterMap["MaximumNumberOfIterations"] = ["1500"]
parameterMap["AutomaticTransformInitialization"] = ["true"]
parameterMap["AutomaticScalesEstimation"] = ["true"]
parameterMap["NewSamplesEveryIteration"] = ["true"]
parameterMap["NewSamplesEveryIteration"] = ["true"]
parameterMap["Interpolator"] = ["BSplineInterpolator"]
parameterMap["NumberOfSamplesForExactGradient"] = ["10000"]
parameterMap["NumberOfSpatialSamples"] = ["10000"]

def register_images_general(image_name, im_fixed, im_tomove, parameter_map, transpath, verbose=False, im_mask=None):

    r"""
    General registration for 2D or 3D images
    This function allows control over the parameter map
    
         
    Parameters
    ----------     
    im_fixed : numpy.ndarray
        Fixed image
    im_tomove: numpy.ndarray
        Image to register (to im_fixed)
    parameter_map: sitk.ParameterMap
        Parameters map object as obtained from sitk.GetParameterMap
    verbose : bool, optional (default=False)
        Enable/Disable log to console 
    im_mask : numpy.ndarray (default=None)
        Mask for sampling points (required). 
        im_mask has same dimensions as im_fixed 
        The mask ndarray has 1 and 0. 1 values denote the area where
        points for the registration are sampled.    
    Returns
    -------
    im_aligned : numpy.ndarray
        Aligned image having same dimensions of im_tomove
                 
    Notes
    -----
    To have further info on parameter maps, go to:
    https://simpleelastix.readthedocs.io/ParameterMaps.html
    For more details on the different parameters, download:
    https://elastix.lumc.nl/download/elastix_manual_v4.8.pdf
    """
    
    itk_image_fixed = sitk.GetImageFromArray(im_fixed)
    itk_image_tomove = sitk.GetImageFromArray(im_tomove)
           
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(itk_image_fixed)
    elastixImageFilter.SetMovingImage(itk_image_tomove)
    elastixImageFilter.SetParameterMap(parameter_map)
    
    if im_mask is not None:
        itk_image_mask = sitk.GetImageFromArray(im_mask)
        # Casting type just in case
        itk_image_mask = sitk.Cast(itk_image_mask, sitk.sitkUInt8)
        elastixImageFilter.SetFixedMask(itk_image_mask)
    
    # Disable/enable logging 
    elastixImageFilter.SetLogToConsole(verbose)
    elastixImageFilter.Execute()
    
    # save transformation vector
    vectorpath = os.path.join(transpath, image_name+'.txt')
    transformParameterMapVector = elastixImageFilter.GetTransformParameterMap()
    sitk.WriteParameterFile(transformParameterMapVector[0], vectorpath)

    im_aligned = elastixImageFilter.GetResultImage()
    # Transform again to numpy
    im_aligned = sitk.GetArrayFromImage(im_aligned)
    return np.uint16(im_aligned) #careful with float images


''' 
Chapter 3: Segmentation
'''

'''
3.1 Denoising
https://docs.scipy.org/doc/scipy/reference/ndimage.html
https://docs.rapids.ai/api/cucim/stable/
https://github.com/rapidsai/cucim
'''
import scipy.ndimage

im = scipy.ndimage.gaussian_filter(im, sigma=1)
im = scipy.ndimage.median_filter(im)

# GPU accelerated image processing
import cupy as cp
import cupyx.scipy.ndimage
import cucim #scikit-image on GPU, not all functions, but many; maybe only Linux

gpu_id = 0 # 1,2,3 if you have more
with cp.cuda.Device(gpu_id):
     imgpu = cp.array(im)
     imgpu = cupyx.scipy.ndimage.median_filter(imgpu)
     imgpu = cucim.skimage.filters.butterworth(imgpu, high_pass=False, cutoff_frequency_ratio=0.4) #lowpass filter needs much GPU RAM
     im = cp.asnumpy(imgpu)
     del imgpu #clear the GPU RAM
     mempool = cp.get_default_memory_pool()
     mempool.free_all_blocks()



'''
3.2 Thresholding
'''

im = im>12000


'''
3.3 Differential segmentation
'''

'''
3.3.1 Difference to reference
'''

Stack_4D = np.zeros((100,200,300,50)) # x,y,z,t mock up

refim = Stack_4D[...,0]
diff_Stack = Stack_4D - refim[...,None]
diff_Stack = diff_Stack>12000

'''
3.3.1 Maximum gradient in filtered per-pixel time series
'''

def low_pass_rule(x,freq,band):
    if abs(freq) > band:
        return 0
    else:
        return x
def high_pass_rule(x,freq,band):
    if abs(freq) < band:
        return 0
    else:
        return x
    
def band_pass_rule(x,freq,band1,band2):
    band=[band1,band2]
    np.sort(band)
    if abs(freq) > band[0] or abs(freq) < band[1]:
        return 0
    else:
        return x

def fourier_filter(signal,band=0.1,dt=1,band2=0.2, filtertype='low'):
    F = np.fft.fft(signal)
    f = np.fft.fftfreq(len(F),dt)
    if filtertype=='low':
        F_filtered = np.array([low_pass_rule(x,freq,band) for x,freq in zip(F,f)])
    if filtertype=='high':
        F_filtered = np.array([high_pass_rule(x,freq,band) for x,freq in zip(F,f)])
    if filtertype=='band':
        F_filtered = np.array([band_pass_rule(x,freq,band,band2) for x,freq in zip(F,f)])
    s_filtered = np.fft.ifft(F_filtered)
    return s_filtered

def get_jump_height(currpx,pos,pos2=0,receding=False):
    if not receding:
        low = np.median(currpx[max(0,pos-20):pos-2])              #"pos-20" changed to 10 because my time resolution is much coarser than Marcelo's
        hig = np.median(currpx[pos+2:min(len(currpx),pos+20)])
        jump = hig-low
    if receding:
        hig = np.median(currpx[max(pos,pos2-20):pos-2])
        low = np.median(currpx[pos2+2:min(len(currpx),pos2+20)])
        jump = low-hig
    return jump

def fft_grad_segmentation(imgs, poremask,z, waterpos=1200):
    '''
    inspired by Marcelo Parada
    Parameters
    ----------
    imgs : numpy
        3D image x,y,t
    poremask : numpy
        2D image x,y
    z : int
        slice position
    waterpos : TYPE, optional
        DESCRIPTION. The default is 1200.

    Returns
    -------
    transitions, transitions2 : numpy (int)
    2D image where pixel value is the (dis)appearance time step of water

    '''
    check = 15000 #T5
    if z<waterpos: check=25000
    transitions = np.zeros([np.shape(imgs)[0],np.shape(imgs)[1]], dtype=np.uint16)
    transitions2 = transitions.copy()

    jumpmin = 1500.0 # minimum jumpheight
    X = imgs.shape[0]
    Y = imgs.shape[1]
    for pX in range(X):
        for pY in range(Y):
#            check if pixel is actually in the relevant pore space
            if poremask[pX,pY] >0:
                # current pixel time series
                currpx = imgs[pX,pY,:].astype(dtype='float')                
                s_filtered = fourier_filter(currpx,band=0.1,dt=1.2)
                
#                # find where maximum gradient occurs
                g_filtered = np.gradient(s_filtered)

                pos = np.argmax(g_filtered)-1                
                jump = get_jump_height(currpx,pos)
    #            
                pos2 = np.argmin(g_filtered)-1
                jump2 = get_jump_height(currpx,pos,pos2=pos2,receding=True)
                
                if jump > jumpmin: # there is a transition; careful, water can also receed!! (can clearly be seen in rare cases)
#                    double check for noise
                    if np.median(currpx[min(pos+10,len(currpx)):min(pos+25,len(currpx))])>check:#7500:#9500:
                        transitions[pX,pY] = pos           
                if jump2 < -2000 and pos2>pos and z<waterpos:
                    transitions2[pX,pY] = pos2
    return transitions, transitions2 

 
'''
Chapter 4: Surface area measurement
'''
import trimesh
# smoothing parameters
k = 0.1
lamb  = 0.6037
iterations = 10

def surface_smoothing(verts, faces, k=k, lamb=lamb, iterations = iterations):
    mesh = trimesh.Trimesh(vertices = verts, faces = faces)
    mesh = trimesh.smoothing.filter_taubin(mesh, lamb=lamb, nu=k, iterations=iterations)
    verts = mesh.vertices
    faces = mesh.faces
    return verts, faces

def interface_extraction(im1, im2):
    '''
    im1, im2 binary volumes for which you want to extract an interface volume
    returns interface as volume
    '''
    im2 = sp.ndimage.binary_dilation(input = im2, structure = cube(3).astype(np.bool))
    interface = np.bitwise_and(im1, im2)
    return interface

def get_surface(im):
    '''
    Parameters
    ----------
    im : numpy array
        binary volume to get surface mesh from
        do not forget to divide surface area by 2 when running on interface
        (front + backside)
    Returns
    -------
    verts, faces of surface meshmesh.

    '''            
    verts, faces, _, _ = skimage.measure.marching_cubes(im)
    verts, faces = surface_smoothing(verts, faces)
    return verts, faces

def calculate_surface_area(verts, faces):
    A = skimage.measure.mesh_surface_area(verts, faces)
    return A



'''
Chapter 5: Pore Network
'''

'''
5.1 restricted watershed
'''
# run within ImageJ
from ij import IJ, ImagePlus, ImageStack

def openSilentStack(folder, show=False, name="stack"):
    # inspired by Marcelo Parada
	imlist=os.listdir(folder)
	imlist.sort()
	isFirst = True
	names = []
	for im in imlist:
		if im == "Thumbs.db": continue
		currImp = IJ.openImage(folder+"/"+im)
		names.append(im)
		if isFirst:
			stack= ImageStack(currImp.getWidth(), currImp.getHeight())
			isFirst = False
		stack.addSlice(currImp.getProcessor())
	imp = ImagePlus("stack",stack)
	if show == True:
		imp.show()
	imp.setTitle(name)
	return imp, names

def labelImage(procfolder,targetFolder,outname,settings):
    if not os.path.exists(targetFolder):
        os.makedirs(targetFolder)  
    imp, names = openSilentStack(procfolder)
    label = IJ.run(imp, "Disconnect Particles", settings)
    savecmd = ''.join(["format=TIFF name=",outname," save=[",targetFolder,
                       '\\',outname,"0000.tif]"])
    IJ.run(label, "Image Sequence... ", savecmd)
    IJ.run("Close All")


'''
5.2 pore neighbors and network extraction
''' 

import deque
from skimage.morphology import cube
import networkx as nx
from joblib import Parallel, delayed


def label_function(struct, pore_object, label, labels):
    mask = pore_object == label
    connections = deque()

    mask = sp.ndimage.binary_dilation(input = mask, structure = struct(3))
    neighbors = np.unique(pore_object[mask])[1:]

    for nb in neighbors:
        if nb != label:
            if nb in labels:
                conn = (label, nb)   
                connections.append(conn)

    return connections

def extract_throat_list(label_matrix, labels, struct = cube): 
    """
    inspired by Jeff Gostick's GETNET

    extracts a list of directed throats connecting pores i->j including a few throat parameters
    undirected network i-j needs to be calculated in a second step
    
    struct ball does not work as you would think (anisotropic expansion)
    """

    def extend_bounding_box(s, shape, pad=3):
        a = deque()
        for i, dim in zip(s, shape):
            start = 0
            stop = dim

            if i.start - pad >= 0:
                start = i.start - pad
            if i.stop + pad < dim:
                stop = i.stop + pad

            a.append(slice(start, stop, None))

        return tuple(a)

    im = label_matrix
    
    crude_pores = sp.ndimage.find_objects(im)

    # throw out None-entries (counterintuitive behavior of find_objects)
    pores = deque()
    bounding_boxes = deque()
    for pore in crude_pores:
        if pore is not None: bb = extend_bounding_box(pore, im.shape)
        if pore is not None and len(np.unique(im[bb])) > 2:
            pores.append(pore)
            bounding_boxes.append(bb)

    connections_raw = Parallel(n_jobs = 32)(
        delayed(label_function)\
            (struct, im[bounding_box], label, labels) \
            for (bounding_box, label) in zip(bounding_boxes, labels)
    )
    # clear out empty objects
    connections = deque()
    for connection in connections_raw:
        if len(connection) > 0:
            connections.append(connection)
    return np.concatenate(connections, axis = 0)


def generate_graph(label_matrix, labels):
    '''
    Parameters
    ----------
    label_matrix : 3D numpy int
        segmented and labeled pore space
    labels : 1D array of labels consider, .e.g. filtered for small spurious labels

    Returns
    -------
    graph :
        networkX graph object.
    '''
    throats = extract_throat_list(label_matrix, labels)
    graph = nx.Graph()
    graph.add_edges_from(np.uint16(throats[:,:2]))
    Gcc = sorted(nx.connected_components(graph), key=len, reverse=True)
    graph = graph.subgraph(Gcc[0])
    return graph


'''
Chapter 6: render in blender
https://github.com/Pescatore23/tomcat-blender
'''

def save_trimesh_as_stl(out_path, mesh):
    stl = trimesh.exchange.stl.export_stl_ascii(mesh)
    stlpath = os.path.join(out_path, ''.join(['mesh.stl']))
    with open(stlpath, 'w+') as file: 
        file.write(stl) 

#run from python env shipped with blender unless you manage somehow to install pyopenvdb
import openvdb 

def numpy_to_vdb(vdbpath, im):
    '''
    Parameters
    ----------
    vdbpath : str
        path where to store vdb.
    im : numpy array
        volume data, can be grayscale or labeled.

    Returns
    -------
    saves volume data as openvdb.

    '''
    
    imc = im*1.0 # dirty conversion to float
    
    # basic and default vdb, maybe there is room for performance and efficiency
    grid = openvdb.FloatGrid()
    grid.copyFromArray(imc.astype(float))
    grid.gridClass = openvdb.GridClass.FOG_VOLUME
    grid.name = 'density'
    openvdb.write(os.path.join(vdbpath, 'volume.vdb'), grid)
