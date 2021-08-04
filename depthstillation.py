# Ctypes package used to call the forward warping C library
from numpy.ctypeslib import ndpointer
from ctypes import *
import ctypes

# Some packages we use 
import matplotlib.pyplot as plt
import collections
import numpy as np
import argparse
import cv2
import os
import torch
import math
import random 
import tqdm

# External scripts
from bilateral_filter import sparse_bilateral_filtering
from flow_colors import *
from geometry import *

# Import warping library
lib = cdll.LoadLibrary("external/forward_warping/libwarping.so")
warp = lib.forward_warping

# Parse input arguments
parser = argparse.ArgumentParser(description="Depthstillation options")
parser.add_argument("--num_motions", dest="num_motions", type=int, help="Number of motions", default=1)
parser.add_argument("--segment", dest="segment", action="store_true", help="Enable segmentation (for moving objects)")
parser.add_argument("--mask_type", dest="mask_type", type=str, default="H'", help="Select mask type", choices=["H", "H'"])
parser.add_argument("--num_objects", dest="num_objects", type=int, help="Number of moving objects", default=1)
parser.add_argument("--no_depth", dest="no_depth", action="store_true", help="Assumes constant depth")
parser.add_argument("--no_sharp", dest="no_sharp", action="store_true", help="Disable depth sharpening")
parser.add_argument("--change_k", dest="change_k", action="store_true", help="Use a different K matrix")
parser.add_argument("--change_motion", dest="change_motion", action="store_true", help="Sample a different random motion")
args = parser.parse_args()

# if num_motions greater than 1, ignore change_motion setting
if args.num_motions > 1:
	args.change_motion = False

# Init progress bar
pbar = tqdm.tqdm(total=args.num_motions)

# Create directories to save outputs
if not os.path.exists(os.path.join("dCOCO", "im0")):
	os.makedirs(os.path.join("dCOCO", "im0"))
if not os.path.exists(os.path.join("dCOCO", "im1_raw")):
	os.makedirs(os.path.join("dCOCO", "im1_raw"))
if not os.path.exists(os.path.join("dCOCO", "im1")):
	os.makedirs(os.path.join("dCOCO", "im1"))
if not os.path.exists(os.path.join("dCOCO", "flow")):
	os.makedirs(os.path.join("dCOCO", "flow"))
if not os.path.exists(os.path.join("dCOCO", "flow_color")):
	os.makedirs(os.path.join("dCOCO", "flow_color"))
if not os.path.exists(os.path.join("dCOCO", "depth_color")):
	os.makedirs(os.path.join("dCOCO", "depth_color"))
if not os.path.exists(os.path.join("dCOCO", "instances_color")):
	os.makedirs(os.path.join("dCOCO", "instances_color"))
if not os.path.exists(os.path.join("dCOCO", "H")):
	os.makedirs(os.path.join("dCOCO", "H"))
if not os.path.exists(os.path.join("dCOCO", "M")):
	os.makedirs(os.path.join("dCOCO", "M"))
if not os.path.exists(os.path.join("dCOCO", "M'")):
	os.makedirs(os.path.join("dCOCO", "M'"))
if not os.path.exists(os.path.join("dCOCO", "P")):
	os.makedirs(os.path.join("dCOCO", "P"))
if not os.path.exists(os.path.join("dCOCO", "H'")):
	os.makedirs(os.path.join("dCOCO", "H'"))
	
# Fix random seeds
random.seed(1024)
np.random.seed(1024)

# Open I0 image
rgb = cv2.imread("samples/im0.jpg", -1)
if len(rgb.shape)<3:
	h, w = rgb.shape
	rgb = np.stack((rgb,rgb,rgb),-1)
else:
	h, w, _ = rgb.shape

# Open D0 (inverse) depth map and resize to I0
depth = cv2.imread("samples/d0.png", -1) / (2**16-1)
if depth.shape[0] != h or depth.shape[1] != w:
	depth = cv2.resize(depth, (w, h))

# Get depth map and normalize
depth = 1.0 / (depth + 0.005)
depth[depth > 100] = 100

# Set depth to constant value in case we do not want to use depth
if args.no_depth:
	depth = depth * 0. + 1.

# Depth sharpening (bilateral filter)
if not args.no_sharp:
	depth = sparse_bilateral_filtering( depth.copy(), rgb.copy(), filter_size=[5, 5], num_iter=2, )

# Load segmentation mask in case we simulate moving objects
if args.segment:
	labels=[]
	instances_mask = cv2.imread("samples/s0.png", -1)

	# Resize instance mask to I0
	if instances_mask.shape[0] != h or instances_mask.shape[1] != w:
		instances_mask = cv2.resize(instances_mask, (w, h))

	# Get total number of objects
	classes = instances_mask.max()
	
	# Get pixels count for each object
	areas = np.array([instances_mask[instances_mask==c].sum() for c in range(classes+1)], np.float32)
	
	# If we have any object
	if areas.shape[0] > 1:

		# Keep args.num_objects labels having the largest amount of pixels
		labels=areas.argsort()[-args.num_objects:][::-1]
		instances = []
		
		# For each object kept
		for l in labels:
		
			# Create a segmentation mask for the single object
			seg_mask = np.zeros_like(instances_mask)
			
			# Set to 1 pixels having label l
			seg_mask[instances_mask==l] = 1
			seg_mask = np.expand_dims(seg_mask, 0)
			
			# Cast to pytorch tensor and append to masks list
			seg_mask = torch.from_numpy(np.stack((seg_mask, seg_mask), -1)).float()
			instances.append(seg_mask)

# Cast I0 and D0 to pytorch tensors
rgb = torch.from_numpy(np.expand_dims(rgb, 0))
depth = torch.from_numpy(np.expand_dims(depth, 0)).float()

# Fix a plausible K matrix
K = np.array([[[0.58, 0, 0.5, 0], [0, 0.58, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]]], dtype=np.float32)

# Fix a different K matrix in case 
if args.change_k:
	K = np.array([[[1.16, 0, 0.5, 0], [0, 1.16, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]]], dtype=np.float32)
K[:, 0, :] *= w
K[:, 1, :] *= h
inv_K = torch.from_numpy(np.linalg.pinv(K))
K = torch.from_numpy(K)

# Create objects in charge of 3D projection
backproject_depth = BackprojectDepth(1, h, w)
project_3d = Project3D(1, h, w)

# Prepare p0 coordinates
meshgrid = np.meshgrid(range(w), range(h), indexing="xy")
p0 = np.stack(meshgrid, axis=-1).astype(np.float32)    

# Loop over the number of motions
for idm in range(args.num_motions):

	# Initiate masks dictionary
	masks = {}

	# Sample random motion (twice, if you want a new one)
	sample_motions = 2 if args.change_motion else 1
	for mot in range(sample_motions):

		# Generate random vector t 
		# Random sign
		scx = ( (-1)**random.randrange(2) )
		scy = ( (-1)**random.randrange(2) ) 
		scz = ( (-1)**random.randrange(2) )
		# Random scalars in -0.2,0.2, excluding -0.1,0.1 to avoid zeros / very small motions
		cx = (random.random()*0.1+0.1)* scx
		cy = (random.random()*0.1+0.1)* scy
		cz = (random.random()*0.1+0.1)* scz
		camera_mot = [cx, cy, cz]

		# generate random triplet of Euler angles
		# Random sign
		sax = ( (-1)**random.randrange(2) )
		say = ( (-1)**random.randrange(2) ) 
		saz = ( (-1)**random.randrange(2) ) 
		# Random angles in -pi/18,pi/18, excluding -pi/36,pi/36 to avoid zeros / very small rotations
		ax = (random.random()*math.pi / 36.0 + math.pi / 36.0) * sax
		ay = (random.random()*math.pi / 36.0 + math.pi / 36.0) * say
		az = (random.random()*math.pi / 36.0 + math.pi / 36.0) * saz
		camera_ang = [ax, ay, az]

	axisangle = torch.from_numpy(np.array([[camera_ang]], dtype=np.float32))
	translation = torch.from_numpy(np.array([[camera_mot]]))

	# Compute (R|t)
	T1 = transformation_from_parameters(axisangle, translation)
	
	# Back-projection  
	cam_points = backproject_depth(depth, inv_K)
	
	# Apply transformation T_{0->1}
	p1, z1 = project_3d(cam_points, K, T1)
	z1 = z1.reshape(1, h, w)

	# Simulate objects moving independently
	if args.segment:

		# Loop over objects
		for l in range(len(labels)):

			sign=1
			# We multiply the sign by -1 to obtain a motion similar to the one shown in the supplementary (not exactly the same). Can be removed for general-purpose use
			if not args.no_depth:
				sign=-1
			
			# Random t (scalars and signs). Zeros and small motions are avoided as before
			cix = (random.random()*0.05+0.05)* ( sign*(-1)**random.randrange(2) )
			ciy = (random.random()*0.05+0.05)* ( sign*(-1)**random.randrange(2) )
			ciz = (random.random()*0.05+0.05)* ( sign*(-1)**random.randrange(2) )
			camerai_mot = [cix, ciy, ciz]

			# Random Euler angles (scalars and signs). Zeros and small rotations are avoided as before
			aix = (random.random()*math.pi / 72.0 + math.pi / 72.0) * ( sign*(-1)**random.randrange(2) )
			aiy = (random.random()*math.pi / 72.0 + math.pi / 72.0) * ( sign*(-1)**random.randrange(2) )
			aiz = (random.random()*math.pi / 72.0 + math.pi / 72.0) * ( sign*(-1)**random.randrange(2) )
			camerai_ang = [aix, aiy, aiz]

			ai =  torch.from_numpy(np.array([[camerai_ang]], dtype=np.float32))
			tri = torch.from_numpy(np.array([[camerai_mot]]))

			# Compute (R|t)
			Ti = transformation_from_parameters(axisangle + ai, translation + tri)
			
			# Apply transformation T_{0->\pi_i}
			pi, zi = project_3d(cam_points, K, Ti)
			
			# If a pixel belongs to object label l, replace coordinates in I1...
			p1[instances[l] > 0] = pi[instances[l] > 0]

			# ... and its depth
			zi = zi.reshape(1, h, w)
			z1[instances[l][:, :, :, 0] > 0] = zi[instances[l][:, :, :, 0] > 0]

	# Bring p1 coordinates in [0,W-1]x[0,H-1] format 
	p1 = (p1 + 1) / 2
	p1[:, :, :, 0] *= w - 1
	p1[:, :, :, 1] *= h - 1

	# Create auxiliary data for warping
	dlut = torch.ones(1, h, w).float() * 1000
	safe_y = np.maximum(np.minimum(p1[:, :, :, 1].long(), h - 1), 0)
	safe_x = np.maximum(np.minimum(p1[:, :, :, 0].long(), w - 1), 0)
	warped_arr = np.zeros(h*w*5).astype(np.uint8)
	img = rgb.reshape(-1)

	# Call forward warping routine (C code)
	warp( c_void_p(img.numpy().ctypes.data), c_void_p(safe_x[0].numpy().ctypes.data), c_void_p(safe_y[0].numpy().ctypes.data), c_void_p(z1.reshape(-1).numpy().ctypes.data), c_void_p(warped_arr.ctypes.data), c_int(h), c_int(w))
	warped_arr = warped_arr.reshape(1,h,w,5).astype(np.uint8)

	# Warped image
	im1_raw = warped_arr[0,:,:,0:3]
	
	# Validity mask H
	masks["H"] = warped_arr[0,:,:,3:4]
	
	# Collision mask M
	masks["M"] = warped_arr[0,:,:,4:5]
	# Keep all pixels that are invalid (H) or collide (M)
	masks["M"] = 1-(masks["M"]==masks["H"]).astype(np.uint8)

	# Dilated collision mask M'
	kernel = np.ones((3,3),np.uint8)
	masks["M'"] = cv2.dilate(masks["M"],kernel,iterations = 1)
	masks["P"] = (np.expand_dims(masks["M'"], -1) == masks["M"]).astype(np.uint8)
	
	# Final mask P
	masks["H'"] = masks["H"]*masks["P"]

	# Compute flow as p1-p0
	flow_01 = p1 - p0

	# Get 16-bit flow (KITTI format) and colored flows
	flow_16bit = cv2.cvtColor( np.concatenate((flow_01 * 64. + (2**15), np.ones_like(flow_01)[:,:,:,0:1]), -1)[0], cv2.COLOR_BGR2RGB )
	flow_color = flow_to_color(flow_01[0].numpy(), convert_to_bgr=True)

	im1 = cv2.inpaint(im1_raw, 1 - masks[args.mask_type], 3, cv2.INPAINT_TELEA)

	# Save images
	cv2.imwrite(os.path.join("dCOCO","im0","95022.jpg"), rgb[0].numpy())
	cv2.imwrite(os.path.join("dCOCO","im1_raw","95022_%02d.jpg"%(idm)), im1_raw)
	cv2.imwrite(os.path.join("dCOCO","im1","95022_%02d.jpg"%(idm)), im1)	
	cv2.imwrite(os.path.join("dCOCO","flow","95022_%02d.png"%(idm)), flow_16bit.astype(np.uint16))
	cv2.imwrite(os.path.join("dCOCO","H","95022_%02d.png"%(idm)), masks["H"]*255)
	cv2.imwrite(os.path.join("dCOCO","M","95022_%02d.png"%(idm)), masks["M"]*255)
	cv2.imwrite(os.path.join("dCOCO","M'","95022_%02d.png"%(idm)), masks["M'"]*255)
	cv2.imwrite(os.path.join("dCOCO","P","95022_%02d.png"%(idm)), masks["P"]*255)
	cv2.imwrite(os.path.join("dCOCO","H'","95022_%02d.png"%(idm)), masks["H'"]*255)
	cv2.imwrite(os.path.join("dCOCO","flow_color","95022_%02d.png"%(idm)), flow_color)
	plt.imsave(os.path.join("dCOCO","depth_color","95022_%02d.png"%(idm)), 1./depth[0].detach().numpy(), cmap="magma")
	if args.segment:
		plt.imsave(os.path.join("dCOCO","instances_color","95022_%02d.png"%(idm)), instances_mask, cmap="magma")	

	# Clear cache and update progress bar
	ctypes._reset_cache() 
	pbar.update(1)

# Close progress bar, cya!
pbar.close()
