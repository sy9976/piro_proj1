import sys
import glob, os
import numpy as np
import random
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.data as data
import skimage.color as color
import skimage
from skimage import io
import ipywidgets
import skimage.filters
from skimage.filters import gaussian
from scipy import misc, ndimage
from skimage import measure
from skimage.feature import corner_harris, corner_subpix, corner_peaks, corner_fast
from skimage.morphology import disk, dilation
from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)
from skimage.feature import canny
import math
from skimage import data, exposure, img_as_float

img_curve_info = []

def get_args():
  if len(sys.argv) < 3:
    print 'not enough args'
    exit(0)
  return sys.argv[1], sys.argv[2]  
    
  
def loop_on_dir(directory, img_count):
  os.chdir(directory)
  for i, file in enumerate(glob.glob("*.png")):
    if i == int(img_count):
      break
    print "[" + str(i) + "] " + str(file)
    process_img(file)
  
def process_img(i):
  #face = misc.ascent()
  #misc.imsave('face.png', face) 
  #plt.imshow(img, cmap=plt.cm.gray)
  #x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
  #r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))
  img = io.imread(i, as_grey=True)
  #edges = canny(img, 3, 1, 25)
  #lines = probabilistic_hough_line(edges, threshold=10, line_length=15, line_gap=10)
  #dil_img = dilation(edges, disk(2))
  #plt.imshow(dil_img, cmap=plt.cm.gray)
  
  #for line in lines:
  #  p0, p1 = line
  #  #dist = math.hypot(p0[0] - p0[1], p1[0] - p1[1])
  #  #print dist
  #  plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
  #plt.show()
  #dil_img = dilation(img, disk(5))
  # Find contours at a constant value of 0.8
  gauss_img = gaussian(img, sigma=1)
  contours = measure.find_contours(gauss_img, 0.3)
  fig, ax = plt.subplots()
  for n, contour in enumerate(contours):
    print contour.shape
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
  #plt.show()
  coords = corner_peaks(corner_harris(gauss_img), min_distance=1)
  #coords = corner_peaks(corner_fast(img, 6), min_distance=1)
  #coords_subpix = corner_subpix(img, coords, window_size=13)
  
  #ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
  #ax.plot(coords[:, 1], coords[:, 0], '.r', markersize=15)
  #ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
  plt.plot(coords[:, 1], coords[:, 0], '.r', markersize=5)
  #plt.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
  plt.show()
  #filtered_coords = corner_harris(img, min_distance=6)
  #plot_harris_points(img, filtered_coords)  

def rotate_by_points(start, end):
  angle = int(math.atan((start[0]-start[1])/(end[0]-end[1]))*180/math.pi)
  
  
def plot_harris_points(image, filtered_coords):
  plt.plot()
  plt.imshow(image)
  plt.plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], 'b.')
  plt.axis('off')
  plt.show()

  
def histogram():
  image = img_as_float(data.camera())
  np.histogram(image, bins=2)
  tmp = exposure.histogram(image, nbins=2)
  print tmp
  


  
directory, img_count = get_args()
print "directory: " + directory
loop_on_dir(directory, img_count)
#histogram()


# Display the image and plot all contours found
'''
fig, ax = plt.subplots()
ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)

for n, contour in enumerate(contours):
  ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
'''
#img = io.imread(i, as_grey=True)
#plt.imshow(img, cmap=plt.cm.gray)
#plt.show()
'''
face = np.flipud(face[150:350, 150:350])
face = ndimage.rotate(face, 45)
face = ndimage.gaussian_filter(face, 3)
plt.imshow(face, cmap=plt.cm.gray)
plt.show()
'''
#for file in os.listdir("/mydir"):
#  if file.endswith(".txt"):
#    print(file)
    
'''
sigma_slider = ipywidgets.FloatSlider(min=0, max=50, step=1, value=1)
@ipywidgets.interact(sigma=sigma_slider)
def gaussian_demo(sigma=5):
  io.imshow(skimage.filters.gaussian_filter(img, sigma))
'''
