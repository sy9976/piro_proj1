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
from skimage.filters import gaussian, sobel
from scipy import misc, ndimage
from skimage import measure
from skimage.draw import circle_perimeter
from skimage.feature import corner_harris, corner_subpix, corner_peaks, corner_fast
from skimage.morphology import disk, dilation, diamond
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
    if file == "0.png":
      process_img(file)
  
def process_img(i):
  fig, ax = plt.subplots()
  img = io.imread(i, as_grey=True)
  simg = sobel(img)
  gauss_img = gaussian(img, sigma=1)
  contours = measure.find_contours(gauss_img, 0.3)
  int_contour = []
  for n, contour in enumerate(contours):
    #ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    for point in contour:
      int_contour.append((round(point[0]), round(point[1])))
  
  int_contour = list(set(int_contour))
  #print int_contour
  black_img = img[:]
  black_img[:] = 0
  for point in contour:
    black_img[point[0]][point[1]] = 1
  #plt.imshow(black_img, cmap=plt.cm.gray)
  #plt.show()
  black_img = dilation(black_img, diamond(1))
  #gauss_img = gaussian(img, sigma=1)
  #coords = corner_peaks(corner_harris(gauss_img), min_distance=1)
  coords = corner_peaks(corner_harris(img), min_distance=1)
  #fig, ax = plt.subplots()
  ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
  plt.plot(coords[:, 1], coords[:, 0], '.r', markersize=5)
  #circ=plt.Circle((200,200), radius=10, color='g', fill=False)
  r = 3
  sy, sx = img.shape
  print sy, sx
  cfiltered = get_extend_coord(img.shape, sy, sx, coords) #filtrowanie najbardziej wystajacych
  tmp_perim = simg
  
  for c in cfiltered:
    cy, cx = c
    print " "
    print "CY CX: " + str(cy) + "  " + str(cx)
    c_perim = get_circle_perimeter(img.shape, cy, cx, 25)
    merge = []
    tmp_points = [] #lista kandydatow na punkty przecinajace sie z okregiem
    for j in range(sy):
      for i in range(sx):
        #if simg[j][i] * c_perim[j][i] > 0.3: print j, i
        if black_img[j][i] * c_perim[j][i] > 0.0: 
          #print j, i
          tmp_points.append((j,i))
          
    #print "po kopiowaniu"
    iter_points = []
    iter_p = []
    for a in tmp_points:
      iter_points.append(a)
    #print type(iter_points)
    for i in range(len(tmp_points)):
      if tmp_points[i] not in iter_points:
        #print "punkt juz usuniety: " + str(tmp_points[i])
        continue
      #print "iteruje dla: " + str(p)
      new_point = [tmp_points[i][0], tmp_points[i][1]]
      count = 1
      for j in reversed(range(len(iter_points))):
	#print j
        #print tmp_points[i]
        #print iter_points[j]
	#print "AAAAAA", iter_points
        if tmp_points[i] != iter_points[j]: #nie badam dystansu sam do siebie
          if distance(tmp_points[i], iter_points[j]) <= 3: #if distance < 2 then delete
            new_point[0] += iter_points[j][0]
	    new_point[1] += iter_points[j][1]
	    count += 1
	    #print "Do usuniecia: " + str(iter_points[j])
            iter_points.remove(iter_points[j])
	    #j -= 2
	else:
	  iter_points.remove(tmp_points[i])
	    #tmp_points.remove(cmpp)
      #print "dodaje ", (round(new_point[0]/count), round(new_point[1]/count))
      iter_p.append((round(new_point[0]/count), round(new_point[1]/count)))
    
    print "Punkty sasiadujace"
    print len(iter_points)
    #for p in iter_points:
    #  print p
    print iter_p[0]#, iter_p[0]
    print iter_p[1]#, iter_p[1]
    angle = get_angle(c, iter_p[0], iter_p[1])
    print "ANGLE: " + str(angle)
    
    
    
    
    
    
    
    '''
    iter_points = []
    for a in tmp_points:
      iter_points.append(a)
    #print type(iter_points)
    for p in tmp_points:
      if p not in iter_points:
        #print "punkt juz usuniety: " + str(p)
        continue
      #print "iteruje dla: " + str(p)
      for cmpp in iter_points:
        #print p
        #print cmpp
        if p != cmpp: #nie badam dystansu sam do siebie
          if distance(p, cmpp) <= 3: #if distance < 2 then delete
            #print "Do usuniecia: " + str(cmpp)
            iter_points.remove(cmpp)
    
    print "Punkty sasiadujace"
    #for p in iter_points:
    #  print p
    print iter_points[0]
    print iter_points[1]
    angle = get_angle(c, iter_points[0], iter_points[1])
    print "ANGLE: " + str(angle)
    '''
    
    #merge = [c_perim == simg]
    #tmp = simg + c_perim
    tmp_perim += c_perim
  plt.imshow(tmp_perim, cmap=plt.get_cmap('gray'))
    #print y, x
    #a = plt.axes([0, sy, 0, sx])
    #circ=plt.Circle((x,y), radius=3, color='g', fill=False)
    #plt.gca().add_patch(circ)
    #plt.axis([0, sx, 0, sy])
    #plt.show()
    #w = np.zeros(size)
    #rr, cc = circle_perimeter(y, x, r)
    #w[rr, cc] = 1
    #plt.imshow(w, cmap=plt.get_cmap('gray'))
    #plt.plot(w[:, 1], w[:, 0], '.b', markersize=1)
    #for j in range(size[0]):
    #  for i in range(size[1]):
    #    if w[j][i] * img[j][i] == 1:
    #      print j, i
    #tmp = [w*img == 1]
    #plt.imshow(w, cmap=plt.get_cmap('gray'))
    #plt.plot(w)
    #print np.sum(tmp)
    #circ=plt.Circle((x,y), radius=3, color='g', fill=False)
    #plt.gca().add_patch(circ)
  #ax.add_patch(circ)
  #print circ
  #plt.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
  plt.show()
  
def distance(p0, p1):
  return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)  
  
def get_extend_coord(size, sy, sx, coords):
  left_supp = ()
  right_supp = ()
  up_supp = ()
  down_supp = ()
  left = (0, sx)
  up = (sy, 0)
  down = (0, 0)
  right = (0, 0)
  for y, x in coords:
    if y == down[0] and len(down_supp) == 0:
      down_supp = (y,x)
    if y > down[0]:
      down = (y,x)
      down_supp = ()
    if y == up[0] and len(up_supp) == 0:
      up_supp = (y,x)
    if y < up[0]:
      up = (y,x)
      up_supp = ()
    if y == right[1] and len(right_supp) == 0:
      right_supp = (y,x)
    if x > right[1]:
      right = (y,x)
      right_supp = ()
    if y == left[1] and len(left_supp) == 0:
      left_supp = (y,x)
    if x < left[1]:
      left = (y,x)
      left_supp = ()
  '''
  print "left:  "  + str(left)
  print "right: "  + str(right)
  print "up:    "  + str(up)
  print "down:  "  + str(down)
  print "leftSup:  "  + str(left_supp)
  print "rightSup: "  + str(right_supp)
  print "upSup:    "  + str(up_supp)
  print "downSup:  "  + str(down_supp)
  '''
  result = []
  result.append(left)
  result.append(right)
  result.append(up)
  result.append(down)
  if len(left_supp) > 0: result.append(left_supp)
  if len(right_supp) > 0: result.append(right_supp)
  if len(up_supp) > 0: result.append(up_supp)
  if len(down_supp) > 0: result.append(down_supp)
  #return np.unique(result)
  return set(result)
  
def get_circle_perimeter(size, y, x, r):
  max_y, max_x = size
  #print "size: " + str(size)
  w = np.zeros(size)
  #print "wsize: " + str(w.shape)
  rr, cc = circle_perimeter(y, x, r)
  trr = np.array([0])
  tcc = np.array([0])
  for i, r in enumerate(rr):
    if r >= 0 and r < max_y:
      if cc[i] >= 0 and cc[i] < max_x:
        trr = np.concatenate((trr, [r]))
        tcc = np.concatenate((tcc, [cc[i]]))
  w[trr, tcc] = 1
  #print w
  return w
  
  
def get_angle(corner, p1, p2):
  vectorA = (corner[0] - p1[0], corner[1] - p1[1])
  vectorB = (corner[0] - p2[0], corner[1] - p2[1])
  result = math.acos((vectorA[0]*vectorB[0] + vectorA[1]*vectorB[1]) / (distance(corner, p1) * distance(corner, p2)))
  return math.degrees(result)

def rotate_by_points(start, end):
  angle = int(math.atan((start[0]-start[1])/(end[0]-end[1]))*180/math.pi)
  
def plt_to_array(fig):
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  print data
  plt.imshow(data)
  plt.show()
  
def histogram():
  image = img_as_float(data.camera())
  np.histogram(image, bins=2)
  tmp = exposure.histogram(image, nbins=2)
  print tmp
  
directory, img_count = get_args()
print "directory: " + directory
loop_on_dir(directory, img_count)

#fig = plt.figure()
#plt.plot([1,2,3,4,5,6,7,8,9,10])
#circ=plt.Circle((5,5), radius=3, color='g', fill=False)
#plt.gca().add_patch(circ)
#plt.show()
#plt.plot([1,2,3,4])
#fig.tight_layout(pad=0)
#fig.canvas.draw()
#plt_to_array(fig)
#histogram()

