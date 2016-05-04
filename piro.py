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
from skimage.draw import circle_perimeter, line, line_aa
from skimage.feature import corner_harris, corner_subpix, corner_peaks, corner_fast
from skimage.morphology import disk, dilation, diamond
from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)
from skimage.feature import canny
import math
from skimage import data, exposure, img_as_float
from scipy import stats
from scipy.stats import chisquare

img_curve_info = []
hist_dict = {}

def get_args():
  if len(sys.argv) < 3:
    print 'not enough args'
    exit(0)
  return sys.argv[1], sys.argv[2]  
    
  
def loop_on_dir(directory, img_count):
  hist_dict = {}
  os.chdir(directory)
  for i, file in enumerate(glob.glob("*.png")):
    if i == int(img_count):
      break
    print "[" + str(i) + "] " + str(file)
    process_img(file)
  compare_hist()
  
def process_img(i):
  fig, ax = plt.subplots()
  img = io.imread(i, as_grey=True)
  white_count = np.sum(img > 0)
  simg = sobel(img)
  gauss_img = gaussian(img, sigma=1)
  contours = measure.find_contours(gauss_img, 0.3)
  int_contour = []
  for n, contour in enumerate(contours):
    for point in contour:
      int_contour.append((round(point[0]), round(point[1])))
  int_contour = list(set(int_contour))
  black_img = img[:]
  black_img[:] = 0
  for point in int_contour: #contour
    black_img[point[0]][point[1]] = 1
  black_img = dilation(black_img, diamond(1))
  coords = corner_peaks(corner_harris(img), min_distance=1)
  ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
  plt.plot(coords[:, 1], coords[:, 0], '.r', markersize=5)
  sy, sx = img.shape
  print sy, sx
  cfiltered = get_extend_coord(img.shape, sy, sx, coords) #filtrowanie najbardziej wystajacych
  tmp_perim = simg
  angles = {}
  for c in cfiltered:
    cy, cx = c
    print " "
    iter_p = get_mean_coords(black_img, cy, cx, 25)
    print "Punkty sasiadujace"
    #print len(iter_points)
    print iter_p[0]
    print iter_p[1]
    try:
      angle = get_angle(c, iter_p[0], iter_p[1])
      print "ANGLE: " + str(angle)
    except ValueError:
      print "ANGLE_ERROR"
    angles[c] = abs(95 - angle)
    #tmp_perim += c_perim
  #plt.imshow(tmp_perim, cmap=plt.get_cmap('gray'))
  #print "Base_vertices:"
  v0, v1 = get_base_vertices(angles)
  black_img_line = img[:]
  black_img_line[:] = 0
  for point in int_contour: #nakladanie konturu
    black_img_line[point[0]][point[1]] = 1
  #black_img = dilation(black_img, diamond(1))
  remove_lines = probabilistic_hough_line(black_img_line, threshold=10, line_length=15, line_gap=2)
  max_line_length = 0
  black_img_line, max_line_length = filter_contours_lines(v0, v1, remove_lines, black_img_line, 6) #filtruje kontur z linii o poczatkach w okolicach punktow
  print "MAX_LINE_LENGTH: ", max_line_length
  print "DIMENSION: ", white_count/max_line_length
  #cnt = 0
  #plt.show()
  print "HISTOGRAM"
  tmp_hist = []
  tmpR = int(math.ceil(max_line_length/40))
  print "R for image is: ", tmpR
  print type(tmpR)
  for point in int_contour: #nakladanie konturu
    if black_img_line[point[0]][point[1]] == 1:
      #tmp_mean_coords = get_mean_coords(black_img, int(point[0]), int(point[1]), 4) #pobiera srednie wspolrzedne
      #print tmpR
      tmp_mean_coords = get_mean_coords(black_img, int(point[0]), int(point[1]), tmpR) #pobiera srednie wspolrzedne
      if len(tmp_mean_coords) < 2:
        print "not enough mean cords ", tmp_mean_coords
        continue
      try:
        angle = get_angle(point, tmp_mean_coords[0], tmp_mean_coords[1])
        tmp_hist.append(angle)
      except ValueError:
        print "ANGLE_ERROR"
        #result = 0  
      #print "HIST_ANGLE: " + str(angle)
      #angles[point] = abs(95 - angle)
  #plt.hist(gaussian_numbers)
  hist, bins = np.histogram(tmp_hist, normed = True, bins = 10)
  hist_dict[i] = hist
  #plt.plot()
  #plt.show()
  #plt.hist()
  
  print "KONIEC"
  #plt.imshow(black_img_line, cmap=plt.cm.gray)
  #plt.show()
  
def chi2_distance(histA, histB, eps = 1e-10):
	d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
		for (a, b) in zip(histA, histB)])
	return d
  
  
def compare_hist():
  print "compering histograms", len(hist_dict)
  for (k1, testHist) in hist_dict.items():
    results = {}
    for (k2, tmpHist) in hist_dict.items():
      if k1 == k2:
        continue
      #print "comparing: ", k1, " with ", k2
      #d = chi2_distance(testHist, tmpHist)
      #d, a = stats.ks_2samp(testHist, tmpHist)
      d, a = chisquare(testHist, tmpHist)
      results[k2] = d
    results = sorted([(v, k) for (k, v) in results.items()])
    print "iter: ", k1
    for r in results:
      print "   ", r
  '''
  results = {}
  for (k, hist) in hist_dict.items():
    d = chi2_distance(index["doge.png"], hist)
    results[k] = d
  results = sorted([(v, k) for (k, v) in results.items()])
  '''
  #for thist in hist_dict:
  #  print thist
  
def get_mean_coords(black_img, cy, cx, radius):
  #print "CY CX: " + str(cy) + "  " + str(cx)
  c_perim = get_circle_perimeter(black_img.shape, cy, cx, radius)
  tmp_points = [] #lista kandydatow na punkty przecinajace sie z okregiem
  sy, sx = black_img.shape
  for j in range(sy):
    for i in range(sx):
      if black_img[j][i] * c_perim[j][i] > 0.0: 
        tmp_points.append((j,i))

  iter_points = []
  iter_p = []
  for a in tmp_points:
    iter_points.append(a)
  for i in range(len(tmp_points)):
    if tmp_points[i] not in iter_points:
      continue
    new_point = [tmp_points[i][0], tmp_points[i][1]]
    count = 1
    for j in reversed(range(len(iter_points))):
      if tmp_points[i] != iter_points[j]: #nie badam dystansu sam do siebie
        if distance(tmp_points[i], iter_points[j]) <= 3: #if distance < 2 then delete
          new_point[0] += iter_points[j][0]
          new_point[1] += iter_points[j][1]
          count += 1
          iter_points.remove(iter_points[j])
      else:
        iter_points.remove(tmp_points[i])
    iter_p.append((round(new_point[0]/count), round(new_point[1]/count)))
  #print iter_p
  return iter_p  
  
def filter_contours_lines(v0, v1, lines, img, dist): #linie zaczynajace sie w danym punkcie w otoczeniu dist pikseli
  print "v0, v1:" , v0, v1
  max_line_length = 0
  for tmp_line in lines:
    tmp0, tmp1 = tmp_line
    tp0 = (tmp0[1], tmp0[0])
    tp1 = (tmp1[1], tmp1[0])
    if distance(tmp0, tmp1) > max_line_length:
      max_line_length = distance(tmp0, tmp1)
      print "NEW_MAX:", tmp0, tmp1
    print "tp0 tp1: " ,tp0, tp1
    print "v0 tp0", distance(v0, tp0)
    print "v0 tp1", distance(v0, tp1)
    print "v1 tp0", distance(v1, tp0)
    print "v1 tp1", distance(v1, tp1)
    if distance(v0, tp0) < dist or distance(v0, tp1) < dist or distance(v1, tp0) < dist or distance(v1, tp1) < dist:
      print "removing line"
      rr, cc, v = line_aa(tmp0[1], tmp0[0], tmp1[1], tmp1[0])
      img[rr,cc] = 0
  return img, max_line_length
  
def get_base_vertices(angles):
  v0 = min(angles, key = angles.get)
  del angles[v0]
  v1 = min(angles, key = angles.get)
  return v0, v1
  
def get_extend_coord(size, sy, sx, coords): #zwraca liste najbardziej wystajacych wierzcholkow
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
  result = []
  result.append(left)
  result.append(right)
  result.append(up)
  result.append(down)
  if len(left_supp) > 0: result.append(left_supp)
  if len(right_supp) > 0: result.append(right_supp)
  if len(up_supp) > 0: result.append(up_supp)
  if len(down_supp) > 0: result.append(down_supp)
  return set(result)
  
def get_circle_perimeter(size, y, x, r): #zwraca czarny obraz z wyrysowanym okregiem o zadanych parametrach
  max_y, max_x = size
  w = np.zeros(size)
  rr, cc = circle_perimeter(y, x, r)
  trr = np.array([0])
  tcc = np.array([0])
  for i, r in enumerate(rr):
    if r >= 0 and r < max_y:
      if cc[i] >= 0 and cc[i] < max_x:
        trr = np.concatenate((trr, [r]))
        tcc = np.concatenate((tcc, [cc[i]]))
  w[trr, tcc] = 1
  return w
  
def distance(p0, p1): #oblica odleglosc pomiedzy punktami
  return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)    
  
def get_angle(corner, p1, p2):
  vectorA = (corner[0] - p1[0], corner[1] - p1[1])
  vectorB = (corner[0] - p2[0], corner[1] - p2[1])
  result = math.acos((vectorA[0]*vectorB[0] + vectorA[1]*vectorB[1]) / (distance(corner, p1) * distance(corner, p2)))
  return math.degrees(result)  
  
def get_angle2(corner, p1, p2):
  vectorA = (corner[0] - p1[0], corner[1] - p1[1])
  vectorB = (corner[0] - p2[0], corner[1] - p2[1])
  try: 
    result = math.acos((vectorA[0]*vectorB[0] + vectorA[1]*vectorB[1]) / (distance(corner, p1) * distance(corner, p2)))
  except ValueError:
    print "ANGLE_ERROR"
    result = 0
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
'''
hist, bins = np.histogram([1,2,3,4,5,4,3,4,5,2,2,3,4,5,1,1,1,2,3,3,4,5,3,4,5,3,3,4,6,4,5,6,7,4,6,6,67,7,7,4,4,56,4,4,6,4,6,4,5], normed = True, bins = 10)
hist2, bins2 = np.histogram([1,2,3,4,5,4,3,4,5,9,92,3,4,5,64,4,4,7,3,3,4,5,3,4,59,38,3,14,6,4,5,6,7,4,6,6,67,7,7,4,4,56,4,4,6,4,6,4,5], normed = True, bins = 10)
hist3, bins = np.histogram([1,2,3,4,5,4,33,32,11,23,99,67,44,23,25,67,89,4,90,89,78,64,33,56,67,4,5,6,7,4,68,6,67,7,7,4,4,56,4,4,6,4,6,4,5], normed = True, bins = 10)
hist4, bins = np.histogram([1,2,3,4,5,6,7,8,9,10,67,44,23,25,67,89,4,90,89,78,64,33,56,67,4,5,6,7,4,68,6,67,7,7], normed = True, bins = 10)
hist5, bins = np.histogram([1,2,3,4,5,6,7,8,9,11], normed = True, bins = 10)
hist6, bins = np.histogram([1,2,3,4,5,6,7,8,9,12], normed = True, bins = 10)
hist7, bins = np.histogram([1,2,3,4,5,6,7,8,9,15], normed = True, bins = 10)

c1 = chisquare(hist)
c2 = chisquare(hist2)
test = stats.ks_2samp(c1, c2)
test2 = stats.ks_2samp(c2, c1)

print c1
print c2
print test
print test2

print "===================="

hist_dict["a"] = hist
hist_dict["b"] = hist2
hist_dict["c"] = hist3
hist_dict["d"] = hist4
hist_dict["e"] = hist5
hist_dict["f"] = hist6
hist_dict["g"] = hist7
compare_hist()
'''


#s1, s2 = np.histogram([1, 2, 3, 4, 3, 1], normed = True, bins = 5)
#print s1
#print "========="

#sam1 = np.histogram([1, 2, 3, 4,
#result = stats.ks_2samp(sam1, sam2)



