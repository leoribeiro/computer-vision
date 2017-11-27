#!/usr/bin/env python
# -*- coding: utf-8 -*-


## python3


import tkinter as tk
window = tk.Tk()

from PIL import Image, ImageTk
import os, tkinter.filedialog
import tkinter.messagebox
import matplotlib.widgets as widgets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Circle
import math
import copy
import cv2
import matplotlib
import random
import sys
from sklearn.preprocessing import normalize

######### RANSAC

path_images = {}
putative_correspondences = []


def center(toplevel):
    toplevel.update_idletasks()
    w = toplevel.winfo_screenwidth()
    h = toplevel.winfo_screenheight()
    size = tuple(int(_) for _ in toplevel.geometry().split('+')[0].split('x'))
    x = w/2 - size[0]/2
    y = h/2 - size[1]/2
    toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))


def norm_x(x):
  if(x[2] == 0):
    return x
  return np.array([x[0]/x[2],x[1]/x[2],x[2]/x[2]])

def loadImage(image,title):
   global fig_a
   fig_a = plt.figure()
   fig_a.canvas.set_window_title(title)
   ax = fig_a.add_subplot(111)
   arr = np.asarray(image)
   plt.imshow(arr)
   plt.ion()
   plt.show()
   return

def get_points_corners(h,img):
    x = np.dot(h,[0,0,1])
    x1 = norm_x(x)

    x = np.dot(h,[0,img.size[1] - 1,1])
    x2 = norm_x(x)

    x = np.dot(h,[img.size[0] - 1,img.size[1] - 1,1])
    x3 = norm_x(x)

    x = np.dot(h,[img.size[0] - 1,0,1])
    x4 = norm_x(x)

    return x1,x2,x3,x4

def convert_points(h,points):
  xs = []
  for p in points:
    x = np.dot(h,p)
    x = norm_x(x)
    xs.append(x)
  return xs

def generateNewImage(hs,filenames):

  middleIndex = int((len(filenames) - 1)/2)
  print ("middleIndex",middleIndex)
  
  width = 0
  img = []
  for k,f in filenames.items():
    img_ = Image.open(f)
    img.append(img_)
    width += img_.size[0]
    height = img_.size[1]

  xs_l = []

  for plano in range(0,middleIndex):
    h = hs[plano]
    x1,x2,x3,x4 = get_points_corners(h,img[plano])
    xs_l = convert_points(h,xs_l)
    xs_l.extend([x1,x2,x3,x4])

  xs_r = []
  for plano in range(len(hs)-1,middleIndex-1,-1):
    h_inv = np.linalg.inv(hs[plano])
    x1,x2,x3,x4 = get_points_corners(h_inv,img[plano])
    xs_r = convert_points(h_inv,xs_r)
    xs_r.extend([x1,x2,x3,x4])

  xs_ = xs_r + xs_l
  xs = []
  ys = []
  for x in xs_:
    xs.append(x[0])
    ys.append(x[1])
    

  xs.append(0)
  xs.append(img[middleIndex].size[0] - 1)
  ys.append(0)
  ys.append(img[middleIndex].size[1] - 1)

  min_x = min(xs)
  min_y = min(ys)
  max_x = max(xs)
  max_y = max(ys)

  #print (min_x,max_x)
  #print (min_y,max_y)

  ratio = (max_x - min_x, max_y - min_y)

  n_width = width
  n_height = int(n_width * (ratio[1] / ratio[0]))

  new_image = Image.new('RGB', (n_width, n_height))

  step_y = (max_y - min_y)/n_height
  step_x = (max_x - min_x)/n_width
  x_cm = min_x
  #print ("width",width)
  #print ("height",height)
  for x in range(n_width):
    y_cm = min_y
    for y in range(n_height):
      for k in range(0,len(hs)+1):
        if(k == middleIndex):
          put_pixel(x_cm,y_cm,x,y,img[k],new_image)
        else:
          h = apply_matrix_recursive(k,middleIndex,hs)
          coords = np.dot(h,[x_cm,y_cm,1])
          coords = norm_x(coords)
          put_pixel(coords[0],coords[1],x,y,img[k],new_image)
      y_cm += step_y
    x_cm += step_x

  return new_image

def apply_matrix_recursive(k,m,hs):
  if(k > m):
    h_ = hs[k-1]
    for h in range(k-2,m-1,-1):
      h_ = np.dot(h_,hs[h])
  else:
    h_ = np.linalg.inv(hs[k])
    for h in range(k+1,m):
      h_inv = np.linalg.inv(hs[h])
      h_ = np.dot(h_,h_inv)
  return h_




def put_pixel(x_,y_,x,y,image,new_image):
  try:
    new_pixel = image.getpixel((x_,y_))
    new_image.putpixel((x,y),new_pixel)
  except IndexError:
    pass

def drawMatches(img1, kp1, img2, kp2, matches):

    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
    out[:rows1,:cols1] = np.dstack([img1])
    out[:rows2,cols1:] = np.dstack([img2])
    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0, 1), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0, 1), 1)
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0, 1), 1)

    return out

def normalizeMatches(matches):
  matches = sorted(matches, key=lambda val: val.distance)
  matches_ = []
  for m in matches:
    if(m.distance < 100):
      matches_.append(m)
  return matches_


def compare(filenames):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    kp = []
    des = []
    img = []
    for k,f in filenames.items():
      img_ = cv2.imread(f)
      img.append(img_)
      # find the keypoints and descriptors with SIFT
      kp_, des_ = sift.detectAndCompute(img_,None)
      kp.append(kp_)
      des.append(des_)

    # BFMatcher with default params
    bf = cv2.BFMatcher()

    matches = []
    for n,d in enumerate(des):
      try:
        matches_ = bf.match(des[n],des[n+1])
        matches_ = normalizeMatches(matches_)
        matches.append(matches_)
      except IndexError:
        pass


    get_putative_correspondences(matches,kp)

    # for m in matches_[-25:]:
    #  print (m.distance)

    # print ("Matches < 200:",len(matches_))

    # img3 = drawMatches(img1,kp1,img2,kp2,matches_[-25:])
    # img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    # fig, ax = plt.subplots(num=None, figsize=(16, 6), dpi=80, facecolor='w', edgecolor='k')
    # fig.subplots_adjust(bottom = 0)
    # fig.subplots_adjust(top = 1)
    # fig.subplots_adjust(right = 1)
    # fig.subplots_adjust(left = 0)
    # plt.imshow(img3) 
    # plt.ion()
    # plt.show()

    return

def returnMatrix(x,x_):
  line1 = [x_[0]*x[0],x_[0]*x[1],x_[0],x_[1]*x[0],x_[1]*x[1],x_[1],x[0],x[1],1]
  return line1

def get_poits_random(num_points,correspondences):
  coords = []
  cont = 0

  while cont < num_points:
    m = random.choice(correspondences)
    (x1,y1) = m[0]
    (x2,y2) = m[1]
    coords.append([(x1,y1),(x2,y2)])
    cont += 1

  return np.array(coords)

def get_putative_correspondences(matches,kp):
  global putative_correspondences

  cont = 0
  cont_k = 0
  for m_ in matches:
    putative_correspondences.append([])
    for m in m_:
      img1_idx = m.queryIdx
      img2_idx = m.trainIdx
      (x1,y1) = kp[cont][img1_idx].pt
      (x2,y2) = kp[cont+1][img2_idx].pt
      putative_correspondences[cont].append([(x1,y1),(x2,y2)])
    cont += 1

def centeroidnp(arr):
    arr = np.array(arr)
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

def norm_poits(coords):
  coords0 = [i[0] for i in coords]
  coords1 = [i[1] for i in coords]
  t0 = get_transform(coords0)
  t1 = get_transform(coords1)

  n_coords = []
  for k,c in enumerate(coords0):
    c0 = np.dot(t0,[coords0[k][0],coords0[k][1],1])
    c1 = np.dot(t1,[coords1[k][0],coords1[k][1],1])
    c0 = norm_x(c0)
    c1 = norm_x(c1)
    n_coords.append([(c0[0],c0[1]),(c1[0],c1[1])])
  
  return n_coords,t0,t1  


def get_transform(coords):
  c = centeroidnp(coords)
  c = [c[0],c[1],1]

  sum_ = 0
  for c_ in coords:
    sum_ += np.sqrt(np.power(c_[0] - c[0],2)+np.power(c_[1] - c[1],2))
  s = np.sqrt(2) / (1.0/len(coords) * sum_)
  
  tx = -s*c[0]
  ty = -s*c[1]

  t = [[s,0,tx],[0,s,ty],[0,0,1]]

  return t

def calc_ps(F):
  U, s, V = np.linalg.svd(F, full_matrices=True)
  e = V[-1]
  e_ = U[:,-1]



  P = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
  P_ = np.dot([[e_[0],e_[1],0],[-e_[1],e_[0],0],[0,0,1]],F)
  P_ = [[P_[0][0],P_[0][1],P_[0][2],e_[0]],
  [P_[1][0],P_[1][1],P_[1][2],e_[1]],
  [P_[2][0],P_[2][1],P_[2][2],e_[2]]]

  P = np.array(P)
  P_ = np.array(P_)

  return P,P_

def calc_matrix(correspondences):
  N = get_num_poits()
  coords = get_poits_random(N,correspondences)
  coords,t0,t1 = norm_poits(coords)
  
  A = [] 
  for c in coords:
    a = returnMatrix([c[0][0],c[0][1]],[c[1][0],c[1][1]])
    A.append(a)
  A = np.array(A)

  U, s, V = np.linalg.svd(A, full_matrices=True)

  

  c = V[-1]
  F = np.array([[c[0],c[1],c[2]],[c[3],c[4],c[5]],[c[6],c[7],c[8]]])

  U, s, V = np.linalg.svd(F, full_matrices=True)

  s_ = np.diag([s[0],s[1],0])
  #print ("U",U)
  #print ("s",s)
  #print ("s_",s_)
  #print ("V",V)

  F = np.dot(np.dot(U,s_),V)

  F = np.dot(np.dot(np.transpose(t1),F),t0)
  
  return F

def symmetric_transfer_error(x,x_,h,h_inv):
  x = np.array(x)
  x_ = np.array(x_)
  d1 = np.linalg.norm(x-norm_x(np.dot(h_inv,x_)))
  d2 = np.linalg.norm(x_-norm_x(np.dot(h,x)))
  return d1 + d2

def triangulation(x,x_,P,P_):
  #t = [[1,0,-x[0]],[0,1,-x[1]],[0,0,1]]
  #t_ = [[1,0,-x_[0]],[0,1,-x_[1]],[0,0,1]]
  #F = np.dot(np.dot(np.linalg.inv(np.transpose(t_)),F),np.linalg.inv(t))
  #a = np.array(F)
  #b = np.array([0,0,0])
  #e = np.linalg.solve(a,b)
  #a = np.array(np.transpose(F))
  #e_ = np.linalg.solve(a,b)
  #print ("e",e)
  #print ("e_",e)
  #e = [ e[0] / normalize([[e[0]],[e[1]]]), e[1] / normalize([[e[0]],[e[1]]]), e[2]]
  #e = [ e_[0] / normalize([[e_[0]],[e_[1]]]), e_[1] / normalize([[e_[0]],[e_[1]]]), e_[2]]
  #print ("e",e)
  #print ("e_",e)

  #print ("F",F)

  #print ("x",x,"x_",x_,"P[2]",P[2])
  A = [x[0]*P[2] - P[0],
       x[1]*P[2] - P[1],
       x_[0]*P_[2] - P_[0],
       x_[1]*P_[2] - P_[1]]
  A = np.array(A)
  #print ("A",A)
  U, s, V = np.linalg.svd(A, full_matrices=True)

  c = V[-1]
  X = np.array([c[0],c[1],c[2],c[3]])

  #print ("X",X)
  #print ("P",P)

  x_v = np.dot(P,X)
  x_v_ = np.dot(P_,X)

  #print ("x",x)
  #print ("x_",x_)
  #print ("x_v",x_v)
  #print ("x_v_",x_v_)
  return x_v,x_v_


def geometric_error(x,x_,p,p_):
  x_v,x_v_ = triangulation(x,x_,p,p_)
  d1 = np.linalg.norm(x-norm_x(x_v))
  d2 = np.linalg.norm(x_-norm_x(x_v_))
  #d1 = np.linalg.norm(x-x_v)
  #d2 = np.linalg.norm(x_-x_v_)
  return d1 + d2


def get_num_poits():
  return int(e3.get())

def get_threshold():
  return int(e2.get())


def get_inliers(F,correspondences,p,p_):
  inliers = 0
  threshold = get_threshold()
  for c in correspondences:
    x = [c[0][0],c[0][1],1]
    x_ = [c[1][0],c[1][1],1]
    d = geometric_error(x,x_,p,p_)
    #print ("d",d)
    if d < threshold:
      inliers += 1

  return inliers


def get_n_samples(s,e,p):
  return np.log(1.0-p)/np.log(1-np.power((1-e),s))


def rensac(correspondences):
  
  total_points = len(correspondences)
  s = get_num_poits()
  print ("Executing adaptative_number_samples...")
  #N = adaptative_number_samples()
  N = 99999999
  sample_count = 0
  print ("Number of samples:",N)
  print ("starting RENSAC...")
  best = 0
  e = 1
  while N > sample_count:
    print ("Executing ",sample_count+1,"iteration...")
    print ( "Calculando matriz...")
    f = calc_matrix(correspondences)
    p,p_ = calc_ps(f)
    print ("Calculada")
    print ("calculando inliear")
    print ("s",s)
    print ("e",e)
    inliers = get_inliers(f,correspondences,p,p_)
    e_ = 1 - (inliers*1.0)/total_points
    if(e_ < e):
      e = e_
      N = get_n_samples(s,e,0.99)
    print ("inliers:",inliers)
    print ("e",e)
    print ("N",N)
    
    print ("calculados")
    if(inliers > best):
      print ("best number of inliers:",inliers)
      F = f
      P,P_ = p,p_
      best = inliers
    print (sample_count+1,"executed.")
    print ("-")
    sample_count += 1
  print ("Final best:",best)
  return F,P,P_



def compareImages():
  print ("Imagens:",path_images)
  compare(path_images)
  return

def generateImage():
  for correspondences in putative_correspondences:
    F,P,P_ = rensac(correspondences)
  #print ("gerando imagem...")
  #new_image = generateNewImage(hs,path_images)
  #print ("imagem gerada.")
  #loadImage(new_image,'Panorama')
  print (F)
  print (P)
  print (P_)
  return 

window.title("Image transformation")
window.geometry("500x500")
center(window)


for n,f in enumerate(sys.argv[1:]):
  path_images[n] = f

compareImages()


# C = tk.Button(window, text ="Exec sift", command = compareImages)
# C.grid(row=1)

tk.Label(window, text="Threshold").grid(row=4, column=0)
tk.Label(window, text="Points").grid(row=5, column=0)

e2 = tk.Entry(window)
e2.insert(tk.END, '30')
e3 = tk.Entry(window)
e3.insert(tk.END, '8')
e2.grid(row=4, column=1)
e3.grid(row=5, column=1)

C = tk.Button(window, text ="Rensac", command = generateImage)
C.grid(row=6)

generateImage()



window.mainloop()