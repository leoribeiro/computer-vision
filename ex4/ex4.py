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

def convert_points(h,points):
  xs = []
  for p in points:
    x = np.dot(h,p)
    x = norm_x(x)
    xs.append(x)
  return xs

def generateNewImage(h,image):

  h_inv = np.linalg.inv(h)

  width = image.size[0]
  height = image.size[1]

  xs = []
  ys = []

  x = np.dot(h,[0,0,1])
  x = norm_x(x)
  xs.append(x[0])
  ys.append(x[1])


  x = np.dot(h,[0,height - 1,1])
  x = norm_x(x)
  xs.append(x[0])
  ys.append(x[1])

  x = np.dot(h,[width - 1,height - 1,1])
  x = norm_x(x)
  xs.append(x[0])
  ys.append(x[1])

  x = np.dot(h,[width - 1,0,1])
  x = norm_x(x)
  xs.append(x[0])
  ys.append(x[1])

  min_x = min(xs)
  min_y = min(ys)
  max_x = max(xs)
  max_y = max(ys)

  print (min_x,max_x)
  print (min_y,max_y)

  #print (min_x,max_x)
  #print (min_y,max_y)

  ratio = (max_x - min_x, max_y - min_y)

  n_width = width
  #n_height = int(n_width * (ratio[1] / ratio[0]))
  n_height = height
  print ("n_width",n_width)
  print ("n_height",n_height)

  new_image = Image.new('RGB', (n_width, n_height))

  step_y = (max_y - min_y)/n_height
  step_x = (max_x - min_x)/n_width
  x_cm = min_x
  #print ("width",width)
  #print ("height",height)
  for x in range(n_width):
    y_cm = min_y
    for y in range(n_height):
      coords = np.dot(h_inv,[x_cm,y_cm,1])
      coords = norm_x(coords)
      try:
        new_pixel = image.getpixel((coords[0],coords[1]))
        new_image.putpixel((x,y),new_pixel)
      except IndexError:
        pass
      y_cm += step_y
    x_cm += step_x

  return new_image


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

  e = norm_x(e)
  e_ = norm_x(e_)



  P = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
  # pag 581
  P_ = np.dot([[0,-e_[2],e_[1]],[e_[2],0,-e_[0]],[-e_[1],e_[0],1]],F)
  P_ = [[P_[0][0],P_[0][1],P_[0][2],e_[0]],
  [P_[1][0],P_[1][1],P_[1][2],e_[1]],
  [P_[2][0],P_[2][1],P_[2][2],e_[2]]]

  P = np.array(P)
  P_ = np.array(P_)

  return P,P_,e_

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
  F = np.multiply(1/F[2][2],F)
  
  return F

def symmetric_transfer_error(x,x_,h,h_inv):
  x = np.array(x)
  x_ = np.array(x_)
  d1 = np.linalg.norm(x-norm_x(np.dot(h_inv,x_)))
  d2 = np.linalg.norm(x_-norm_x(np.dot(h,x)))
  return d1 + d2

def triangulation(x,x_,P,P_):

  A = [x[0]*P[2] - P[0],
       x[1]*P[2] - P[1],
       x_[0]*P_[2] - P_[0],
       x_[1]*P_[2] - P_[1]]
  A = np.array(A)
  #print ("A",A)
  U, s, V = np.linalg.svd(A, full_matrices=True)

  c = V[-1]
  X = np.array([c[0]/c[3],c[1]/c[3],c[2]/c[3],1])

  #print ("X",X)
  #print ("P",P)

  x_v = np.dot(P,X)
  x_v_ = np.dot(P_,X)

  #print ("x",x)
  #print ("x_",x_)
  #print ("x_v",x_v)
  #print ("x_v_",x_v_)
  return x_v,x_v_

def get_3dpoint(x,x_,P,P_):
  A = [x[0]*P[2] - P[0],
       x[1]*P[2] - P[1],
       x_[0]*P_[2] - P_[0],
       x_[1]*P_[2] - P_[1]]
  A = np.array(A)
  U, s, V = np.linalg.svd(A, full_matrices=True)

  c = V[-1]
  X = np.array([c[0]/c[3],c[1]/c[3],c[2]/c[3]])

  return X

def ImageRectification(coords,e_,P,P_,F):
  
  img_ = Image.open(path_images[0])
  width = img_.size[0]
  height = img_.size[1]
  centerx = int(width/2)
  centery = int(height/2)

  T = [[1,0,-centerx],[0,1,-centery],[0,0,1]]
  
  # ep: [epx epy 1]' is mapped (by G) into [epx-x0 epy-y0 1]'
  # Set rotation matrix R = [cos(a) -sin(a) 0; sin(a) cos(a) 0; 0 0 1]
  # s.t. R[epx-x0, epy-y0, 1]'=[f 0 1]'
  # cos(a)*(epx-x0)-sin(a)*(epy-y0)=f
  # sin(a)*(epx-x0)+cos(a)*(epy-y0)=0

  alpha = np.arctan(-(e_[1]/e_[2] - centery)/(e_[0]/e_[2]-centerx))
  f = np.cos(alpha)*(e_[0]/e_[2]-centerx)-np.sin(alpha)*(e_[1]/e_[2]-centery)

  R = [[np.cos(alpha),-np.sin(alpha),0],[np.sin(alpha),np.cos(alpha),0],[0,0,1]]

  # Set G = [1 0 0; 0 1 0; -1/f 0 1]
  G = [[1,0,0],[0,1,0],[-1/f,0,1]]

  # H' = GRT
  # H' will send e' to [f 0 0]' 
  H_ = np.dot(np.dot(G,R),T)


  x_new = np.dot(H_,[centerx,centery,1])
  x_new = norm_x(x_new)

  # T2 = [1 0 centerx-xpnew.x; 0 1 centery-xpnew.y; 0 0 1]; 
  t2 = [[1,0,x_new[0]],[0,1,x_new[1]],[0,0,1]]

  H_ = np.dot(t2,H_)
  H_ = np.multiply(1/H_[2][2],H_)
  print ("H_",H_)

  # M = P'P+ 
  print ("P",P)
  print ("P_",P_)
  M = np.dot(P_,np.linalg.pinv(P))
  # H0 = H'M 
  H0 = np.dot(H_,M)

  A = [] 
  B = []
  # for Ha solve linear equations
  #coords = get_poits_random(3,coords)
  for c in coords:
    x = [c[0][0],c[0][1],1]
    x_ = [c[1][0],c[1][1],1]
    # transform x: H0x 
    x_n = np.dot(H0,x)
    x_n = norm_x(x_n)
    # transform x': H'x' 
    x_n_ = np.dot(H_,x_)
    x_n_ = norm_x(x_n_)
    a = x_n
    A.append(a)
    B.append(x_n_[0])
  
  A = np.array(A)
  B = np.array(B)
  #t = np.linalg.solve(A,B)
  result,resids,rank,s = np.linalg.lstsq(A,B)
  t = result

  print ("t",t)

  # Set Ha = [a b c; 0 1 0; 0 0 1]
  Ha = np.array([t,[0,1,0],[0,0,1]])
  print ("Ha",Ha) 

  # H = HaH0
  H = np.dot(Ha,H0)
  H = np.multiply(1/H[2][2],H)
  print ("H",H)

  # update F = H'^{-t} F H^{-1}
  H_inv_ = np.linalg.inv(H_)
  H_inv = np.linalg.inv(H)

  F = np.dot(np.transpose(H_inv_),F)
  F = np.dot(F,H_inv)
  F = np.multiply(1/F[2][2],F)
  print ("F",F)

  H_inv_ = np.multiply(1/H_inv_[2][2],H_inv_)
  H_inv = np.multiply(1/H_inv[2][2],H_inv)

  return H,H_

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


def get_inliers(correspondences,p,p_):
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
    p,p_,e__ = calc_ps(f)
    print ("Calculada")
    print ("calculando inliear")
    print ("s",s)
    print ("e",e)
    inliers = get_inliers(correspondences,p,p_)
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
      P,P_,E_ = p,p_,e__
      best = inliers
    print (sample_count+1,"executed.")
    print ("-")
    sample_count += 1
  print ("Final best:",best)
  return F,P,P_,E_

def reconstruct3DPt(correspondences,P,P_):
  new_correspondences = []
  for c in correspondences:
    x = [c[0][0],c[0][1],1]
    x_ = [c[1][0],c[1][1],1]
    X = get_3dpoint(x,x_,P,P_)
    new_correspondences.append([x,x_,X])
  return new_correspondences

def compareImages():
  print ("Imagens:",path_images)
  compare(path_images)
  return

def generateImage():
  correspondences =  putative_correspondences[0]
  F,P,P_,e_ = rensac(correspondences)
  n_coords = reconstruct3DPt(correspondences,P,P_)
  H,H_ = ImageRectification(correspondences,e_,P,P_,F)

  img1 = Image.open(path_images[0])
  img2 = Image.open(path_images[1])
  print ("gerando imagem1 retificada...")
  img1_retified = generateNewImage(H,img1)
  print ("imagem gerada.")
  print ("gerando imagem2 retificada...")
  img2_retified = generateNewImage(H_,img2)
  print ("imagem gerada.")
  #print ("gerando imagem...")
  #new_image = generateNewImage(hs,path_images)
  #print ("imagem gerada.")
  loadImage(img1_retified,'img1_retified')
  loadImage(img2_retified,'img2_retified')
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
e2.insert(tk.END, '25')
e3 = tk.Entry(window)
e3.insert(tk.END, '8')
e2.grid(row=4, column=1)
e3.grid(row=5, column=1)

C = tk.Button(window, text ="Rensac", command = generateImage)
C.grid(row=6)

generateImage()



window.mainloop()