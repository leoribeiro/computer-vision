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

kp1 = []
kp2 = []
matches = []
path_images = {}
coords = {}
images = {}
images_ = {}
cont_images = 0
q_points = 10


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
  return [x[0]/x[2],x[1]/x[2],x[2]/x[2]]

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

def generateNewImage(h,h_inv):

  img1 = Image.open(path_images[1])
  img2 = Image.open(path_images[2])
  
  #pixels = image.load() # create the pixel map
  width0, height0 = img1.size
  width1, height1 = img2.size

  height = height0
  width = width0 + width1

  xs = []
  ys = []

  x = np.dot(h_inv,[0,0,1])
  x = norm_x(x)
  xs.append(x[0])
  ys.append(x[1])


  x = np.dot(h_inv,[0,height1 - 1,1])
  x = norm_x(x)
  xs.append(x[0])
  ys.append(x[1])

  x = np.dot(h_inv,[width1 - 1,height1 - 1,1])
  x = norm_x(x)
  xs.append(x[0])
  ys.append(x[1])

  x = np.dot(h_inv,[width1 - 1,0,1])
  x = norm_x(x)
  xs.append(x[0])
  ys.append(x[1])

  xs.append(0)
  xs.append(width0 - 1)
  ys.append(0)
  ys.append(height0 - 1)

  min_x = min(xs)
  min_y = min(ys)
  max_x = max(xs)
  max_y = max(ys)

  print (min_x,max_x)
  print (min_y,max_y)

  ratio = (max_x - min_x, max_y - min_y)

  n_width = width
  n_height = int(n_width * (ratio[1] / ratio[0]))

  new_image = Image.new('RGB', (n_width, n_height))

  step_y = (max_y - min_y)/n_height
  step_x = (max_x - min_x)/n_width
  x_cm = min_x
  print ("width",width)
  print ("height",height)
  for x in range(n_width):
    y_cm = min_y
    for y in range(n_height):
      coords = np.dot(h,[x_cm,y_cm,1])
      coords = norm_x(coords)
      try:
        new_pixel = img1.getpixel((x_cm,y_cm))
        new_image.putpixel((x,y),new_pixel)
      except IndexError:
        pass
      try:
        new_pixel = img2.getpixel((coords[0],coords[1]))
        new_image.putpixel((x,y),new_pixel)
      except IndexError:
        pass
      y_cm += step_y
    x_cm += step_x

  return new_image

def applyMatrix(h,h_inv):

  print ("gerando imagem...")
  new_image = generateNewImage(h,h_inv)
  print ("imagem gerada.")
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

def compare(filename1, filename2):
    global matches
    global kp1,kp2

    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.match(des1,des2)

    matches = sorted(matches, key=lambda val: val.distance)

    img3 = drawMatches(img1,kp1,img2,kp2,matches[:25])
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(num=None, figsize=(16, 6), dpi=80, facecolor='w', edgecolor='k')
    fig.subplots_adjust(bottom = 0)
    fig.subplots_adjust(top = 1)
    fig.subplots_adjust(right = 1)
    fig.subplots_adjust(left = 0)
    plt.imshow(img3) 
    plt.ion()
    plt.show()

    return

def returnMatrix(x,x_):
  print ("x",x)
  print ("x_",x_)
  line1 = [0,0,0,-x_[2]*x[0],-x_[2]*x[1],-x_[2]*x[2],x_[1]*x[0],x_[1]*x[1],x_[1]*x[2]]
  line2 = [x_[2]*x[0],x_[2]*x[1],x_[2]*x[2],0,0,0,-x_[0]*x[0],-x_[0]*x[1],-x_[0]*x[2]]
  #line3 = [-x_[1]*x[0],-x_[1]*x[1],-x_[1]*x[2],x_[0]*x[0],x_[0]*x[1],x_[0]*x[2],0,0,0]
  return line1,line2

def get_poits():
  print ("kp1",kp1)
  global coords
  coords[0] = []
  coords[1] = []
  cont = 0
  for m in matches:
        img1_idx = m.queryIdx
        img2_idx = m.trainIdx
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt
        coords[0].append((x1,y1))
        coords[1].append((x2,y2))
        cont += 1
        if(cont == q_points):
          break

def calc_matrix():
  get_poits()
  a = [] 
  for i in range(0,q_points):
    A = returnMatrix([coords[0][i][0],coords[0][i][1],1],[coords[1][i][0],coords[1][i][1],1])
    a.append(A[0])
    a.append(A[1])
  a = np.array(a)

  U, s, V = np.linalg.svd(a, full_matrices=True)
  print ("U",U)
  print ("s",s)
  print ("V",V)

  #c = V[:, -1]
  c = V[-1]
  print ("c",c)
  h = np.array([[c[0],c[1],c[2]],[c[3],c[4],c[5]],[c[6],c[7],c[8]]])
  print ("h",h)
  #h = []
  return h

def compareImages():
  print (path_images)
  compare(path_images[1],path_images[2])
  return

def generateImage():
  h = calc_matrix()
  h_inv = np.linalg.inv(h)
  new_image = applyMatrix(h,h_inv)
  loadImage(new_image,'Panorama')
  return 

window.title("Image transformation")
window.geometry("500x500")
center(window)

img1 = sys.argv[1]
img2 = sys.argv[2]
path_images[1] = img1
path_images[2] = img2

C = tk.Button(window, text ="Exec sift", command = compareImages)
C.grid(row=4)


C = tk.Button(window, text ="panoramic", command = generateImage)
C.grid(row=6)


window.mainloop()