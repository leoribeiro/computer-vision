#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Tkinter as tk
from PIL import Image, ImageTk
import os, tkFileDialog
import tkMessageBox
import matplotlib.widgets as widgets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Circle
import math
import copy
import cv2


coords = {}
images = {}
images_ = {}
cont_images = 0
q_points = 4


def center(toplevel):
    toplevel.update_idletasks()
    w = toplevel.winfo_screenwidth()
    h = toplevel.winfo_screenheight()
    size = tuple(int(_) for _ in toplevel.geometry().split('+')[0].split('x'))
    x = w/2 - size[0]/2
    y = h/2 - size[1]/2
    toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))

def onclick_image(event,cont_images):
  print "cont_images",cont_images
  x = event.xdata
  y = event.ydata
  global coords
  global images
  fig = images[cont_images]
  c = coords[cont_images]
  onclick(fig,c,x,y)

def onclick(fig,coords,x,y):
   global q_points
   if x != None and y != None:
       circle = plt.Circle((x, y), 3, color='r')
       
       fig.add_subplot(111).add_artist(circle)
       
       
       if(len(coords) == q_points):
          coords = []
       coords.append((x,y))

       print "coords",coords

       fig.canvas.draw()
       


def openImage():
   filename_image = tkFileDialog.askopenfilename()
   global cont_images
   global images
   global images_
   global coords
   fig = plt.figure()
   images[cont_images] = fig
   coords[cont_images] = []
   print "Imagem carregada: ",filename_image
   global image_text
   image_text.set('Imagem carregada: \n' + str(filename_image))
   ax = fig.add_subplot(111)
   image = Image.open(filename_image)
   images_[cont_images] = image
   arr = np.asarray(image)
   plt_image=plt.imshow(arr)
   fig.canvas.set_window_title('Image')
   print "cont_images_openimage",cont_images
   num_image = copy.copy(cont_images)
   fig.canvas.mpl_connect('button_press_event', lambda event: onclick_image(event, num_image))

   plt.ion()
   plt.show()
   
   cont_images += 1
   return

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



def norm_x(x):
  if(x[2] == 0):
    return x
  return [x[0]/x[2],x[1]/x[2],x[2]/x[2]]


def generateNewImage(h,h_inv):

  global images_
  
  #pixels = image.load() # create the pixel map
  width0, height0 = images_[0].size
  width1, height1 = images_[1].size

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

  print min_x,max_x
  print min_y,max_y

  ratio = (max_x - min_x, max_y - min_y)

  n_width = width
  n_height = int(n_width * (ratio[1] / ratio[0]))

  new_image = Image.new('RGB', (n_width, n_height))

  step_y = (max_y - min_y)/n_height
  step_x = (max_x - min_x)/n_width
  x_cm = min_x
  print "width",width
  print "height",height
  for x in range(n_width):
    y_cm = min_y
    for y in range(n_height):
      coords = np.dot(h,[x_cm,y_cm,1])
      coords = norm_x(coords)
      try:
        new_pixel = images_[0].getpixel((x_cm,y_cm))
        new_image.putpixel((x,y),new_pixel)
      except IndexError:
        pass
      try:
        new_pixel = images_[1].getpixel((coords[0],coords[1]))
        new_image.putpixel((x,y),new_pixel)
      except IndexError:
        pass
      y_cm += step_y
    x_cm += step_x

  return new_image

def applyMatrix(h,h_inv):

  print "gerando imagem..."
  new_image = generateNewImage(h,h_inv)
  print "imagem gerada."
  return new_image
  
def matches_images(img1,img2):

  # Initiate SIFT detector
  sift = cv2.xfeatures2d.SIFT_create()

  # find the keypoints and descriptors with SIFT
  kp1, des1 = sift.detectAndCompute(img1,None)
  kp2, des2 = sift.detectAndCompute(img2,None)

  # BFMatcher with default params
  bf = cv2.BFMatcher()
  matches = bf.knnMatch(des1,des2, k=2)

  print "matches",matches
  return matches  

def returnMatrix(x,x_):
  print "x",x
  print "x_",x_
  line1 = [0,0,0,-x_[2]*x[0],-x_[2]*x[1],-x_[2]*x[2],x_[1]*x[0],x_[1]*x[1],x_[1]*x[2]]
  line2 = [x_[2]*x[0],x_[2]*x[1],x_[2]*x[2],0,0,0,-x_[0]*x[0],-x_[0]*x[1],-x_[0]*x[2]]
  #line3 = [-x_[1]*x[0],-x_[1]*x[1],-x_[1]*x[2],x_[0]*x[0],x_[0]*x[1],x_[0]*x[2],0,0,0]
  return line1,line2


def calc_matrix():
  a = [] 
  for i in range(0,q_points):
    A = returnMatrix([coords[0][i][0],coords[0][i][1],1],[coords[1][i][0],coords[1][i][1],1])
    a.append(A[0])
    a.append(A[1])
  a = np.array(a)

  U, s, V = np.linalg.svd(a, full_matrices=True)
  print "U",U
  print "s",s
  print "V",V

  #c = V[:, -1]
  c = V[-1]
  print "c",c
  h = np.array([[c[0],c[1],c[2]],[c[3],c[4],c[5]],[c[6],c[7],c[8]]])
  print "h",h
  #h = []
  return h

def generateImage():
  global images_

  matches_images(images_[0],images_[1])

  #h = calc_matrix()
  #h_inv = np.linalg.inv(h)
  #new_image =  applyMatrix(h,h_inv)
  #loadImage(new_image,'Panorama')
  return 


window = tk.Tk()
window.title("Image transformation")
window.geometry("500x500")
center(window)

B = tk.Button(window, text ="Open image", command = openImage)
B.grid(row=0)

coords_text = tk.StringVar()
coords_text.set('')
image_text = tk.StringVar()
image_text.set('')
l1 = tk.Label(window, textvariable = image_text,fg="black")
l1.grid(row=1,columnspan=2)
l2 = tk.Label(window, textvariable = coords_text,fg="black")
l2.grid(row=2,columnspan=2)


C = tk.Button(window, text ="panoramic", command = generateImage)
C.grid(row=6)


window.mainloop()