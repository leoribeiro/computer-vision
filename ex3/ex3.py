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


coords = {}
images = {}
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
  global q_points
  fig = images[cont_images]
  coords = coords[cont_images]
  onclick(fig,coords,x,y)

def onclick(fig,coords,x,y):
   if x != None and y != None:
       circle = plt.Circle((x, y), 3, color='r')
       
       fig.add_subplot(111).add_artist(circle)
       
       if(len(coords[cont_images]) == q_points):
          coords[cont_images] = []
       coords.append((x,y))

       fig.canvas.draw()
       


def openImage():
   filename_image = tkFileDialog.askopenfilename()
   global cont_images
   global images
   global coords
   fig = plt.figure()
   images[cont_images] = fig
   coords[cont_images] = []
   print "Imagem carregada: ",filename_image
   global image_text
   image_text.set('Imagem carregada: \n' + str(filename_image))
   ax = fig.add_subplot(111)
   global image
   image = Image.open(filename_image)
   arr = np.asarray(image)
   plt_image=plt.imshow(arr)
   fig.canvas.set_window_title('Projective Space')
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


def generateNewImage(h,h_inv,image,interp=False):
  
  #pixels = image.load() # create the pixel map
  width, height = image.size

  new_positions = []
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

  print min_x,max_x
  print min_y,max_y

  n_width = width
  n_height = height

  new_image = Image.new('RGB', (n_width, n_height))
  n_width, n_height = new_image.size
  
  step_y = (max_y - min_y)/n_height
  step_x = (max_x - min_x)/n_width

  x_cm = min_x
  for x in range(n_width):
    y_cm = min_y
    for y in range(n_height):
      coords = np.dot(h_inv,[x_cm,y_cm,1])
      coords = norm_x(coords)
      try:
        if not interp:
          new_pixel = image.getpixel((coords[0],coords[1]))
        else:
          new_pixel = bilinear(image,coords[0],coords[1])
        new_image.putpixel((x,y),new_pixel)
      except IndexError:
        try:
          new_pixel = image.getpixel((coords[0],coords[1]))
        except IndexError:
          pass
      y_cm += step_y
    x_cm += step_x

  return new_image

def applyMatrix(h,h_inv,image):

  print "gerando imagem..."
  new_image = generateNewImage(h,h_inv,image,False)
  print "imagem gerada."
  #print "gerando imagem interpolada..."
  #new_image_i = generateNewImage(h,h_inv,True)
  #print "imagem interpolada gerada."
  return new_image
  #loadImage(new_image_i, " - Interpolada")
  


def generateMatrix():
  pass





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


C = tk.Button(window, text ="projective to similarity", command = generateMatrix)
C.grid(row=6)


window.mainloop()