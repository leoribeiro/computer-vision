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


coords_p = []
circles_p = []
lines_p = []
q_points_p = 20


def center(toplevel):
    toplevel.update_idletasks()
    w = toplevel.winfo_screenwidth()
    h = toplevel.winfo_screenheight()
    size = tuple(int(_) for _ in toplevel.geometry().split('+')[0].split('x'))
    x = w/2 - size[0]/2
    y = h/2 - size[1]/2
    toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))

def clearPoints(coords,lines,circles):
  for l in circles:
    l.remove()
  for l in lines:
    l.remove()
  del coords[:]
  del lines[:]
  del circles[:]


def onclick_p(event):
  x = event.xdata
  y = event.ydata
  global coords_p
  global circles_p
  global lines_p
  global q_points_p
  global fig
  onclick(fig,coords_p,circles_p,lines_p,q_points_p,x,y)

def onclick(fig,coords,circles,lines,q_points,x,y):
   if x != None and y != None:
       circle = plt.Circle((x, y), 3, color='r')
       
       fig.add_subplot(111).add_artist(circle)
       
       if(len(circles) == q_points):
          clearPoints(coords,lines,circles)
       circles.append(circle)
       coords.append((x,y))

       if(len(coords) > 1):
        for l in lines:
           l.remove()
        del lines[:]
        index = 0
        while True:
          if((index + 1) >= len(coords)):
            break
          x_ = [coords[index][0],coords[index + 1][0]]
          y_ = [coords[index][1],coords[index + 1][1]]
          line = plt.Line2D(x_,y_,color = 'r', lw=3)
          fig.add_subplot(111).add_artist(line)
          lines.append(line)          
          index += 2

       fig.canvas.draw()
       


def openImage():
   filename_image = tkFileDialog.askopenfilename()
   global fig
   fig = plt.figure()
   print "Imagem carregada: ",filename_image
   global image_text
   image_text.set('Imagem carregada: \n' + str(filename_image))
   ax = fig.add_subplot(111)
   global image
   image = Image.open(filename_image)
   arr = np.asarray(image)
   plt_image=plt.imshow(arr)
   fig.canvas.set_window_title('Projective Space')
   fig.canvas.mpl_connect('button_press_event', onclick_p)

   plt.ion()
   plt.show()
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

def lerp(a, b, coord):
    if isinstance(a, tuple):
        return tuple([lerp(c, d, coord) for c,d in zip(a,b)])
    ratio = coord - math.floor(coord)
    return int(round(a * (1.0-ratio) + b * ratio))

def bilinear(im, x, y):
    x1, y1 = int(math.floor(x)), int(math.floor(y))
    x2, y2 = x1+1, y1+1
    left = lerp(im.getpixel((x1, y1)), im.getpixel((x1, y2)), y)
    right = lerp(im.getpixel((x2, y1)), im.getpixel((x2, y2)), y)
    return lerp(left, right, x)

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
  

def returnTerms(l,m):
  term1 = l[0]*m[0]
  term2 = ((l[0]*m[1])+(l[1]*m[0]))/2
  term3 = l[1]*m[1]
  term4 = ((l[0]*m[2])+(l[2]*m[0]))/2
  term5 = ((l[1]*m[2])+(l[2]*m[1]))/2
  term6 = (l[2]*m[2])* -1
  return term1,term2,term3,term4,term5,term6


def alg_1(c):
  print "c",c
  U, s, V = np.linalg.svd(c, full_matrices=True)
  print "U",U
  print "s",s
  print "V",V

  d_1 = np.diag([np.sqrt(s[0]),np.sqrt(s[1]),10])
  d_2 = np.diag([s[0],s[1],s[2]/100])
  d = np.dot(d_1,d_2)
  d = np.dot(d,d_1)

  print "d",d

  h = np.dot(U,d)
  return h

def alg_2(c):
  s = np.array([[c[0][0],c[0][1]],[c[1][0],c[1][1]]])
  k = np.linalg.cholesky(s)
  print "k",k
  k_t = np.transpose(k)
  print "k.t",k_t

  a = np.array(s)
  b = np.array([c[0][2],c[1][2]])
  v = np.linalg.solve(a,b)
  h_a = np.array([[k_t[0][0],k_t[0][1],0],[k_t[1][0],k_t[1][1],0],[0,0,1]])
  h_p = np.array([[1,0,0],[0,1,0],[v[0],v[1],1]])
  h = np.dot(h_p,h_a)
  return h

def projective_to_similarity():
  global coords_p
  coords = coords_p
  print coords
  i = 0
  a = []
  b = []
  while (i < 20):
    print i
    l = np.cross([coords[i][0],coords[i][1],1],[coords[i+1][0],coords[i+1][1],1])
    m = np.cross([coords[i+2][0],coords[i+2][1],1],[coords[i+3][0],coords[i+3][1],1])
    terms = returnTerms(l,m)
    a.append(terms[:-1])
    b.append(terms[-1])
    i += 4

  a = np.array(a)
  b = np.array(b)
  x = np.linalg.solve(a,b)
  print "a",a
  print "b",b
  print "x",x

  c = np.array([[x[0],x[1]/2,x[3]/2],[x[1]/2,x[2],x[4]/2],[x[3]/2,x[4]/2,1]])

  h = alg_2(c)
  print "h",h
  return h



def transformSimilarity():
   global image
   h = projective_to_similarity()
   h_inv = np.linalg.inv(h)
   new_image =  applyMatrix(h_inv,h,image)
   loadImage(new_image,'Similarity Space')
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


C = tk.Button(window, text ="projective to similarity", command = transformSimilarity)
C.grid(row=6)


window.mainloop()