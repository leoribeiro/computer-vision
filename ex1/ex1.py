import Tkinter as tk
from PIL import Image, ImageTk
import os, tkFileDialog
import tkMessageBox
import matplotlib.widgets as widgets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Circle

# pontos no mundo real - brahma
#x_r = 81.9
#y_r = 61.3

# legiao
x_r = 32
y_r = 45
real_points = [[0,0],[0,y_r],[x_r,y_r],[x_r,0]]

coords = []
circles = []

# def order_coords(coords):
#   for c1 in coords:
#     for c2 in coords:
#       if(c1[0] < c2[0] and c1[1] < c2[1]):
#         s0 = c1
#       if(c1[0] > c2[0] and c1[1] > c2[1]):
#         s3 = c1


def center(toplevel):
    toplevel.update_idletasks()
    w = toplevel.winfo_screenwidth()
    h = toplevel.winfo_screenheight()
    size = tuple(int(_) for _ in toplevel.geometry().split('+')[0].split('x'))
    x = w/2 - size[0]/2
    y = h/2 - size[1]/2
    toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))

def onclick(event):
   x = event.xdata
   y = event.ydata
   if x != None and y != None:
       global coords
       global image
       width, height = image.size
       coords = [(x,y)] + coords
       coords = coords[:4]
       #print coords
       global circles
       circle = plt.Circle((event.xdata, event.ydata), 3, color='r')
       global fig
       fig.add_subplot(111).add_artist(circle)
       circles = [circle] + circles
       if(len(circles) > 4):
          circles[-1].remove()
       circles = circles[:4]
       fig.canvas.draw()
       
       if(len(coords) > 3):
         global coords_text
         coords_text.set('Coordenadas Escolhidas: \n' + returnCoordsText())


def openImage():
   #tkMessageBox.showinfo( "Hello Python", "Hello World")
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
   fig.canvas.mpl_connect('button_press_event', onclick)

   plt.ion()
   plt.show()
   return

def loadImage(image):
   global fig
   fig = plt.figure()

   ax = fig.add_subplot(111)
   arr = np.asarray(image)
   plt.imshow(arr)
   plt.ion()
   plt.show()
   fig.canvas.set_window_title('Similarity Space')
   return

def returnCoordsText():
   coords_order = returnCoords()
   text =  "X0 : ("+"{0:.2f}".format(coords_order[0])+" , "+"{0:.2f}".format(coords_order[1])+") \n X1 : ("+"{0:.2f}".format(coords_order[2])+" , "+"{0:.2f}".format(coords_order[3])+") "
   text += "\n X2 : ("+"{0:.2f}".format(coords_order[4])+" , "+"{0:.2f}".format(coords_order[5])+") \n X3 : ("+"{0:.2f}".format(coords_order[6])+" , "+"{0:.2f}".format(coords_order[7])+") "

   return text

def returnCoords():
   global coords
   imagem_points = sorted(coords, key=lambda tup: tup[0]) 
   if(imagem_points[2][1] > imagem_points[3][1]):
      x2 = imagem_points[2][0]
      y2 = imagem_points[2][1]
      x3 = imagem_points[3][0]
      y3 = imagem_points[3][1]
   else:
      x3 = imagem_points[2][0]
      y3 = imagem_points[2][1]
      x2 = imagem_points[3][0]
      y2 = imagem_points[3][1]   
   if(imagem_points[0][1] > imagem_points[1][1]):
      x1 = imagem_points[0][0]
      y1 = imagem_points[0][1]
      x0 = imagem_points[1][0]
      y0 = imagem_points[1][1]
   else:
      x0 = imagem_points[0][0]
      y0 = imagem_points[0][1]
      x1 = imagem_points[1][0]
      y1 = imagem_points[1][1]

   return x0,y0,x1,y1,x2,y2,x3,y3



def solveSystem():
   global coords
   global real_points
   

   x0_ = real_points[0][0]
   y0_ = real_points[0][1]
   x1_ = real_points[1][0]
   y1_ = real_points[1][1]
   x2_ = real_points[2][0]
   y2_ = real_points[2][1]
   x3_ = real_points[3][0]
   y3_ = real_points[3][1]

   print "X0: ("+str(x0_)+" , "+str(y0_)+") "
   print "X1: ("+str(x1_)+" , "+str(y1_)+") "
   print "X2: ("+str(x2_)+" , "+str(y2_)+") "
   print "X3: ("+str(x3_)+" , "+str(y3_)+") "

   x0,y0,x1,y1,x2,y2,x3,y3 = returnCoords()

   b = np.array([x0_, y0_, x1_, y1_, x2_, y2_, x3_, y3_])


   print real_points

   equations = np.array([
    [x0, y0, 1, 0, 0, 0, -x0*x0_, -y0*x0_],
    [0, 0, 0, x0, y0, 1, -x0*y0_, -y0*y0_],
    [x1, y1, 1, 0, 0, 0, -x1*x1_, -y1*x1_],
    [0, 0, 0, x1, y1, 1, -x1*y1_, -y1*y1_],
    [x2, y2, 1, 0, 0, 0, -x2*x2_, -y2*x2_],
    [0, 0, 0, x2, y2, 1, -x2*y2_, -y2*y2_],
    [x3, y3, 1, 0, 0, 0, -x3*x3_, -y3*x3_],
    [0, 0, 0, x3, y3, 1, -x3*y3_, -y3*y3_],
   ])

   x = np.linalg.solve(equations,b)

   h = np.array([[x[0], x[1], x[2]], [x[3], x[4], x[5]], [x[6], x[7], 1]])
   print "h",h
   h_inv = np.linalg.inv(h)
   print "h_inv",h_inv
   print "x:",x

   applyMatrix(h,h_inv)


   #print coords
   #print equations
   


def norm_x(x):
  return [x[0]/x[2],x[1]/x[2],x[2]/x[2]]

def applyMatrix(h,h_inv):
  global image
  #pixels = image.load() # create the pixel map
  width, height = image.size

  #print width,height

  new_positions = []
  xs = []
  ys = []

  # for i in range(width):
  #    for j in range(height):
  #      x = np.dot(h,[i,j,1])
  #      x = norm_x(x)
  #      new_positions.append(x)
  #      xs.append(x[0])
  #      ys.append(x[1])



  x = np.dot(h,[0,0,1])
  print "h * (0,0) -> ",x,norm_x(x)
  x = norm_x(x)
  xs.append(x[0])
  ys.append(x[1])


  x = np.dot(h,[0,height - 1,1])
  print "h * (0,599) -> ",x,norm_x(x)
  x = norm_x(x)
  xs.append(x[0])
  ys.append(x[1])

  x = np.dot(h,[width - 1,height - 1,1])
  print "h * (799,599) -> ",x,norm_x(x)
  x = norm_x(x)
  xs.append(x[0])
  ys.append(x[1])

  x = np.dot(h,[width - 1,0,1])
  print "h * (799,0) -> ",x,norm_x(x)
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
      #print coords
      coords = norm_x(coords)
      #print coords
      try:
        new_pixel = image.getpixel((coords[0],coords[1]))
        new_image.putpixel((x,y),new_pixel)
      except IndexError:
        pass
      y_cm += step_y
    x_cm += step_x

  loadImage(new_image)






def transformImage():
   solveSystem()
   return


window = tk.Tk()
window.title("Image transformation")
window.geometry("500x500")
center(window)

B = tk.Button(window, text ="Open image", command = openImage)
B.pack()

#T = tk.Text(window, height=5, width=400)

#T.insert(tk.END, "Coordenadas:\n")
#T.pack()

coords_text = tk.StringVar()
coords_text.set('')
image_text = tk.StringVar()
image_text.set('')
l1 = tk.Label(window, textvariable = image_text,fg="black")
l1.pack()
l2 = tk.Label(window, textvariable = coords_text,fg="black")
l2.pack()

C = tk.Button(window, text ="Transform!", command = transformImage)
C.pack()


window.mainloop()