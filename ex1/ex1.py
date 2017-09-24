import Tkinter as tk
from PIL import Image, ImageTk
import os, tkFileDialog
import tkMessageBox
import matplotlib.widgets as widgets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Circle

# pontos no mundo real
real_points = [(0,61.3),(81.9,61.3),(0,0),(81.9,0)]

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
   x = event.x
   y = event.y
   if x != None and y != None:
       #print(x,y)
       global coords
       coords = [(x,y)] + coords
       coords = coords[:4]
       print coords
       global circles
       circle = plt.Circle((event.xdata, event.ydata), 3, color='r')
       global fig
       fig.add_subplot(111).add_artist(circle)
       circles = [circle] + circles
       if(len(circles) > 4):
          circles[-1].remove()
       circles = circles[:4]
       fig.canvas.draw()
       global coords_text
       coords_text.set('Coordenadas Escolhidas: \n' + str(coords).strip('[]'))


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
   fig.canvas.mpl_connect('button_press_event', onclick)
   plt.ion()
   plt.show()
   print "ola"
   return


def solveSystem():
   global coords
   global real_points
   imagem_points = coords

   x0_ = imagem_points[0][0]
   y0_ = imagem_points[0][1]
   x1_ = imagem_points[1][0]
   y1_ = imagem_points[1][1]
   x2_ = imagem_points[2][0]
   y2_ = imagem_points[2][1]
   x3_ = imagem_points[3][0]
   y3_ = imagem_points[3][1]

   b = np.array([x0_, y0_, x1_, y1_, x2_, y2_, x3_, y3_])

   x0 = real_points[0][0]
   y0 = real_points[0][1]
   x1 = real_points[1][0]
   y1 = real_points[1][1]
   x2 = real_points[2][0]
   y2 = real_points[2][1]
   x3 = real_points[3][0]
   y3 = real_points[3][1]

   equations = np.array([
    [x0, y0, 1, 0, 0, 0, -x0*x0_, -y0*y0_],
    [0, 0, 0, x0, y0, 1, -x0*y0_, -y0*y0_],
    [x1, y1, 1, 0, 0, 0, -x1*x1_, -y1*y1_],
    [0, 0, 0, x1, y1, 1, -x1*y1_, -y1*y1_],

    [x2, y2, 1, 0, 0, 0, -x2*x2_, -y2*y2_],
    [0, 0, 0, x2, y2, 1, -x2*y2_, -y2*y2_],
    [x3, y3, 1, 0, 0, 0, -x3*x3_, -y3*y3_],
    [0, 0, 0, x3, y3, 1, -x3*y3_, -y3*y3_],
   ])
   x = np.linalg.solve(equations,b)

   h = [[x[0], x[1], x[2]], [x[3], x[4], x[5]], [x[6], x[7], 1]]
   h_inv = np.linalg.inv(h)

   applyMatrix(h_inv)


   #print coords
   #print equations
   #print x


def norm_x(x):
  return [x[0]/x[2],x[1]/x[2],x[2]/x[2]]

def applyMatrix(h):
  global image
  pixels = image.load() # create the pixel map
  width, height = image.size

  print width,height

  new_positions = []


  for i in range(height):
      for j in range(width):
        x = np.dot(h,[i,j,1])
        x = norm_x(x)
        new_positions.append(x)

  x = np.dot(h,[0,0,1])
  print "h * (0,0) -> ",x,norm_x(x)

  x = np.dot(h,[0,height - 1,1])
  print "h * (0,599) -> ",x,norm_x(x)

  x = np.dot(h,[width - 1,height - 1,1])
  print "h * (799,599) -> ",x,norm_x(x)

  x = np.dot(h,[width - 1,0,1])
  print "h * (799,0) -> ",x,norm_x(x)
  
  #step = (max_x-min_x)/width



def transformImage():
   solveSystem()
   return


window = tk.Tk()
window.title("Image transformation")
window.geometry("500x500")
center(window)

B = tk.Button(window, text ="Abrir imagem", command = openImage)
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

C = tk.Button(window, text ="Ajustar", command = transformImage)
C.pack()


window.mainloop()