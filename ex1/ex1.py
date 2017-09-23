import Tkinter as tk
from PIL import Image, ImageTk
import os, tkFileDialog
import tkMessageBox
import matplotlib.widgets as widgets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Circle

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
   imagem_points = coords
   real_points = [(0,61.3),(81.9,61.3),(0,0),(81.9,0)]
   equations = [
    [],
    [],
    [],
    [],

    [],
    [],
    [],
    [],
   ]
   print coords
   print equations

def applyMatrix():
  global image
  pixels = image.load() # create the pixel map



def transformImage():
   solveSystem()
   applyMatrix()
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