import tkinter as tk
from tkinter import ttk
from tkinter import font

from PIL import Image, ImageTk

import numpy as np

import sys
import os
import argparse

# Directory containing input images (defaults to the one that contains this script)
input_dir = os.path.dirname(os.path.realpath(__file__))

# Input values
input_image = ""
input_width = 0
input_height = 0

# Constants
draw_color = "red"
point_diameter = 10
point_center_tag_prefix = "center_"
point_circle_tag_prefix = "circle_"
point_text_tag_prefix = "text_"
line_tag_prefix = "line_"

class MyContent:
	current_point = 0
	points = [(0,0)]*8
	
	def __init__(self, root):
		# Label variables for showing point coordinates
		var_coord0 = tk.StringVar()
		var_coord1 = tk.StringVar()
		var_coord2 = tk.StringVar()
		var_coord3 = tk.StringVar()
		var_coord4 = tk.StringVar()
		var_coord5 = tk.StringVar()
		var_coord6 = tk.StringVar()
		var_coord7 = tk.StringVar()
		self.var_coords = [var_coord0, var_coord1, var_coord2, var_coord3, var_coord4, var_coord5, var_coord6, var_coord7]
		for lbl in self.var_coords:
			lbl.set((0, 0))
		
		# Create frame that will contain all elements
		content = ttk.Frame(root)
		content.grid(column=0, row=0)
		
		# Load input image
		image = Image.open(os.path.join(input_dir, input_image))
		global input_width
		global input_height
		input_width, input_height = image.size
		
		# Create canvas for showing input image and drawing points
		self.canvas = tk.Canvas(content)
		self.canvas.config(width=input_width, height=input_height)
		self.canvas.grid(column=0, row=0, rowspan=3)
		
		# Create frame that will contain point labels
		lbl_content = ttk.Frame(content)
		lbl_content.grid(column=1, row=0)
		
		# Create point labels
		lbl_line0 = ttk.Label(lbl_content, text="Line 1 (x1 y1 x2 y2):")
		lbl_line0.grid(column=0, row=0, sticky='E')
		lbl_line1 = ttk.Label(lbl_content, text="Line 2 (x3 y3 x4 y4):")
		lbl_line1.grid(column=0, row=1, sticky='E')
		lbl_line2 = ttk.Label(lbl_content, text="Line 3 (x5 y5 x6 y6):")
		lbl_line2.grid(column=0, row=2, sticky='E')
		lbl_line3 = ttk.Label(lbl_content, text="Line 4 (x7 y7 x8 y8):")
		lbl_line3.grid(column=0, row=3, sticky='E')
		lbl_coord0 = ttk.Label(lbl_content, textvariable=self.var_coords[0])
		lbl_coord0.grid(column=1, row=0, sticky='E')
		lbl_coord1 = ttk.Label(lbl_content, textvariable=self.var_coords[1])
		lbl_coord1.grid(column=2, row=0, sticky='E')
		lbl_coord2 = ttk.Label(lbl_content, textvariable=self.var_coords[2])
		lbl_coord2.grid(column=1, row=1, sticky='E')
		lbl_coord3 = ttk.Label(lbl_content, textvariable=self.var_coords[3])
		lbl_coord3.grid(column=2, row=1, sticky='E')
		lbl_coord4 = ttk.Label(lbl_content, textvariable=self.var_coords[4])
		lbl_coord4.grid(column=1, row=2, sticky='E')
		lbl_coord5 = ttk.Label(lbl_content, textvariable=self.var_coords[5])
		lbl_coord5.grid(column=2, row=2, sticky='E')
		lbl_coord6 = ttk.Label(lbl_content, textvariable=self.var_coords[6])
		lbl_coord6.grid(column=1, row=3, sticky='E')
		lbl_coord7 = ttk.Label(lbl_content, textvariable=self.var_coords[7])
		lbl_coord7.grid(column=2, row=3, sticky='E')
		
		# Create frame for output width widgets
		output_width_content = ttk.Frame(content)
		output_width_content.grid(column=1, row=1)
		
		# Create label for output width
		lbl_output_width = ttk.Label(output_width_content, text="Desired output width:")
		lbl_output_width.grid(column=0, row=0, sticky='E')
		
		# Create entry for output image width
		self.var_output_width = tk.StringVar()
		self.var_output_width.set("800")
		output_width_entry = ttk.Entry(output_width_content, textvariable=self.var_output_width)
		output_width_entry.grid(column=1, row=0)
		
		# Create button for running algorithm
		button = ttk.Button(content, text="Rectify!", command=lambda:rectify(self))
		button.grid(column=1, row=2)
		
		# Draw image in canvas
		self.photo = ImageTk.PhotoImage(image)
		canvasImg = self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
		
		content.pack()
	
		# Bind function to LMB click event
		self.canvas.bind("<Button 1>", self.update_canvas)
	
	def erase_point(self):
		self.canvas.delete(point_circle_tag_prefix + str(self.current_point))
		self.canvas.delete(point_center_tag_prefix + str(self.current_point))
		self.canvas.delete(point_text_tag_prefix + str(self.current_point))

	def draw_point(self, event):
		self.canvas.create_oval(event.x - point_diameter/2, event.y - point_diameter/2, event.x + point_diameter/2, event.y + point_diameter/2, outline=draw_color, width=2.0, tags=point_circle_tag_prefix + str(self.current_point))
		self.canvas.create_oval(event.x - 1, event.y - 1, event.x + 1, event.y + 1, outline=draw_color, fill=draw_color, tags=point_center_tag_prefix + str(self.current_point))
		self.canvas.create_text(event.x - point_diameter, event.y - point_diameter, anchor='s', fill=draw_color, text=str(self.current_point + 1), tags=point_text_tag_prefix + str(self.current_point), font=font.Font(family="Helvetica", size=12, weight="bold"))
		self.points[self.current_point] = (event.x, event.y)
		self.var_coords[self.current_point].set(self.points[self.current_point])
		
	def redraw_line(self):
		current_line = int(self.current_point / 2)
		line_tag = line_tag_prefix + str(current_line)
		self.canvas.delete(line_tag)
		self.canvas.create_line(self.points[current_line*2][0], self.points[current_line*2][1], self.points[current_line*2+1][0], self.points[current_line*2+1][1], fill=draw_color, tags=line_tag, width=2.0)
		
	def update_current_point(self):
		if self.current_point == 7:
			self.current_point = 0
		else:
			self.current_point += 1
		
	# Function to be called when LMB is clicked
	def update_canvas(self, event):
		self.erase_point()
		self.draw_point(event)
		self.redraw_line()
		self.update_current_point()

# Solves Ax = b for x
def solve(points):	
	line0 = np.cross([points[0][0], points[0][1], 1], [points[1][0], points[1][1], 1])
	line1 = np.cross([points[2][0], points[2][1], 1], [points[3][0], points[3][1], 1])
	line2 = np.cross([points[4][0], points[4][1], 1], [points[5][0], points[5][1], 1])
	line3 = np.cross([points[6][0], points[6][1], 1], [points[7][0], points[7][1], 1])
	print(line0, line1, line2, line3)
	
	inf_point0 = np.cross(line0, line1)
	inf_point1 = np.cross(line2, line3)
	print(inf_point0, inf_point1)
	
	inf_line = np.cross(inf_point0, inf_point1)
	print(inf_line)
	
	ret = np.array([[1,0,0],[0,1,0],inf_line])
	print(ret)
	
	inv_ret = np.linalg.inv(np.transpose(ret))
	print(inv_ret.dot(inf_line))
	
	return ret

def draw(h, desired_width, desired_height, left, top, ratio):
	# Open input image
	im = Image.open(os.path.join(input_dir, input_image))
	pixelMap = im.load()
	
	# Create new image for output
	img = Image.new(im.mode, (desired_width, desired_height))
	pixelsNew = img.load()
	
	# Draw transformed image
	for i in range(desired_width):
		for j in range(desired_height):
			pixel_x = left + i * (ratio[0] / desired_width)
			pixel_y = top + j * (ratio[1] / desired_height)
			img_pos = h.dot(np.array([pixel_x, pixel_y, 1]))
			img_pos = np.divide(img_pos, img_pos[2])
			if img_pos[0] >= 0 and img_pos[0] < input_width and img_pos[1] >= 0 and img_pos[1] < input_height:
				pixelsNew[i,j] = pixelMap[int(img_pos[0]),int(img_pos[1])]
	
	# Show and save transformed image
	img.show()
	#img.save(str.format("{0}_method1_{1}by{2}.jpg", os.path.splitext(input_image)[0], desired_width, desired_height))
	
# Function to be called when 'Rectify' button is clicked
def rectify(content):
	# Get matrix H and its inverse
	x = solve(content.points)
	inv_x = np.linalg.inv(x)
	
	# Transform input image corners to real world coordinates
	topleft_real = x.dot(np.array([0, 0, 1]))
	topleft_real = np.divide(topleft_real, topleft_real[2])
	bottomleft_real = x.dot(np.array([0, input_height, 1]))
	bottomleft_real = np.divide(bottomleft_real, bottomleft_real[2])
	bottomright_real = x.dot(np.array([input_width, input_height, 1]))
	bottomright_real = np.divide(bottomright_real, bottomright_real[2])
	topright_real = x.dot(np.array([input_width, 0, 1]))
	topright_real = np.divide(topright_real, topright_real[2])
	
	# Get bounds for transformed image
	left = min(topleft_real[0], bottomleft_real[0], topright_real[0], bottomright_real[0])
	top = min(topleft_real[1], bottomleft_real[1], topright_real[1], bottomright_real[1])
	right = max(topleft_real[0], bottomleft_real[0], topright_real[0], bottomright_real[0])
	bottom = max(topleft_real[1], bottomleft_real[1], topright_real[1], bottomright_real[1])
	
	# Calculate desired dimensions
	ratio = (right - left, bottom - top)
	desired_width = int(content.var_output_width.get())
	if desired_width < 0 or desired_width > 1920: # avoid errors and very large images
		desired_width = 800
	
	desired_height = int(desired_width * (ratio[1] / ratio[0]))
	draw(inv_x, desired_width, desired_height, left, top, ratio)

if __name__ == "__main__":
	# Read command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", help="The directory that contains the input image. If not given, it's assumed to be the same as the one in which the script file is located.")
	parser.add_argument("-i", required=True, help="The name of the input image file. Passing this requires passing -s as well.")
	args = vars(parser.parse_args())
	
	# Load command line arguments accordingly
	if args["d"]:
		input_dir = args["d"]
	input_image = args["i"]

	# Create Tkinter window with custom content
	root = tk.Tk()
	root.title("Rectify")
	myContent = MyContent(root)
	root.mainloop()
