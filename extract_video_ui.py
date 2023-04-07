import cv2
import os
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
# Import additional modules
import numpy as np


# Global variables
video_path = ""
frame_number = 0
regions = []

# Set up the UI
root = Tk()
root.title("Video Region Extractor")
root.geometry("800x600")


# Custom Canvas for handling mouse events and displaying preview
class PreviewCanvas(Canvas):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.bind("<Button-1>", self.mouse_callback)
        self.bind("<B1-Motion>", self.mouse_callback)
        self.bind("<ButtonRelease-1>", self.mouse_callback)

    def mouse_callback(self, event):
        x, y = int(event.x / self.ratio), int(event.y / self.ratio)
        event_type = None
        if event.type == "4":  # Button Press
            event_type = cv2.EVENT_LBUTTONDOWN
        elif event.type == "5":  # Button Release
            event_type = cv2.EVENT_LBUTTONUP
        elif event.type == "6":  # Mouse Move
            event_type = cv2.EVENT_MOUSEMOVE
        mouse_callback(event_type, x, y, None, None)

# Set up the preview window
preview_canvas = PreviewCanvas(root, width=500, height=500)
preview_canvas.grid(row=1, column=0, columnspan=3, padx=10, pady=10)



# Select the video file
def select_file():
    global video_path, frame_number
    video_path = filedialog.askopenfilename()
    file_path_entry.delete(0, END)
    file_path_entry.insert(0, video_path)
    update_frame()
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    slider.configure(to=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)-1))
    slider.set(0)
    cap.release()

# Update the preview frame
def update_frame():
    global frame_number
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret == True:
        # Draw selected regions on frame
        for (x, y, w, h) in regions:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Resize and display the frame
        height, width, channels = frame.shape
        ratio = 1
        if width > height:
            ratio = 500 / width
        else:
            ratio = 500 / height
        resized_frame = cv2.resize(frame, (int(width*ratio), int(height*ratio)))
        preview_canvas.ratio = ratio
        cv2image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        preview_canvas.imgtk = imgtk
        preview_canvas.create_image(0, 0, image=imgtk, anchor=NW)
    cap.release()

# Handle slider movement
def slider_moved(value):
    global frame_number
    frame_number = int(value)
    update_frame()


# Create the output directories
def create_directories():
    output_path = os.path.splitext(video_path)[0]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i in range(len(regions)):
        region_path = os.path.join(output_path, "region_" + str(i))
        if not os.path.exists(region_path):
            os.makedirs(region_path)

# Extract the selected regions from the video and save as separate video files
# Update the preview frame
def update_frame():
    global frame_number
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret == True:
        # Draw selected regions on frame
        for (x, y, w, h) in regions:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Resize and display the frame
        height, width, channels = frame.shape
        ratio = 1
        if width > height:
            ratio = 500 / width
        else:
            ratio = 500 / height
        resized_frame = cv2.resize(frame, (int(width*ratio), int(height*ratio)))
        cv2image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        preview_label.imgtk = imgtk
        preview_label.configure(image=imgtk)
    cap.release()

# Handle mouse events on preview window
def mouse_callback(event, x, y, flags, param):
    global regions
    if event == cv2.EVENT_LBUTTONDOWN:
        # Start dragging a new region
        regions.append((x, y, 0, 0))
    elif event == cv2.EVENT_MOUSEMOVE:
        # Update the last region being dragged
        if len(regions) > 0:
            x1, y1, _, _ = regions[-1]
            w = x - x1
            h = y - y1
            regions[-1] = (x1, y1, w, h)
            update_frame()
    elif event == cv2.EVENT_LBUTTONUP:
        # Stop dragging the last region
        if len(regions) > 0:
            x1, y1, w, h = regions[-1]
            if w < 0:
                x1 += w
                w = abs(w)
            if h < 0:
                y1 += h
                h = abs(h)
            regions[-1] = (x1, y1, w, h)
            update_frame()

# Extract the selected regions from the video and save as separate video files
def extract_regions():
    # Create video capture object and get video properties
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create video writer objects for all selected regions
    output_path = os.path.splitext(video_path)[0]
    writers = []
    for j, (x, y, w, h) in enumerate(regions):
        region_path = os.path.join(output_path, "region_" + str(j), "output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(region_path, fourcc, fps, (w, h))
        writers.append(writer)

    # Process each frame of the video and write to the appropriate video writer
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        for j, (x, y, w, h) in enumerate(regions):
            region_frame = frame[y:y+h, x:x+w]
            writers[j].write(region_frame)

    # Release video capture and writer objects
    cap.release()
    for writer in writers:
        writer.release()

    messagebox.showinfo("Extraction Complete", "Regions extracted successfully!")



# Set up the file path entry
file_path_label = Label(root, text="Video File:")
file_path_label.grid(row=0, column=0, padx=10, pady=10)
file_path_entry = Entry(root, width=50)
file_path_entry.grid(row=0, column=1, padx=10, pady=10)
select_file_button = Button(root, text="Select", command=select_file)
select_file_button.grid(row=0, column=2, padx=10, pady=10)

# Set up the preview window
preview_label = Label(root)
preview_label.grid(row=1, column=0, columnspan=3, padx=10, pady=10)
cv2.startWindowThread()
cv2.setMouseCallback("Preview", mouse_callback)

# Set up the slider
slider_frame = Frame(root)
slider_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=10)
slider_label = Label(slider_frame, text="Frame Number:")
slider_label.grid(row=0, column=0, sticky=W)
slider_number_label = Label(slider_frame)
slider_number_label.grid(row=0, column=1, padx=10, sticky=W)
slider = Scale(slider_frame, from_=0, to=0, command=slider_moved, orient=HORIZONTAL, length=600, resolution=1, troughcolor="white", takefocus=1)
slider.grid(row=0, column=2, sticky=W)


# Set up the region selection buttons
select_region_button = Button(root, text="Select Region", command=lambda:[create_directories(), extract_regions()])
select_region_button.grid(row=3, column=0, padx=10, pady=10)
clear_regions_button = Button(root, text="Clear Regions", command=lambda:[regions.clear(), update_frame()])
clear_regions_button.grid(row=3, column=1, padx=10, pady=10)

# Set up the extraction button
extract_button = Button(root, text="Extract Regions", command=lambda:[create_directories(), extract_regions()])
extract_button.grid(row=3, column=2, padx=10, pady=10)

root.mainloop()
