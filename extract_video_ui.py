import tkinter as tk
import cv2
from tkinter import filedialog
from PIL import Image, ImageTk
from yolos_demo import detect_laptop
from trocr_demo import build_trocr_model, read_video

class VideoPlayer:
    def __init__(self, master):
        self.master = master
        self.master.title("Video Player")

        # Create variables to store the video and bounding box information
        self.video = None
        self.bounding_boxes = []

        # Create a canvas to display the video
        self.canvas = tk.Canvas(self.master)
        self.canvas.pack()
        self.canvas_width = 800  # Change to desired width
        self.canvas_height = 500  # Change to desired height
        self.canvas.config(width=self.canvas_width, height=self.canvas_height)

        # Create a button to load the video
        self.load_button = tk.Button(self.master, text="Load Video", command=self.load_video)
        self.load_button.pack(side="left")

        # Create a button to detect the laptop
        self.yolos_button = tk.Button(self.master, text="Laptop Detect", command=self.create_bounding_box)
        self.yolos_button.pack(side="left")

        # Create a button to clear the bounding boxes
        self.clear_bbox_button = tk.Button(self.master, text="Clear Bbox", command=self.clear_bounding_box)
        self.clear_bbox_button.pack(side="left")

        # Create a button to extract the bounding boxes into TrOCR
        self.read_text_button = tk.Button(self.master, text="Read Text", command=self.read_text)
        self.read_text_button.pack(side="left")

        # Create a slider to control the preview time frame
        self.slider = tk.Scale(self.master, from_=0, to=0, orient=tk.HORIZONTAL, command=self.update_preview)
        self.slider.pack()

        # Bind mouse events to the canvas
        self.canvas.bind("<ButtonPress-1>", self.start_drag)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.end_drag)
        
        self.trocr_model, self.trocr_processor = build_trocr_model()

    def load_video(self):
        # Open a file dialog to select a video file
        filetypes = (("Video files", "*.mp4"), ("All files", "*.*"))
        filepath = filedialog.askopenfilename(title="Select a video file", filetypes=filetypes)

        # Load the video using OpenCV
        self.video = cv2.VideoCapture(filepath)

        # Set the slider range to match the number of frames in the video
        self.slider.config(to=int(self.video.get(cv2.CAP_PROP_FRAME_COUNT)))

        # Display the first frame of the video on the canvas
        self.show_frame()

    def show_frame(self):
        # Set the current frame of the video to the value of the slider
        self.video.set(cv2.CAP_PROP_POS_FRAMES, int(self.slider.get()))

        # Read the current frame from the video
        ret, frame = self.video.read()

        # Convert the frame from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize the frame to fit on the canvas
        height, width, _ = frame.shape
        scale = min(self.canvas_width / width, self.canvas_height / height)
        resized = cv2.resize(frame, (int(width * scale), int(height * scale)))

        # Convert the resized frame to a PhotoImage object and display it on the canvas
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(resized))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # Draw the bounding boxes on the canvas
        for bbox in self.bounding_boxes:
            x1, y1, x2, y2 = bbox
            self.canvas.create_rectangle(x1 * scale, y1 * scale, x2 * scale, y2 * scale, outline="red")

    def start_drag(self, event):
        # Save the starting point of the drag
        self.drag_start = (event.x, event.y)

    def drag(self, event):
        # Delete the previous bounding box
        self.canvas.delete("bbox")

        # Draw the new bounding box
        x1, y1 = self.drag_start
        x2, y2 = event.x, event.y
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", tags="bbox")

    def end_drag(self, event):
        # Save the bounding box coordinates
        x1, y1 = self.drag_start
        x2, y2 = event.x, event.y

        # Read the current frame from the video
        ret, frame = self.video.read()
        height, width, _ = frame.shape
        scale = min(self.canvas_width / width, self.canvas_height / height)
        bbox = (x1, y1, x2, y2)
        bbox = ([int(_b/scale) for _b in bbox])
        self.bounding_boxes.append(bbox)

        # Clear the canvas
        self.canvas.delete("all")

        # Update the preview
        self.show_frame()

    def update_preview(self, value):
        # Set the current frame of the video to the value of the slider
        self.video.set(cv2.CAP_PROP_POS_FRAMES, int(value))

        # Update the preview
        self.show_frame()

    def run(self):
        # Start the main event loop
        self.master.mainloop()

    def create_bounding_box(self):
        # Detect the laptop bounding boxes and add them to the list of bounding boxes
        self.video.set(cv2.CAP_PROP_POS_FRAMES, int(self.slider.get()))
        _, frame = self.video.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        nms_boxes, nms_scores, labels = detect_laptop(image)
        self.bounding_boxes = nms_boxes.int().tolist()
        self.show_frame()

    def clear_bounding_box(self):
        # Detect the laptop bounding boxes and add them to the list of bounding boxes
        self.bounding_boxes = []
        self.canvas.delete("all")
        self.show_frame()

    def read_text(self):
        for i, bbox in enumerate(self.bounding_boxes):
            result = read_video(self.video, self.trocr_model, self.trocr_processor, bbox, device='mps')
            result_path = "result_%d.txt"%i
            with open(result_path, "w") as f:
                for res in result:
                    f.write('%s, %s\n' % (res[0], res[1]))


root = tk.Tk()
root.title("Video Region Extractor")
root.geometry("800x600")
app = VideoPlayer(root)
app.run()