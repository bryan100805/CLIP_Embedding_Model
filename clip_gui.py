import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from glob import glob

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_viT_32, preprocess_viT_32 = clip.load("./openaiclipweights/clip/CLIP/models/ViT-B-32.pt")
model_viT_32.cpu().eval()

# Precompute image embeddings
files = glob('*.jpeg') + glob('*.jpg') + glob('*.png')
image_embeddings = []

for file in files:
    image = preprocess_viT_32(Image.open(file).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        image_embeddings.append(model_viT_32.encode_image(image).cpu().detach().numpy())

image_embeddings = np.vstack(image_embeddings)

QUERIES = [
    "A red toy car",
    "A blue toy car",
    "A pink toy car",
    "A black toy car",
    "A white toy car",
    "A silver toy car",
    "A toy car with racing stripes",
    "A toy car with big wheels",
    "A vintage toy car",
    "A toy car collection",
    "A toy car in a box",
    "A toy car with a driver inside",
    "A toy car with many passengers inside",
    "A toy car with an open roof",
    "A toy car and a toy truck",
    "A toy car on top of a toy truck",
    "A GTR toy car",
    "A Mazda toy car",
    "A Bugatti toy car",
    "Barbie toy car",
    "Two toy cars",
    "Four toy cars",
    "Eight toy cars",
    "A toy car in the UK",
    "A toy car moving in a city",
    "A police toy car"
]

# GUI functions
def process_input(user_input):
    if isinstance(user_input, str):
        # Text input
        text_embedding = model_viT_32.encode_text(clip.tokenize(user_input)).cpu().detach().numpy()
        similarities = (image_embeddings @ text_embedding.T).squeeze()
        best_match_idx = np.argmax(similarities)
        best_image = Image.open(files[best_match_idx])
        show_image(best_image)
        result_text.set(f"Query: {user_input}")

    elif isinstance(user_input, Image.Image):
        # Image input
        image_embedding = model_viT_32.encode_image(preprocess_viT_32(user_input).unsqueeze(0)).cpu().detach().numpy()
        query_embeddings = model_viT_32.encode_text(clip.tokenize(QUERIES)).cpu().detach().numpy()
        similarities = (query_embeddings @ image_embedding.T).squeeze()
        best_match_idx = np.argmax(similarities)
        result_text.set(f"Best Match: {QUERIES[best_match_idx]}")

def show_image(image):
    img = ImageTk.PhotoImage(image.resize((250, 250)))
    panel.configure(image=img)
    panel.image = img  # Keep a reference to avoid garbage collection

def browse_file():
    filename = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg")])
    if filename:
        image_input = Image.open(filename)
        show_image(image_input)
        process_input(image_input)

def submit_text():
    text_input = text_entry.get()
    if text_input:
        process_input(text_input)

# Setting up Tkinter window
root = tk.Tk()
root.title("CLIP Model GUI")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

# File input button
browse_button = tk.Button(frame, text="Upload Image", command=browse_file)
browse_button.grid(row=0, column=0, padx=5, pady=5)

# Text input field
text_entry = tk.Entry(frame, width=50)
text_entry.grid(row=1, column=0, padx=5, pady=5)

# Submit text button
submit_button = tk.Button(frame, text="Submit Text", command=submit_text)
submit_button.grid(row=1, column=1, padx=5, pady=5)

# Display area for images
panel = tk.Label(root)
panel.pack(padx=10, pady=10)

# Result label for displaying best match
result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=("Helvetica", 16))
result_label.pack(padx=10, pady=10)

root.mainloop()