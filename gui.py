# %%
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# %%
# Load the pre-trained model
model = load_model('age_gender_model.h5')

# %%
# Preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (200, 200))
    img = np.expand_dims(img, axis=0)  
    return img


# %%
# Function to load an image
def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image = ImageTk.PhotoImage(image)
        panel.configure(image=image)
        panel.image = image
        global img_path
        img_path = file_path  
        result_text.set("")  
        detect_btn.pack(pady=10)  

# %%
# Function to make predictions
def detect():
    if img_path:
        img = preprocess_image(img_path)
        age_pred, gender_pred = model.predict(img)
        age = int(age_pred[0])
        gender = 'Female' if gender_pred[0] > 0.5 else 'Male'
        result_text.set(f'Predicted Age: {age}\nPredicted Gender: {gender}')
    else:
        result_text.set("No image loaded. Please upload an image first.")

# %%
# Create the main window
root = tk.Tk()
root.geometry('800x600')
root.title("Age and Gender Detection")

# %%
# Add a label to display the selected image
panel = tk.Label(root)
panel.pack(padx=10, pady=10)

# %%
# Add a button to open the file dialog
btn = tk.Button(root, text="Upload Image", command=load_image)
btn.pack(pady=10)

# %%
# Add a button to detect age and gender, initially hidden
detect_btn = tk.Button(root, text="Detect", command=detect)
detect_btn.pack_forget()  

# %%
# Add a label to display the result
result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=('Helvetica', 12))
result_label.pack(pady=10)

# %%
# Start the GUI event loop
root.mainloop()


