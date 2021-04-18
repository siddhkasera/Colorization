import sys
from tkinter import filedialog

from PIL import Image
from BasicAgent import BasicAgent

filePath = filedialog.askopenfilename()
if not filePath:
    print("No file detected")
    sys.exit()

im = Image.open(filePath)
b = BasicAgent(im)
