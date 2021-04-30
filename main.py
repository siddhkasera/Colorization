import sys
from tkinter import filedialog
from PIL import Image
from BasicAgent import BasicAgent

print("Get file from prompt")
filePath = filedialog.askopenfilename()
if not filePath:
    print("No file detected")
    sys.exit()


numColors = input("How many colors do you want to use? (standard is 5) ")
if not numColors.isnumeric():
    print("No numeric value recognized, using the default 5")
    numColors = "5"

img = Image.open(filePath)

select = input("Which agent do you want to use? B for basic, or I for improved ")
if select == "B" or select == "b" or select == "Basic" or select == "basic":
    a = BasicAgent(img, int(numColors))

elif select == "i" or select == "I" or select == "Improved" or select == "improved" or select == "Advanced":
    print("Improved agent not implemented")
