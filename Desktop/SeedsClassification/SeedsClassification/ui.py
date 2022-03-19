'''from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter import simpledialog
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
from PIL import Image, ImageTk

main = tkinter.Tk()
main.title("Seed Classification") 
main.geometry("600x500")
font = ('times', 16, 'bold')
title = Label(main, text='Seed Classification', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=100,y=5)
title.pack()
font1 = ('times', 14, 'bold')
def upload ():
 global filename
 filename = ImageTk.PhotoImage(filedialog.askopenfilename(initialdir="testimages"))
 #messagebox.showinfo("File Information", "image file loaded")
uploadimage = Button(main, text="Upload Test Image", command=upload)
uploadimage.place(x=200,y=150)
uploadimage.config(font=font1)
classifyimage = Button(main, text="Classify Picture In Image", command=classify)
classifyimage.place(x=200,y=200)
classifyimage.config(font=font1)
exitapp = Button(main, text="Extension Gabor Filter vs LBP Accuracy",
command=extensionGabor)
exitapp.place(x=200,y=250)
exitapp.config(font=font1)
main.config(bg='light coral')
main.mainloop()'''
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk

my_w = tk.Tk()
my_w.geometry("410x300")  # Size of the window 
my_w.title('major project')
my_font1=('times', 18, 'bold')
l1 = tk.Label(my_w,text='Seed Classification',width=30,font=my_font1)  
l1.grid(row=1,column=1,columnspan=4)
b1 = tk.Button(my_w, text='Upload File', 
   width=20,command = lambda:upload_file())
b1.grid(row=2,column=1,columnspan=4)

def upload_file():
    f_types = [('Jpg Files', '*.jpg'),
    ('PNG Files','*.png')]   # type of files to select 
    filename = tk.filedialog.askopenfilename(multiple=True,filetypes=f_types)
    col=1 # start from column 1
    row=3 # start from row 3 
    for f in filename:
        img=Image.open(f) # read the image file
        img=img.resize((100,100)) # new width & height
        img=ImageTk.PhotoImage(img)
        e1 =tk.Label(my_w)
        e1.grid(row=row,column=col)
        e1.image = img
        e1['image']=img # garbage collection 
        if(col==3): # start new line after third column
            row=row+1# start wtih next row
            col=1    # start with first column
        else:       # within the same row 
            col=col+1 # increase to next column       



my_w.mainloop()  # Keep the window open

