# python3.7
import tkinter as tk
from tkinter import ttk

import black_face
import glasses_drop
import tags2D
import blink_star
import tongue
import swap_face


window = tk.Tk()
window.title('Face Filters')
window.geometry('500x200+300+200')  # set window size and position

label = ttk.Label(window, text='               Please choose the sticker you want：',
                  font=('Arial', 20), background='white')
label.pack(ipadx=60, pady=30)

# stickers menu
tags = ttk.Combobox(window)
tags.pack(ipadx=60, pady=0)
tags['value'] = ('3D swap face（choose the face you want）', 'black face', 'glasses drop(live)',
                 'twinkling stars(when blinking)', 'Tongue sticking out', 'cat ears', 'cat paws(live)',
                 'hat', 'cartoon eyes', 'beard', 'Moustache')
tags_index = 0


# Execute function
def tags_chosen(event):
    global tags_index
    tags_index = tags.current()  # initial 0


tags.bind("<<ComboboxSelected>>", tags_chosen)


# set button
def hit_button():
    global tags_index
    if tags_index == 0:
        FaceSwap.main()
    elif tags_index == 1:
        black_face.black_face()
    elif tags_index == 2:
        glasses_drop.glasses_drop()
    elif tags_index == 3:
        blink_star.star()
    elif tags_index == 4:
        tongue.tongue()
    else:
        tags2D.tags(tags_index - 5)


bt = tk.Button(window, text='OK', font=('Arial', 12), width=10, height=2, command=hit_button)
bt.pack(pady=20)

if __name__ == "__main__":
    window.mainloop()

