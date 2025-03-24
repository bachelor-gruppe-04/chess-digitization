import tkinter as tk
from view.chess_board import ChessBoard

root = tk.Tk()
frame_right = tk.Frame(root)
frame_right.pack(side="right", padx=10, pady=10)

chess_app = ChessBoard(root)

root.mainloop()



