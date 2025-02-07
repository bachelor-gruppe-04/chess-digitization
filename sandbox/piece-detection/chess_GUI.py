# import tkinter as tk
# from create_chess_board import ChessBoard
# from eval_bar import ChessEvalBar
# from PIL import Image, ImageTk
# import cairosvg
# import chess.svg
# import io

# class ChessGUI:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Chessboard")

#         # Create frames for layout
#         self.frame_left = tk.Frame(root)
#         self.frame_left.pack(side="left", padx=10, pady=10)

#         self.frame_right = tk.Frame(root)
#         self.frame_right.pack(side="right", padx=10, pady=10)

#         # Create an instance of ChessBoard
#         self.board = ChessBoard()

#         # Create canvas for the chessboard and place it in the right frame
#         self.canvas = tk.Canvas(self.frame_right, width=400, height=400)
#         self.canvas.pack()

#         # Create an instance of ChessEvalBar and place it in the left frame
#         self.eval_bar = ChessEvalBar(self.frame_left, self.board)

#         self.piece_images = {}  # Store piece images
#         self.load_piece_images()

#         self.draw_board()

#         # Schedule the first move
#         self.schedule_random_move()

#         # Bind the custom event to update the evaluation bar
#         self.root.bind("<<UpdateEvalBar>>", self.update_eval_bar)

#     def load_piece_images(self):
#         """Loads chess piece images from SVGs and converts them to Tkinter format"""
#         piece_symbols = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
#         for symbol in piece_symbols:
#             svg_data = chess.svg.piece(chess.Piece.from_symbol(symbol))
#             png_data = io.BytesIO()
#             cairosvg.svg2png(bytestring=svg_data.encode('utf-8'), write_to=png_data)
#             image = Image.open(png_data)
#             image = image.resize((50, 50), Image.LANCZOS)  # Resize to fit the squares
#             self.piece_images[symbol] = ImageTk.PhotoImage(image)

#     def draw_board(self, best_move="e2e4"):
#         """Draws the chessboard with pieces and optional best move arrow"""
#         self.canvas.delete("all")  # Clear previous drawings
#         colors = ["#D18B47", "#FFCE9E"]
#         square_size = 50

#         for row in range(8):
#             for col in range(8):
#                 color = colors[(row + col) % 2]
#                 self.canvas.create_rectangle(col * square_size, row * square_size,
#                                             (col + 1) * square_size, (row + 1) * square_size,
#                                             fill=color, outline="black")

#         # Draw pieces using images
#         for square, piece in self.board.board.piece_map().items():
#             piece_symbol = str(piece)
#             row, col = divmod(square, 8)
#             x, y = (col * square_size), ((7 - row) * square_size)  # Flip row
#             self.canvas.create_image(x + square_size / 2, y + square_size / 2,
#                                     image=self.piece_images[piece_symbol])

#         # Draw best move arrow if provided
#         if best_move:
#             self.draw_arrow(best_move)

#     def draw_arrow(self, move):
#         """Draws an arrow from the start square to the end square"""
#         square_size = 50

#         start_square = chess.SQUARE_NAMES.index(move[:2])  # e2 -> index
#         end_square = chess.SQUARE_NAMES.index(move[2:])  # e4 -> index

#         start_row, start_col = divmod(start_square, 8)
#         end_row, end_col = divmod(end_square, 8)

#         x1, y1 = (start_col * square_size) + square_size // 2, (7 - start_row) * square_size + square_size // 2
#         x2, y2 = (end_col * square_size) + square_size // 2, (7 - end_row) * square_size + square_size // 2

#         self.canvas.create_line(x1, y1, x2, y2, fill="red", width=3, arrow=tk.LAST, arrowshape=(10, 15, 6))

#     def update_best_move(self, best_move):
#         """Updates the board with the best move arrow"""
#         self.draw_board(best_move)

#     def schedule_random_move(self):
#         self.root.after(500, self.make_random_move)  # Schedule the next move in 3 seconds

#     def make_random_move(self):
#         move = self.board.schedule_random_move()
#         if move:
#             self.draw_board(move)  # You can also call the update_best_move here
#         self.schedule_random_move()  # Schedule the next move

#     def event_update_eval_bar(self):
#         self.root.event_generate("<<UpdateEvalBar>>", when="tail")

#     def update_eval_bar(self, event):
#         """Handles the event to update the eval bar"""
#         self.eval_bar.fetch_eval()  # Fetch and update the evaluation score
