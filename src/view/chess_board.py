import tkinter as tk
import chess
import random
from view.eval_bar import ChessEvalBar
import chess.svg
from PIL import Image, ImageTk
import io
import cairosvg
import logic.analysis.chess_analysis as chess_analysis

class ChessBoard:
    def __init__(self, root):
        self.root = root
        self.root.title("Chessboard")

        self.frame_left = tk.Frame(root)
        self.frame_left.pack(side="left", padx=10, pady=10)

        self.frame_right = tk.Frame(root)
        self.frame_right.pack(side="right", padx=10, pady=10)

        self.board = chess.Board()

        self.canvas = tk.Canvas(self.frame_right, width=400, height=400)
        self.canvas.pack()

        self.eval_bar = ChessEvalBar(self.frame_left, self.board)

        self.piece_images = {}
        self.load_piece_images()

        self.draw_board()
        self.schedule_random_move()

        # Bind the custom event to update the evaluation bar
        self.root.bind("<<UpdateEvalBar>>", self.update_eval_bar)

    def load_piece_images(self):
        """Loads chess piece images from SVGs and converts them to Tkinter format"""
        piece_symbols = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
        for symbol in piece_symbols:
            svg_data = chess.svg.piece(chess.Piece.from_symbol(symbol))
            png_data = io.BytesIO()
            cairosvg.svg2png(bytestring=svg_data.encode('utf-8'), write_to=png_data)
            image = Image.open(png_data)
            image = image.resize((50, 50), Image.LANCZOS)  # Resize to fit the squares
            self.piece_images[symbol] = ImageTk.PhotoImage(image)

 
    def draw_board(self):
        """Draws the chessboard with pieces and optional best move arrow"""
        self.canvas.delete("all")
        colors = ["#D18B47", "#FFCE9E"]
        square_size = 50

        for row in range(8):
            for col in range(8):
                color = colors[(row + col) % 2]
                self.canvas.create_rectangle(col * square_size, row * square_size,
                                            (col + 1) * square_size, (row + 1) * square_size,
                                            fill=color, outline="black")

        for square, piece in self.board.piece_map().items():
            piece_symbol = str(piece)
            row, col = divmod(square, 8)
            x, y = (col * square_size), ((7 - row) * square_size)  # Flip row
            self.canvas.create_image(x + square_size / 2, y + square_size / 2,
                                    image=self.piece_images[piece_symbol])

        best_move = chess_analysis.fetch_best_move(self.board.fen())
        self.draw_arrow(best_move)


    def draw_arrow(self, move):
        """Draws an arrow from the start square to the end square"""
        square_size = 50

        start_square = chess.SQUARE_NAMES.index(move[:2])  # e2 -> index
        end_square = chess.SQUARE_NAMES.index(move[2:])  # e4 -> index

        start_row, start_col = divmod(start_square, 8)
        end_row, end_col = divmod(end_square, 8)

        x1, y1 = (start_col * square_size) + square_size // 2, (7 - start_row) * square_size + square_size // 2
        x2, y2 = (end_col * square_size) + square_size // 2, (7 - end_row) * square_size + square_size // 2

        self.canvas.create_line(x1, y1, x2, y2, fill="red", width=3, arrow=tk.LAST, arrowshape=(10, 15, 6))

    def update_best_move(self, best_move):
        """Updates the board with the best move arrow"""
        self.draw_board(best_move)


    def fen(self):
        return self.board.fen()
    
    def make_move(self, move):
        if self.board.is_legal(move):
            self.board.push(move)
            self.draw_board()
            self.event_update_eval_bar()

    def schedule_random_move(self):
        self.root.after(2000, self.make_random_move) 

    def make_random_move(self):
        legal_moves = list(self.board.legal_moves)
        if legal_moves:
            move = random.choice(legal_moves)
            self.make_move(move)
        self.schedule_random_move()  # Schedule the next move

    def event_update_eval_bar(self):
        self.root.event_generate("<<UpdateEvalBar>>", when="tail")

    def update_eval_bar(self, event):
        """Handles the event to update the eval bar"""
        self.eval_bar.fetch_eval() 
