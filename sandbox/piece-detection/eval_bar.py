import tkinter as tk
import chess_analysis

class ChessEvalBar:
    def __init__(self, root, board):
        self.root = root
        self.board = board  # Store the board reference
        self.canvas = tk.Canvas(root, width=50, height=400, bg="gray")
        self.canvas.pack()

        self.eval_score = 0
        self.target_eval_score = 0
        self.animation_steps = 100
        self.current_step = 0

        self.fetch_eval()
        self.update_bar()

        self.test_button = tk.Button(root, text="Analyze Position")
        self.test_button.pack()

    def fetch_eval(self):
        """Fetch the evaluation from stockfish and update bar"""
        fen = self.board.fen()  # Get the current FEN
        eval_score = chess_analysis.fetch_evaluation(fen)
        self.set_evaluation(eval_score)


    def set_evaluation(self, eval_score):
        """Sets a new evaluation score and updates the bar with animation"""
        self.target_eval_score = eval_score
        self.current_step = 0
        self.animate_bar()


    def update_bar(self):
        """Updates the evaluation bar based on self.eval_score"""
        self.canvas.delete("all")  # Clear canvas

        eval_clamped = max(min(self.eval_score, 10), -10)
        eval_bar_limited = max(min(self.eval_score, 8.5), -8.5)  # Limit bar movement

        # Convert eval to percentage based on the limited range
        eval_percentage = (eval_bar_limited + 10) / 20  # Normalize to 0-1 range
        bar_height = int(eval_percentage * 400)  # Scale to canvas height

        # Draw bar (white on top, black on bottom)
        self.canvas.create_rectangle(0, 0, 50, 400 - bar_height, fill="black", outline="")
        self.canvas.create_rectangle(0, 400 - bar_height, 50, 400, fill="white", outline="")

        # Display evaluation score (full range, so text can show ±10 even if bar is limited)
        if self.eval_score >= 0:
            self.canvas.create_text(25, 10, text=f"{eval_clamped:.1f}", fill="white", font=("Arial", 12, "bold"))
        else:
            self.canvas.create_text(25, 390, text=f"{eval_clamped:.1f}", fill="black", font=("Arial", 12, "bold"))

    def animate_bar(self):
        """Animates the evaluation bar towards the target score"""
        if self.current_step < self.animation_steps:
            # Calculate the intermediate evaluation score for this animation step
            step_eval = self.eval_score + (self.target_eval_score - self.eval_score) / (self.animation_steps - self.current_step)
            self.eval_score = step_eval
            self.update_bar()
            self.current_step += 1

            self.root.after(10, self.animate_bar)  # Call animate_bar again after 10ms