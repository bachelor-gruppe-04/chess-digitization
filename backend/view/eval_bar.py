import tkinter as tk
import logic.analysis.chess_analysis as chess_analysis

class ChessEvalBar:
    def __init__(self, root, board):
        """
        Initializes the ChessEvalBar class, setting up the canvas, evaluation variables, and UI components.

        Parameters:
            root (tk.Tk): The parent Tkinter window for the evaluation bar.
            board (chess.Board): The chessboard object representing the current game state.
        """
        self.root = root
        self.board = board
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
        """
        Fetches the current evaluation score from the chess analysis engine and updates the evaluation bar.

        Retrieves the FEN string of the current chess position and sends it to the external chess engine
        (e.g., Stockfish) to get the evaluation score, which is then used to update the evaluation bar.
        """
        fen = self.board.fen()
        eval_score = chess_analysis.fetch_evaluation(fen)
        self.set_evaluation(eval_score)

    def set_evaluation(self, eval_score):
        """
        Sets the evaluation score and triggers the animation to update the evaluation bar.

        Parameters:
            eval_score (float): The evaluation score from the chess analysis engine.
        """
        self.target_eval_score = eval_score
        self.current_step = 0
        self.animate_bar()



    def update_bar(self):
        """
        Updates the evaluation bar on the canvas based on the current evaluation score.

        This method clears the canvas and redraws the evaluation bar, showing the current evaluation score
        as a graphical representation. The score is clamped to a range of -10 to +10, and a portion of the
        canvas is filled according to the evaluation score.
        """
        self.canvas.delete("all")  # Clear canvas

        eval_clamped = max(min(self.eval_score, 10), -10)
        eval_bar_limited = max(min(self.eval_score, 8.5), -8.5)  # Limit bar movement

        # Convert eval to percentage based on the limited range
        eval_percentage = (eval_bar_limited + 10) / 20  # Normalize to 0-1 range
        bar_height = int(eval_percentage * 400)  # Scale to canvas height

        # Draw bar (white on top, black on bottom)
        self.canvas.create_rectangle(0, 0, 50, 400 - bar_height, fill="black", outline="")
        self.canvas.create_rectangle(0, 400 - bar_height, 50, 400, fill="white", outline="")

        if self.eval_score >= 0:
            self.canvas.create_text(25, 10, text=f"{eval_clamped:.1f}", fill="white", font=("Arial", 12, "bold"))
        else:
            self.canvas.create_text(25, 390, text=f"{eval_clamped:.1f}", fill="black", font=("Arial", 12, "bold"))



    def animate_bar(self):
        """
        Animates the evaluation bar towards the target evaluation score.

        This method gradually updates the evaluation score over a number of steps to smoothly animate
        the transition from the current score to the target score. The animation takes place over
        a series of small updates, and the `update_bar` method is called repeatedly to reflect
        the changes on the canvas.
        """
        if self.current_step < self.animation_steps:
            # Calculate the intermediate evaluation score for this animation step
            step_eval = self.eval_score + (self.target_eval_score - self.eval_score) / (self.animation_steps - self.current_step)
            self.eval_score = step_eval
            self.update_bar()
            self.current_step += 1

            self.root.after(10, self.animate_bar)