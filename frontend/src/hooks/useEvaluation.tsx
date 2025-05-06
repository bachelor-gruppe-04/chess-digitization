import { useState, useEffect } from "react";

/**
 * useEvaluation Hook
 *
 * Fetches a chess evaluation (centipawn or mate) from the Stockfish.online API based on a given FEN string.
 * Automatically updates when the FEN or depth changes.
 * Supports both numeric evaluations and mate announcements (e.g., "M1", "-M3").
 *
 * @param fen   - The FEN string representing the current board position
 * @param depth - (Optional) The search depth to use for the engine (default: 15)
 *
 * @returns evaluation - Can be:
 *   - A number (e.g., 0.5)
 *   - A string representing mate (e.g., "M1")
 *   - Null if no evaluation is available or on error
 */
const useEvaluation = (fen: string, depth: number = 15) => {
  const [evaluation, setEvaluation] = useState<number | string | null>(null);  // Stores the current evaluation or mate string

  useEffect(() => {
    const fetchStockfishEvaluation = async () => {
      const url = "https://stockfish.online/api/s/v2.php";
      const params = new URLSearchParams({ fen, depth: depth.toString() });

      try {
        const response = await fetch(`${url}?${params.toString()}`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        console.log("Stockfish evaluation result:", data);

        // Check if 'evaluation' exists (regular evaluation)
        if (data.evaluation) {
          setEvaluation(data.evaluation);
        } 
        // Check for mate information (M1, -M1, M2, etc.)
        else if (data.mate) {
          setEvaluation(`M${data.mate}`);  // Return the mate value
        } else {
          setEvaluation(null); // Reset to null if no evaluation or mate
        }
        console.log("Evaluation set to:", data.evaluation || data.mate);
      } catch (error) {
        console.error("Error fetching evaluation from Stockfish API:", error);
        setEvaluation(null); // Reset to null in case of error
      }
    };

    fetchStockfishEvaluation();
  }, [fen, depth]); // Re-run when FEN or depth changes

  return evaluation;
};

export default useEvaluation;
