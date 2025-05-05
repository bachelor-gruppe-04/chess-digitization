import { useState, useEffect } from "react";

const useEvaluation = (fen: string, depth: number = 15) => {
  const [evaluation, setEvaluation] = useState<number | string | null>(null);  // Update state to accept string (mate)

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
