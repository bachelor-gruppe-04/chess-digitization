import { useState, useEffect } from "react";

const useEvaluation = (fen: string, depth: number = 15) => {
  const [evaluation, setEvaluation] = useState<number | null>(null);

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
        if (data.evaluation) {
          setEvaluation(data.evaluation);
        } else {
          setEvaluation(null);
        }
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
