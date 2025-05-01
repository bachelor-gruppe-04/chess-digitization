/**
 * boards Array
 *
 * Represents a list of chess games being played in the tournament.
 * Each object contains:
 * - `id`: Unique board number (used for routing and display)
 * - `whitePlayer`: Name of the player playing white
 * - `whiteRating`: White player's ELO rating
 * - `blackPlayer`: Name of the player playing black
 * - `blackRating`: Black player's ELO rating
 *
 */

export const boards = [
  { id: 1, whitePlayer: 'Magnus Carlsen', whiteRating: 2837, blackPlayer: 'Fabiano Caruana', blackRating: 2776 },
  { id: 2, whitePlayer: 'Hikaru Nakamura', whiteRating: 2804, blackPlayer: 'Ian Nepomniachtchi', blackRating: 2757 },
  { id: 3, whitePlayer: 'Alireza Firouzja', whiteRating: 2757, blackPlayer: 'Ding Liren', blackRating: 2734 },
  { id: 4, whitePlayer: 'Anish Giri', whiteRating: 2738, blackPlayer: 'Wei Yi', blackRating: 2751 },
  { id: 5, whitePlayer: 'Levon Aronian', whiteRating: 2747, blackPlayer: 'Gukesh Dommaraju', blackRating: 2787 },
  { id: 6, whitePlayer: 'Arjun Erigaisi', whiteRating: 2781, blackPlayer: 'Shakhriyar Mamedyarov', blackRating: 2745 },
  { id: 7, whitePlayer: 'Richard Rapport', whiteRating: 2722, blackPlayer: 'Maxime Vachier-Lagrave', blackRating: 2723 },
  { id: 8, whitePlayer: 'Viswanathan Anand', whiteRating: 2743, blackPlayer: 'Vincent Keymer', blackRating: 2720 }
];
  