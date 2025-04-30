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
    { id: 1, whitePlayer: 'Magnus Carlsen', whiteRating: 2864, blackPlayer: 'Fabiano Caruana', blackRating: 2781 },
    { id: 2, whitePlayer: 'Hikaru Nakamura', whiteRating: 2768, blackPlayer: 'Ian Nepomniachtchi', blackRating: 2775 },
    { id: 3, whitePlayer: 'Alireza Firouzja', whiteRating: 2759, blackPlayer: 'Ding Liren', blackRating: 2788 },
    { id: 4, whitePlayer: 'Anish Giri', whiteRating: 2754, blackPlayer: 'Wesley So', blackRating: 2760 },
    { id: 5, whitePlayer: 'Levon Aronian', whiteRating: 2745, blackPlayer: 'Sergey Karjakin', blackRating: 2723 },
    { id: 6, whitePlayer: 'Teimour Radjabov', whiteRating: 2726, blackPlayer: 'Shakhriyar Mamedyarov', blackRating: 2741 },
    { id: 7, whitePlayer: 'Richard Rapport', whiteRating: 2719, blackPlayer: 'Maxime Vachier-Lagrave', blackRating: 2751 },
    { id: 8, whitePlayer: 'Viswanathan Anand', whiteRating: 2730, blackPlayer: 'Vladimir Kramnik', blackRating: 2720 }
  ];
  