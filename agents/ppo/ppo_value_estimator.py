from treys import Evaluator, Card, Deck
import pandas as pd

class ValueEstimator:
    def __init__(self):
        self.evaluator = Evaluator()

        # Define the order of ranks and suits as given
        self.RANKS = ['A','2','3','4','5','6','7','8','9','T','J','Q','K']
        self.SUITS = ['s','h','d','c']  # Spades, Hearts, Diamonds, Clubs
        
        # Precompute a list of all cards in the given order
        self.all_cards = []
        for suit in self.SUITS:
            for rank in self.RANKS:
                self.all_cards.append(rank + suit)

                # Load the preflop lookup table
        # The CSV should have columns: "Hole Cards", "Win Prob", "Tie Prob"
        self.preflop_data = pd.read_csv('preflop_probs.csv')
        # Create a dictionary for quick lookup: { 'AAo': win_prob, 'KKo': win_prob, ... }
        self.preflop_lookup = {row['Hole Cards'].strip(): float(row['Win Prob'])
                               for _, row in self.preflop_data.iterrows()}

        # Map ranks to a numerical value for sorting by strength (A high)
        self.rank_strength = {
            'A': 14, 'K': 13, 'Q': 12, 'J': 11,
            'T': 10, '9': 9, '8': 8, '7': 7,
            '6': 6, '5': 5, '4': 4, '3': 3, '2': 2
        }


    def calculate_heuristic_win_prob(self, card_vector):
        """
        Calculate the heuristic win probability using a one-hot encoded 52-length vector.

        card_vector: List or array of length 52 with 0/1 indicating which cards are present.
                     The first two '1's found will be considered as hole cards,
                     and the remaining '1's (if any) are considered as board cards.
        """

        if len(card_vector) > 52:
            card_vector = card_vector[:52]

        selected_indices = [i for i, val in enumerate(card_vector) if val == 1]

        if len(selected_indices) < 2:
            raise ValueError("At least 2 cards are needed (2 hole cards).")

        # First two chosen as hole cards, rest as board
        hole_indices = selected_indices[:2]
        board_indices = selected_indices[2:]

        # Convert indices to card strings
        hole_cards_str = [self.all_cards[i] for i in hole_indices]
        board_cards_str = [self.all_cards[i] for i in board_indices]

        # Check if preflop (no board cards)
        if len(board_indices) == 0:
            # Preflop scenario: Use the lookup table
            # Construct a key like 'AAo' or 'AKs' from hole_cards_str
            preflop_key = self._construct_preflop_key(hole_cards_str)

            if preflop_key in self.preflop_lookup:
                return self.preflop_lookup[preflop_key]
            else:
                # If not found in lookup, return a default value
                return 0.5
        else:
            # Postflop scenario: Use the heuristic (Treys evaluator)
            hole_cards = [Card.new(c) for c in hole_cards_str]
            board = [Card.new(c) for c in board_cards_str]
            score = self.evaluator.evaluate(board, hole_cards)
            percentile = self.evaluator.get_five_card_rank_percentage(score)
            win_prob = 1.0 - percentile
            return win_prob
        
    def _construct_preflop_key(self, hole_cards_str):
        """
        Construct a preflop hand key (e.g., 'AAo', 'AKs') from two hole cards strings (e.g. ['As', 'Kd']) to index the csv.

        Steps:
        1. Extract ranks and suits from the two hole cards.
        2. Determine the higher-ranked card and put it first.
        3. Determine if suited ('s') or offsuit ('o').
        """

        # Extract ranks and suits
        ranks = [c[0] for c in hole_cards_str]
        suits = [c[1] for c in hole_cards_str]

        # Sort by rank strength so highest rank comes first
        # We have two cards, so we just find the order
        if self.rank_strength[ranks[0]] > self.rank_strength[ranks[1]]:
            high_rank, low_rank = ranks[0], ranks[1]
            suit1, suit2 = suits[0], suits[1]
        else:
            high_rank, low_rank = ranks[1], ranks[0]
            suit1, suit2 = suits[1], suits[0]

        # Determine suited or offsuit
        if suit1 == suit2:
            suited_char = 's'
        else:
            suited_char = 'o'

        # Construct hand key
        hand_key = f"{high_rank}{low_rank}{suited_char}"
        return hand_key

    def calculate_mc_win_prob(self, card_vector, num_samples=1000):
        """
        Monte Carlo approximation of heads-up win probability with full game simulation
        using a one-hot encoded 52-length vector for the hero's hole cards and board.

        card_vector: A list/array of length 52 with 0/1 indicating which cards are present.
                    The first two '1's found will be considered hole cards,
                    and the rest are considered board cards.
        """
        # Decode card_vector into hole_cards_str and board_cards_str
        selected_indices = [i for i, val in enumerate(card_vector) if val == 1]

        if len(selected_indices) < 2:
            raise ValueError("At least 2 cards are needed (2 hole cards).")

        hole_indices = selected_indices[:2]
        board_indices = selected_indices[2:]

        hole_cards_str = [self.all_cards[i] for i in hole_indices]
        board_cards_str = [self.all_cards[i] for i in board_indices]

        evaluator = Evaluator()

        # Convert hero's hole cards and board cards to Treys Card ints
        hero_hole = [Card.new(c) for c in hole_cards_str]
        board = [Card.new(c) for c in board_cards_str]

        known_cards = hero_hole + board

        wins, ties = 0, 0

        for _ in range(num_samples):
            # Initialize a new deck and remove known cards
            deck = Deck()
            for kc in known_cards:
                deck.cards.remove(kc)

            # Opponent hand
            opp_hand = deck.draw(2)

            # Complete the board if needed
            remaining_board = deck.draw(5 - len(board))
            full_board = board + remaining_board

            # Evaluate final hands
            hero_score = evaluator.evaluate(full_board, hero_hole)
            opp_score = evaluator.evaluate(full_board, opp_hand)

            if hero_score < opp_score:
                wins += 1
            elif hero_score == opp_score:
                ties += 1

        # Compute win probability
        win_prob = wins / num_samples
        return win_prob

if __name__ == "__main__":
    # Example usage
    value_estimator = ValueEstimator()
    print(value_estimator.preflop_data.head())