from treys import Evaluator, Card, Deck

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

        # Convert to treys Card ints
        hole_cards = [Card.new(c) for c in hole_cards_str]
        board = [Card.new(c) for c in board_cards_str]

        # Evaluate the hand
        score = self.evaluator.evaluate(board, hole_cards)
        percentile = self.evaluator.get_five_card_rank_percentage(score)

        # Heuristic: 1 - percentile
        win_prob = 1.0 - percentile
        return win_prob

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
