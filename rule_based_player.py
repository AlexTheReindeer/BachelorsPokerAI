import random
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
from pypokerengine.engine.hand_evaluator import HandEvaluator

class RuleBasedPlayer(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        self.hole_card = None
        self.community_card = None
        self.game_state = None
        self.round_state = None

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.hole_card = hole_card
        self.round_state = {'seats': seats}

    def receive_street_start_message(self, street, round_state):
        self.round_state.update(round_state)
        if 'community_cards' in round_state:
            self.community_card = round_state['community_cards']

    def receive_game_update_message(self, new_action, round_state):
        self.round_state.update(round_state)

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

    def calculate_pot_odds(self, round_state):
        """Calculate pot odds for calling"""
        pot_size = sum(seat.get('bet', 0) for seat in round_state.get('seats', []))
        pot_amount = round_state.get('pot', {}).get('main', {}).get('amount', 0)
        return pot_size / (pot_size + pot_amount) if (pot_size + pot_amount) > 0 else 0

    def estimate_hand_strength(self):
        """Estimate hand strength using win rate and hand rank"""
        if not self.hole_card:
            return 0.0
            
        # Convert card strings to card objects
        hole_cards = gen_cards(self.hole_card)
        community_cards = gen_cards(self.community_card) if self.community_card else []
        
        # Calculate win rate
        win_rate = estimate_hole_card_win_rate(
            nb_player=2,
            hole_card=hole_cards,
            community_card=community_cards,
            nb_simulation=100
        )
        
        # Calculate hand rank if we have community cards
        if community_cards:
            hand_rank = HandEvaluator.eval_hand(hole_cards, community_cards)
            # Normalize hand rank (lower is better)
            hand_rank_score = 1 - (hand_rank / 7462)  # 7462 is max rank
            
            # Combine win rate and hand rank
            return (win_rate + hand_rank_score) / 2
        else:
            return win_rate

    def declare_action(self, valid_actions, hole_card, round_state):
        """Declare an action based on hand strength and pot odds"""
        # Update game state
        self.hole_card = hole_card
        self.community_card = round_state.get('community_cards', [])
        self.round_state = round_state
        
        # Calculate pot odds
        pot_odds = self.calculate_pot_odds(round_state)
        
        # Estimate hand strength
        hand_strength = self.estimate_hand_strength()
        
        # Define action thresholds
        STRONG_HAND = 0.8
        MARGINAL_HAND = 0.5
        
        # Make decision based on hand strength and pot odds
        if hand_strength > STRONG_HAND:
            # Strong hand - raise or call
            if pot_odds < hand_strength:
                # Good pot odds - call
                for action in valid_actions:
                    if action['action'] == 'call':
                        return action['action'], action['amount']
            else:
                # Raise to build pot
                for action in valid_actions:
                    if action['action'] == 'raise':
                        return action['action'], action['amount']['min']
        elif hand_strength > MARGINAL_HAND:
            # Marginal hand - call if pot odds are good
            if pot_odds < hand_strength:
                for action in valid_actions:
                    if action['action'] == 'call':
                        return action['action'], action['amount']
            else:
                # Fold if pot odds are bad
                for action in valid_actions:
                    if action['action'] == 'fold':
                        return action['action'], action['amount']
        else:
            # Weak hand - fold
            for action in valid_actions:
                if action['action'] == 'fold':
                    return action['action'], action['amount']
        
        # Default to fold if no valid action found
        return 'fold', 0

    def raise_action(self, valid_actions, amount):
        """Helper method to raise"""
        for action in valid_actions:
            if action['action'] == 'raise':
                return action['action'], min(amount, action['amount']['max'])
        return self.call_action(valid_actions)
    
    def call_action(self, valid_actions):
        """Helper method to call"""
        for action in valid_actions:
            if action['action'] == 'call':
                return action['action'], action['amount']
        return self.fold_action(valid_actions)
    
    def fold_action(self, valid_actions):
        """Helper method to fold"""
        for action in valid_actions:
            if action['action'] == 'fold':
                return action['action'], action['amount']
        return self.call_action(valid_actions) 