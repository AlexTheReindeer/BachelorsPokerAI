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
        # Cache for hand evaluations
        self.hand_strength_cache = {}
        # Pre-calculate card mappings
        self.rank_values = {
            'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10,
            '9': 9, '8': 8, '7': 7, '6': 6, '5': 5,
            '4': 4, '3': 3, '2': 2
        }

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.hole_card = hole_card
        self.round_state = {'seats': seats}
        # Clear cache at start of new hand
        self.hand_strength_cache.clear()

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

    def quick_hand_strength(self):
        """Quick estimate of hand strength without simulations"""
        if not self.hole_card:
            return 0.0
            
        # Convert cards to numerical values
        hole_values = [self.rank_values[card[1:]] for card in self.hole_card]
        hole_suits = [card[0] for card in self.hole_card]
        
        # Check for pairs
        if hole_values[0] == hole_values[1]:
            return 0.8 + (hole_values[0] / 14.0) * 0.2
            
        # Check for suited cards
        if hole_suits[0] == hole_suits[1]:
            return 0.6 + (max(hole_values) / 14.0) * 0.2
            
        # Check for connected cards
        gap = abs(hole_values[0] - hole_values[1])
        if gap <= 2:
            return 0.5 + (max(hole_values) / 14.0) * 0.2
            
        # High cards
        return 0.3 + (max(hole_values) / 14.0) * 0.2

    def estimate_hand_strength(self):
        """Estimate hand strength using cached values and quick evaluation"""
        if not self.hole_card:
            return 0.0
            
        # Create cache key
        cache_key = tuple(sorted(self.hole_card))
        if self.community_card:
            cache_key += tuple(sorted(self.community_card))
            
        # Check cache
        if cache_key in self.hand_strength_cache:
            return self.hand_strength_cache[cache_key]
            
        # Quick evaluation for pre-flop
        if not self.community_card:
            strength = self.quick_hand_strength()
            self.hand_strength_cache[cache_key] = strength
            return strength
            
        # For post-flop, use a simplified evaluation
        hole_cards = gen_cards(self.hole_card)
        community_cards = gen_cards(self.community_card)
        
        # Calculate hand rank (lower is better)
        hand_rank = HandEvaluator.eval_hand(hole_cards, community_cards)
        strength = 1 - (hand_rank / 7462)  # Normalize to [0,1]
        
        # Cache the result
        self.hand_strength_cache[cache_key] = strength
        return strength

    def declare_action(self, valid_actions, hole_card, round_state):
        """Declare an action based on simplified decision logic"""
        # Update game state
        self.hole_card = hole_card
        self.community_card = round_state.get('community_cards', [])
        self.round_state = round_state
        
        # Get hand strength and pot odds
        hand_strength = self.estimate_hand_strength()
        pot_odds = self.calculate_pot_odds(round_state)
        
        # Simplified decision thresholds
        STRONG_HAND = 0.7
        MARGINAL_HAND = 0.4
        
        # Make decision based on hand strength and pot odds
        if hand_strength > STRONG_HAND:
            # Strong hand - raise or call
            for action in valid_actions:
                if action['action'] == 'raise':
                    return action['action'], action['amount']['min']
                elif action['action'] == 'call':
                    return action['action'], action['amount']
        elif hand_strength > MARGINAL_HAND and pot_odds < hand_strength:
            # Marginal hand with good pot odds - call
            for action in valid_actions:
                if action['action'] == 'call':
                    return action['action'], action['amount']
        
        # Default to fold
        for action in valid_actions:
            if action['action'] == 'fold':
                return action['action'], action['amount']
        
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