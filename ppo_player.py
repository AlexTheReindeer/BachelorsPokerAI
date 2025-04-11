import numpy as np
from pypokerengine.players import BasePokerPlayer
from ppo_agent import PPOAgent
import random

def card_rank_to_int(rank):
    """Convert card rank to integer value (1-13)"""
    if rank == 'A':
        return 1
    elif rank == 'K':
        return 13
    elif rank == 'Q':
        return 12
    elif rank == 'J':
        return 11
    elif rank == 'T':
        return 10
    else:
        return int(rank)

def encode_observation(hole_card, community_card, round_state):
    """
    Encode the poker game state into a 10-dimensional observation vector:
    - 2 values for hole cards (rank and suit)
    - 5 values for community cards (rank and suit)
    - 1 value for pot size
    - 1 value for current bet
    - 1 value for player stack
    """
    obs = np.zeros(10, dtype=np.float32)
    
    # Encode hole cards (first 2 values)
    if hole_card:
        for i, card in enumerate(hole_card):
            suit = card[0]  # First character is suit (H, D, C, S)
            rank = card[1:]  # Rest is rank (1-13)
            obs[i] = (card_rank_to_int(rank) - 1) / 13.0  # Normalize rank to [0,1]
            obs[i+1] = (ord(suit) - ord('S')) / 4.0  # Normalize suit to [0,1]
    
    # Encode community cards (next 5 values)
    if community_card:
        for i, card in enumerate(community_card):
            suit = card[0]  # First character is suit (H, D, C, S)
            rank = card[1:]  # Rest is rank (1-13)
            obs[i+2] = (card_rank_to_int(rank) - 1) / 13.0  # Normalize rank to [0,1]
            obs[i+3] = (ord(suit) - ord('S')) / 4.0  # Normalize suit to [0,1]
    
    # Encode pot size (8th value)
    pot_size = 0
    if 'seats' in round_state:
        for seat in round_state['seats']:
            pot_size += seat.get('bet', 0)
    obs[7] = pot_size / 1000.0  # Normalize pot size
    
    # Encode current bet (9th value) - using the current bet from the round state
    current_bet = 0
    if 'seats' in round_state:
        for seat in round_state['seats']:
            if 'bet' in seat:
                current_bet = max(current_bet, seat['bet'])
    obs[8] = current_bet / 1000.0  # Normalize current bet
    
    # Encode player stack (10th value)
    if 'seats' in round_state:
        for seat in round_state['seats']:
            if 'stack' in seat:
                obs[9] = seat['stack'] / 1000.0  # Normalize stack
                break
    
    return obs

class PPOPlayer(BasePokerPlayer):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.uuid = None
        self.name = "PPO Player"
        self.total_reward = 0
        self.wins = 0
        self.total_hands = 0
        self.game_state = {
            'round_state': {
                'seats': [],
                'community_card': [],
                'pot': {'main': {'amount': 0}},
                'dealer_pos': 0,
                'street': 'preflop'
            },
            'hole_card': [],
            'community_card': [],
            'hand_over': False
        }
        self.current_reward = 0
        self.last_obs = None

    # Required callbacks:
    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        """Called at the start of each round"""
        # Update game state
        self.game_state['round_state']['seats'] = seats
        self.game_state['hole_card'] = hole_card
        self.game_state['community_card'] = []
        self.game_state['hand_over'] = False
        self.current_reward = 0

    def receive_street_start_message(self, street, round_state):
        """Called when a new street starts"""
        self.game_state['round_state'].update(round_state)
        if 'community_cards' in round_state:
            self.game_state['community_card'] = round_state['community_cards']
            
        # Print street information
        # print("\n=== New Street ===")
        # print(f"Street: {street}")
        # if round_state.get('community_cards'):
        #     print(f"Community cards: {', '.join(round_state['community_cards'])}")
        # print("==================\n")

    def receive_game_update(self, action, round_state):
        """
        REMOVE partial reward here. 
        We'll only store partial reward in declare_action to keep states and rewards in sync.
        """
        pass

    def receive_game_update_message(self, new_action, round_state):
        """Called after each action"""
        self.game_state['round_state'].update(round_state)
        
        # Print action information
        # print("\n=== Action Update ===")
        # print(f"Street: {round_state.get('street', 'unknown')}")
        # if round_state.get('community_cards'):
        #     print(f"Community cards: {', '.join(round_state['community_cards'])}")
        # player_name = new_action.get('uuid', 'unknown')
        # print(f"Player: {player_name}")
        # print(f"Action: {new_action.get('action', 'unknown')}")
        # if 'amount' in new_action:
        #     print(f"Amount: {new_action['amount']}")
        # print("===================\n")

    def receive_round_result_message(self, winners, hand_info, round_state):
        """Called at the end of each hand"""
        # Calculate final reward based on whether we won or lost
        is_winner = any(w.get('uuid') == self.uuid for w in winners)
        pot_size = round_state['pot']['main']['amount']
        our_contribution = pot_size / 2  # In heads-up, we contribute half the pot
        
        # Calculate reward based only on chips won/lost
        if is_winner:
            # If we won, get the pot minus our contribution
            final_reward = pot_size - our_contribution
        else:
            # If we lost, lose our contribution
            final_reward = -our_contribution
        
        # Store final reward
        self.current_reward = final_reward
        self.agent.store_reward(final_reward, True, {'hand_result': 'win' if is_winner else 'loss'})
        self.last_obs = None
        self.game_state['hand_over'] = True

    # Actual action logic
    def declare_action(self, valid_actions, hole_card, round_state):
        obs = self.get_observation(hole_card, round_state)
        action_tuple = self.agent.select_action(obs)
        action = action_tuple[0]  # Get the action index
        
        # Map action index to poker action
        if action == 0:  # Fold
            return valid_actions[0]['action'], valid_actions[0]['amount']
        elif action == 1:  # Call
            return valid_actions[1]['action'], valid_actions[1]['amount']
        else:  # Raise with different amounts
            pot_size = round_state['pot'].get('main', {}).get('amount', 0)
            min_raise = valid_actions[2]['amount']['min']
            max_raise = valid_actions[2]['amount']['max']
            
            if action == 2:  # Min raise
                raise_amount = min_raise
            elif action == 3:  # Small raise (1/4 pot)
                raise_amount = min(max(min_raise, pot_size // 4), max_raise)
            elif action == 4:  # Medium raise (1/2 pot)
                raise_amount = min(max(min_raise, pot_size // 2), max_raise)
            elif action == 5:  # Large raise (3/4 pot)
                raise_amount = min(max(min_raise, 3 * pot_size // 4), max_raise)
            else:  # Full pot raise
                raise_amount = min(max(min_raise, pot_size), max_raise)
                
            return valid_actions[2]['action'], raise_amount

    def get_observation(self, hole_card=None, round_state=None):
        """Get the current observation with enhanced encoding"""
        # Use provided values or fall back to game state
        if hole_card is None:
            hole_card = self.game_state.get('hole_card', [])
        if round_state is None:
            round_state = self.game_state.get('round_state', {})
            
        obs = np.zeros(117)  # Updated dimension
        
        try:
            # Encode hole cards (52 dimensions, one-hot)
            for card in hole_card:
                rank_idx = self._get_rank_index(card[0])
                suit_idx = self._get_suit_index(card[1])
                card_idx = rank_idx * 4 + suit_idx
                obs[card_idx] = 1
                
            # Encode community cards (52 dimensions, one-hot)
            community_cards = round_state.get('community_card', [])
            for card in community_cards:
                rank_idx = self._get_rank_index(card[0])
                suit_idx = self._get_suit_index(card[1])
                card_idx = 52 + rank_idx * 4 + suit_idx
                obs[card_idx] = 1
                
            # Normalize pot size and current bet (2 dimensions)
            pot = round_state.get('pot', {})
            if isinstance(pot, dict):
                pot_size = pot.get('main', {}).get('amount', 0)
            else:
                pot_size = 0
            initial_stack = 1000  # Assuming initial stack of 1000
            obs[104] = pot_size / (2 * initial_stack)  # Normalize by max possible pot
            
            current_bet = round_state.get('current_bet', 0)
            obs[105] = current_bet / initial_stack  # Normalize by initial stack
            
            # Encode position relative to dealer and current street (6 dimensions)
            dealer_pos = round_state.get('dealer_btn', 0)
            my_pos = round_state.get('next_player', 0)
            relative_pos = (my_pos - dealer_pos) % 6
            obs[106 + relative_pos] = 1
            
            # Encode current street (4 dimensions)
            street_idx = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
            current_street = round_state.get('street', 'preflop')
            obs[112 + street_idx[current_street]] = 1
            
            # Add opponent modeling features
            seats = round_state.get('seats', [])
            for i, seat in enumerate(seats):
                if seat.get('uuid') != self.uuid:
                    stack = seat.get('stack', 0)
                    obs[116] = stack / initial_stack  # Normalize opponent stack
                    
        except Exception as e:
            print(f"Error in get_observation: {e}")
            # Return zero observation in case of error
            return np.zeros(117)
            
        return obs

    def get_reward(self):
        """Get the current reward"""
        return self.current_reward
        
    def is_hand_over(self):
        """Check if the current hand is over"""
        return self.game_state.get('hand_over', False)

    def _get_suit_index(self, suit):
        """Convert card suit to index (h=0, d=1, c=2, s=3)"""
        try:
            suit_map = {'h': 0, 'd': 1, 'c': 2, 's': 3}
            return suit_map.get(suit.lower(), 0)  # Default to 0 if suit not found
        except Exception as e:
            print(f"Error parsing suit {suit}: {e}")
            return 0  # Return 0 as default index

    def _get_rank_index(self, rank):
        """Convert card rank to index (A=12, K=11, Q=10, J=9, T=8, 9-2=7-0)"""
        try:
            # Handle case where full card string is passed
            if len(rank) > 1:
                rank = rank[1:]  # Take everything after the suit
                
            rank_map = {
                'A': 12, 'K': 11, 'Q': 10, 'J': 9, 'T': 8,
                '9': 7, '8': 6, '7': 5, '6': 4, '5': 3, '4': 2, '3': 1, '2': 0
            }
            return rank_map.get(rank, 0)  # Default to 0 if rank not found
        except Exception as e:
            print(f"Error parsing rank {rank}: {e}")
            return 0  # Return 0 as default index
