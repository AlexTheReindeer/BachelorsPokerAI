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
    def __init__(self, ppo_agent):
        super().__init__()
        self.ppo_agent = ppo_agent
        self.last_obs = None
        self.game_state = {
            'round_state': {},
            'hole_card': [],
            'community_card': [],
            'hand_over': False
        }
        self.current_reward = 0

    # Required callbacks:
    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.game_state['hole_card'] = hole_card
        self.game_state['round_state']['seats'] = seats

    def receive_street_start_message(self, street, round_state):
        """Called when a new street starts"""
        self.game_state['round_state'].update(round_state)
        if 'community_cards' in round_state:
            self.game_state['community_card'] = round_state['community_cards']
            
        # Print street information
        print("\n=== New Street ===")
        print(f"Street: {street}")
        if round_state.get('community_cards'):
            print(f"Community cards: {', '.join(round_state['community_cards'])}")
        print("==================\n")

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
        print("\n=== Action Update ===")
        print(f"Street: {round_state.get('street', 'unknown')}")
        if round_state.get('community_cards'):
            print(f"Community cards: {', '.join(round_state['community_cards'])}")
        player_name = new_action.get('uuid', 'unknown')
        print(f"Player: {player_name}")
        print(f"Action: {new_action.get('action', 'unknown')}")
        if 'amount' in new_action:
            print(f"Amount: {new_action['amount']}")
        print("===================\n")

    def receive_round_result_message(self, winners, hand_info, round_state):
        """Called at the end of each hand"""
        print("\n=== Round Result ===")
        print(f"Street: {round_state.get('street', 'showdown')}")
        if round_state.get('community_cards'):
            print(f"Community cards: {', '.join(round_state['community_cards'])}")
        print("Winners:", [w.get('name', w.get('uuid', 'unknown')) for w in winners])
        
        # Calculate final reward based on whether we won or lost
        is_winner = any(w.get('uuid') == self.uuid for w in winners)
        pot_size = round_state['pot']['main']['amount']
        our_contribution = pot_size / 2  # In heads-up, we contribute half the pot
        
        # If we won, we get the pot. If we lost, we lose our contribution to the pot
        final_reward = pot_size if is_winner else -our_contribution  # Lose our entire contribution when we lose
        
        print(f"Winner: {'Prodigy' if is_winner else 'Trainer'}")
        print(f"Final reward: {final_reward}")
        print("===================\n")
        
        # Store final reward
        self.current_reward = final_reward
        self.ppo_agent.store_reward(final_reward, True)
        self.last_obs = None
        self.game_state['hand_over'] = True

    # Actual action logic
    def declare_action(self, valid_actions, hole_card, round_state):
        """
        1) Build observation
        2) Use the action passed from the environment
        3) Return the PyPokerEngine action + bet amount
        """
        # Get community cards safely
        community_cards = []
        if isinstance(round_state, dict):
            if 'street' in round_state:
                street = round_state['street']
                if street == 1:  # Flop
                    community_cards = round_state.get('community_cards', [])[:3]
                elif street == 2:  # Turn
                    community_cards = round_state.get('community_cards', [])[:4]
                elif street == 3:  # River
                    community_cards = round_state.get('community_cards', [])[:5]
            else:
                # If no street info, just get all community cards
                community_cards = round_state.get('community_cards', [])
        
        # Get the action from the environment
        act_id = self.ppo_agent.last_action if hasattr(self.ppo_agent, 'last_action') else None
        
        if act_id is None:
            # If no action was passed, make our own decision
            obs = encode_observation(hole_card, community_cards, round_state)
            act_id = self.ppo_agent.select_action(obs)
            self.ppo_agent.store_reward(0.0, False)

        # Convert act_id => fold/call/raise
        if act_id == 0:
            # Find fold action
            for va in valid_actions:
                if va['action'] == 'fold':
                    return 'fold', va['amount']
            return 'fold', 0
        elif act_id == 1:
            # Find call action
            for va in valid_actions:
                if va['action'] == 'call':
                    return 'call', va['amount']
            return 'fold', 0
        else:
            # Find raise action
            for va in valid_actions:
                if va['action'] == 'raise':
                    # Get min and max raise amounts
                    min_amount = va['amount']['min']
                    max_amount = va['amount']['max']
                    
                    # Ensure min_amount is at least the big blind
                    big_blind = round_state.get('small_blind_amount', 10) * 2
                    min_amount = max(min_amount, big_blind)
                    
                    # Ensure max_amount doesn't exceed player's stack
                    player_stack = round_state.get('seats', [{}])[0].get('stack', 1000)
                    max_amount = min(max_amount, player_stack)
                    
                    # If max_amount is less than min_amount, use min_amount
                    if max_amount < min_amount:
                        raise_amount = min_amount
                    else:
                        # Calculate raise amount between min and max
                        raise_amount = random.randint(min_amount, max_amount)
                    
                    return 'raise', raise_amount
            return 'fold', 0

    def get_observation(self):
        """Get the current observation from the game state"""
        # Get current game state
        round_state = self.game_state.get('round_state', {})
        hole_card = self.game_state.get('hole_card', [])
        community_card = self.game_state.get('community_card', [])
        
        # Encode observation
        return encode_observation(hole_card, community_card, round_state)
        
    def get_reward(self):
        """Get the current reward"""
        return self.current_reward
        
    def is_hand_over(self):
        """Check if the current hand is over"""
        return self.game_state.get('hand_over', False)
