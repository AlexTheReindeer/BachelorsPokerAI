import random
from pypokerengine.players import BasePokerPlayer

class RandomPlayer(BasePokerPlayer):
    def __init__(self):
        super().__init__()

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update(self, action, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        pass

    def receive_round_result(self, winners, hand_info, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        """
        Also required. We'll just no-op here.
        """
        pass

    def declare_action(self, valid_actions, hole_card, round_state):
        va = random.choice(valid_actions)
        if va['action'] == 'raise':
            return 'raise', va['amount']['min']
        elif va['action'] == 'call':
            return 'call', va['amount']
        else:
            return va['action'], va['amount']
