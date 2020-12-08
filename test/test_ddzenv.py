import unittest

from agents import RandomAgent
from envs import DoudizhuEnv
from utils import get_downstream_player_id


class MyTestCase(unittest.TestCase):
    def test_reset_and_extract_state(self):
        env = DoudizhuEnv()
        state, _ = env.reset()
        self.assertEqual(state['obs'].size, 450)

    # def test_is_deterministic(self):
    #     self.assertTrue(is_deterministic('doudizhu'))

    def test_get_legal_actions(self):
        env = DoudizhuEnv()
        env.set_agents([RandomAgent(env.action_num) for _ in range(env.player_num)])
        env.reset()
        legal_actions = env._get_legal_actions()
        for legal_action in legal_actions:
            self.assertLessEqual(legal_action, env.action_num-1)

    def test_step(self):
        env = DoudizhuEnv()
        _, player_id = env.reset()
        player = env.game.players[player_id]
        _, next_player_id = env.step(env.action_num-1)
        self.assertEqual(next_player_id, get_downstream_player_id(
            player, env.game.players))

    def test_step_back(self):
        env = DoudizhuEnv(name='doudizhu', config={'allow_step_back':True})
        _, player_id = env.reset()
        env.step(2)
        _, back_player_id = env.step_back()
        self.assertEqual(player_id, back_player_id)
        self.assertEqual(env.step_back(), False)

        env = DoudizhuEnv()
        with self.assertRaises(Exception):
            env.step_back()

    def test_run(self):
        env = DoudizhuEnv()
        env.set_agents([RandomAgent(env.action_num) for _ in range(env.player_num)])
        trajectories, payoffs = env.run(is_training=False)
        self.assertEqual(len(trajectories), 3)
        win = []
        for player_id, payoff in enumerate(payoffs):
            if payoff == 1:
                win.append(player_id)
        if len(win) == 1:
            self.assertEqual(env.game.players[win[0]].role, 'landlord')
        if len(win) == 2:
            self.assertEqual(env.game.players[win[0]].role, 'peasant')
            self.assertEqual(env.game.players[win[1]].role, 'peasant')

    def test_decode_action(self):
        env = DoudizhuEnv()
        env.reset()
        env.game.state['actions'] = ['33366', '33355']
        env.game.judger.playable_cards[0] = ['5', '6', '55', '555', '33366', '33355']
        decoded = env._decode_action(54)
        self.assertEqual(decoded, '33366')
        env.game.state['actions'] = ['444', '44466', '44455']
        decoded = env._decode_action(29)
        self.assertEqual(decoded, '444')

    def test_get_perfect_information(self):
        env = DoudizhuEnv()
        _, player_id = env.reset()
        self.assertEqual(player_id, env.get_perfect_information()['current_player'])

if __name__ == '__main__':
    unittest.main()