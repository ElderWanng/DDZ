import unittest

from games.doudizhu.utils import CARD_TYPE
from games.doudizhu.judger import DoudizhuJudger as Judger


class TestDoudizhuGame(unittest.TestCase):

    def test_playable_cards_from_hand(self):

        playable_cards = list(Judger.playable_cards_from_hand('3333444455556666777788889999TTTTJJJJQQQQKKKKAAAA2222BR'))
        all_cards_list = CARD_TYPE[1]
        for c in playable_cards:
            # if (c not in all_cards_list):
            #     print(c)
            self.assertIn(c, all_cards_list)
        for c in all_cards_list:
            # if (c not in playable_cards):
            #     print('\t' + c)
            self.assertIn(c, playable_cards)
        self.assertEqual(len(playable_cards), len(all_cards_list))


if __name__ == '__main__':
    unittest.main()
