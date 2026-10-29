# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Synthetic metric fixtures; never benchmark evidence.'
import unittest
from control.signal import metrics
class SignalTests(unittest.TestCase):
    def test_equal_steps_do_not_imply_equal_reward_variation_or_native_signal(self):
        report = metrics([1., 1., 0., 1.], ['a', 'a', 'b', 'b'],
                         [[1., 0.], [0., 2.], [-1., 2.], [3., 0.]],
                         [[1, 0], [0, 0], [1, 1], [1, 0]])
        self.assertEqual(report['signal/batch_reward_groups'], 2)
        self.assertEqual(report['signal/groups_with_reward_variation'], 1)
        self.assertEqual(report['signal/native_optimized_tokens'], 4)
        self.assertEqual(report['signal/active_advantage_tokens'], 4)
        self.assertEqual(report['signal/fully_masked_trajectories'], 1)
    def test_episode_variation_is_not_needed_for_step_signal(self):
        report = metrics([1, 1], ['a', 'a'], [[1], [-1]], [[1], [1]])
        self.assertEqual(report['signal/groups_with_reward_variation'], 0)
        self.assertEqual(report['signal/active_advantage_tokens'], 2)
    def test_no_active_tokens_is_reported_not_fabricated_signal(self):
        report = metrics([1], ['a'], [[9]], [[0]])
        self.assertEqual(report['signal/active_advantage_fraction'], 0)
    def test_shaped_reward_variation_is_not_answer_improvement(self):
        report = metrics([1., 1.1, 0., 1.], ['a', 'a', 'b', 'b'], [[1]]*4, [[1]]*4,
                         answer_correctness=[1, 1, 0, 1])
        self.assertEqual(report['signal/groups_with_reward_variation'], 2)
        self.assertEqual(report['signal/groups_with_correctness_variation'], 1)
        with self.assertRaises(ValueError):
            metrics([1], ['a'], [[1]], [[1]], answer_correctness=[float('nan')])
    def test_unknown_correctness_is_not_filled_as_wrong(self):
        report = metrics([1, 1], ['a', 'a'], [[1], [1]], [[1], [1]], answer_correctness=[1, None])
        self.assertEqual(report['signal/groups_with_known_correctness'], 0)
    def test_invalid_alignment_and_values_rejected(self):
        for values in (([], [], [], []), ([1], ['a'], [[1]], [[1, 0]]),
                       ([float('nan')], ['a'], [[1]], [[1]]),
                       ([1], ['a'], [[float('inf')]], [[1]]), ([1], ['a'], [[1]], [[.5]])):
            with self.subTest(values=values), self.assertRaises(ValueError):
                metrics(*values)
if __name__ == '__main__':
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
