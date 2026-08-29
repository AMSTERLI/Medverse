import math
import unittest

import torch

from medverse.models.nn.ccti import ChannelSelectiveContextTargetInteraction


class CCTITest(unittest.TestCase):
    def test_learned_topk_preserves_shapes_and_gradients(self):
        target = torch.randn(2, 16, 4, 4, 4, requires_grad=True)
        context = torch.randn(2, 3, 16, 4, 4, 4, requires_grad=True)
        block = ChannelSelectiveContextTargetInteraction(16, channel_ratio=0.25)
        target_out, context_out = block(target, context, torch.tensor([0.8, 0.5]))
        self.assertEqual(target_out.shape, target.shape)
        self.assertEqual(context_out.shape, context.shape)
        self.assertEqual(block.last_routing["indices"].shape[1], math.ceil(16 * 0.25))
        (target_out.mean() + context_out.mean()).backward()
        self.assertIsNotNone(target.grad)
        self.assertIsNotNone(context.grad)

    def test_none_is_exact_bypass(self):
        target = torch.randn(1, 8, 3, 3, 3)
        context = torch.randn(1, 1, 8, 3, 3, 3)
        block = ChannelSelectiveContextTargetInteraction(8, mode="none")
        target_out, context_out = block(target, context)
        self.assertIs(target_out, target)
        self.assertIs(context_out, context)

    def test_all_and_random_select_expected_budget(self):
        target = torch.randn(1, 10, 3, 3, 3)
        context = torch.randn(1, 2, 10, 3, 3, 3)
        for mode, expected in (("all", 10), ("random", 3)):
            block = ChannelSelectiveContextTargetInteraction(10, channel_ratio=0.3, mode=mode)
            block(target, context)
            self.assertEqual(block.last_routing["indices"].shape[1], expected)


if __name__ == "__main__":
    unittest.main()
