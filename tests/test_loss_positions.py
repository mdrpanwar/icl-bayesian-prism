import os
import sys
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from loss_positions import resolve_loss_range


class LossPositionTests(unittest.TestCase):
    def test_all_and_legacy_suffix(self):
        self.assertEqual(resolve_loss_range(41, "all"), (0, 41))
        self.assertEqual(resolve_loss_range(41, "10"), (31, 41))
        self.assertEqual(resolve_loss_range(11, "20"), (0, 11))

    def test_explicit_suffix_and_range(self):
        self.assertEqual(resolve_loss_range(41, loss_positions="last:10"), (31, 41))
        self.assertEqual(
            resolve_loss_range(41, loss_positions="range:25:40"),
            (25, 40),
        )

    def test_rejects_ambiguous_or_invalid_specs(self):
        with self.assertRaisesRegex(ValueError, "either"):
            resolve_loss_range(41, "10", "range:0:15")
        with self.assertRaisesRegex(ValueError, "invalid"):
            resolve_loss_range(41, loss_positions="range:40:42")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            resolve_loss_range(41, loss_positions="middle:10")


if __name__ == "__main__":
    unittest.main()
