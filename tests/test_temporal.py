import unittest
from media_auth_forensics.infer.temporal import kalman_binary_pattern

class TestTemporal(unittest.TestCase):
    def test_pattern_length(self):
        scores = [0.1, 0.2, 0.8, 0.9]
        pattern = kalman_binary_pattern(scores, threshold=0.5)
        self.assertEqual(len(pattern), len(scores))

if __name__ == "__main__":
    unittest.main()
