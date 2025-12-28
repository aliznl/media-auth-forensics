import unittest
from PIL import Image
import numpy as np
from media_auth_forensics.infer.region_scan import irregularity_scan

class TestRegionScan(unittest.TestCase):
    def test_scan_outputs(self):
        img = Image.fromarray((np.random.rand(256,256,3)*255).astype("uint8"))
        out = irregularity_scan(img)
        self.assertIn("image_score", out)
        self.assertIn("binary_mask", out)

if __name__ == "__main__":
    unittest.main()
