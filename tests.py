import unittest

from features.extractor import FeatureExtraction
from features import uima
from cassis.xmi import load_cas_from_xmi

class ReadCASTestCase(unittest.TestCase):
    EXAMPLE_XMI_PATH = "testdata/xmi/1ET5_7_0.xmi"

    def test_full_extraction(self):
        final_ts = uima.load_isaac_ts()
        file_to_read = ReadCASTestCase.EXAMPLE_XMI_PATH
        with open(file_to_read, 'rb') as f:
            cas = load_cas_from_xmi(f, typesystem=final_ts)
        
        extraction = FeatureExtraction()
        feats = extraction.run(cas)
        
        self.assertIsInstance(feats, list)
        self.assertEqual(len(extraction.extractors),
                         len(feats))
        
if __name__ == '__main__':
    unittest.main()
