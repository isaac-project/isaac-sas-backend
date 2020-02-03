import unittest
from features import * 
from cassis import *

class ReadCASTestCase(unittest.TestCase):
    def test_read(self):
        dkpro_ts = load_dkpro_core_typesystem()
        isaac_ts_file = "features/isaac-type-system.xml"
        with open(isaac_ts_file, 'rb') as f:
            typesystem = load_typesystem(f, dkpro_ts)

        file_to_read = "data/xmi/1ET5_1_25.xmi"
        with open(file_to_read, 'rb') as f:
            cas = load_cas_from_xmi(f, typesystem=typesystem)
        self.assertIsNotNone(cas)

if __name__ == '__main__':
    unittest.main()
