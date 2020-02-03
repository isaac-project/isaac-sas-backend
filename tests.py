import unittest
from features import * 
from cassis import *

class ReadCASTestCase(unittest.TestCase):
    def test_load_typesystem(self):
        dkpro_ts = load_dkpro_core_typesystem()
        isaac_ts_file = "features/isaac-type-system.xml"
        with open(isaac_ts_file, 'rb') as f:
            typesystem = load_typesystem(f)

        final_ts = merge_typesystems(dkpro_ts,typesystem)
        
        for t in final_ts.get_types():
            print(t.name)
            for feat in t.features:
                print(feat.name)
        
        self.assertIsInstance(final_ts, TypeSystem)

    def test_read_cas(self):
        dkpro_ts = load_dkpro_core_typesystem()
        isaac_ts_file = "features/isaac-type-system.xml"
        with open(isaac_ts_file, 'rb') as f:
            isaac_ts = load_typesystem(f)

        final_ts = merge_typesystems(dkpro_ts,isaac_ts)
        file_to_read = "data/xmi/1ET5_7_0.xmi"
        with open(file_to_read, 'rb') as f:
            cas = load_cas_from_xmi(f, typesystem=final_ts)
        self.assertIsInstance(cas, Cas)
        
if __name__ == '__main__':
    unittest.main()
