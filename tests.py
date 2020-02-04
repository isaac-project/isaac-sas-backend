import unittest
from cassis import *
from features.extractor import Outcome, PercentOfMappingType,\
    AlignmentLabel, MatchedAnnotation, FeatureExtractor, TripleOverlap, Variety

class ReadCASTestCase(unittest.TestCase):
    def test_load_typesystem(self):
        dkpro_ts = load_dkpro_core_typesystem()
        isaac_ts_file = "features/isaac-type-system.xml"
        with open(isaac_ts_file, 'rb') as f:
            typesystem = load_typesystem(f)

        final_ts = merge_typesystems(dkpro_ts,typesystem)
        
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
        
    def test_extract_outcome(self):
        dkpro_ts = load_dkpro_core_typesystem()
        isaac_ts_file = "features/isaac-type-system.xml"
        with open(isaac_ts_file, 'rb') as f:
            isaac_ts = load_typesystem(f)

        final_ts = merge_typesystems(dkpro_ts,isaac_ts)
        file_to_read = "data/xmi/1ET5_7_0.xmi"
        with open(file_to_read, 'rb') as f:
            cas = load_cas_from_xmi(f, typesystem=final_ts)
        
        o = Outcome().extract(cas)
        self.assertTrue(o)
        
    def test_extract_kw_overlap(self):
        dkpro_ts = load_dkpro_core_typesystem()
        isaac_ts_file = "features/isaac-type-system.xml"
        with open(isaac_ts_file, 'rb') as f:
            isaac_ts = load_typesystem(f)

        final_ts = merge_typesystems(dkpro_ts,isaac_ts)
        file_to_read = "data/xmi/1ET5_7_0.xmi"
        with open(file_to_read, 'rb') as f:
            cas = load_cas_from_xmi(f, typesystem=final_ts)
        
        kw = MatchedAnnotation(FeatureExtractor.TARGET_ANSWER_VIEW,
                               FeatureExtractor.KEYWORD_TYPE).extract(cas)
        self.assertTrue(kw > 0.0);
    
    def test_extract_lc_token_overlap(self):
        dkpro_ts = load_dkpro_core_typesystem()
        isaac_ts_file = "features/isaac-type-system.xml"
        with open(isaac_ts_file, 'rb') as f:
            isaac_ts = load_typesystem(f)

        final_ts = merge_typesystems(dkpro_ts,isaac_ts)
        file_to_read = "data/xmi/1ET5_7_0.xmi"
        with open(file_to_read, 'rb') as f:
            cas = load_cas_from_xmi(f, typesystem=final_ts)
        
        lc = PercentOfMappingType(AlignmentLabel.LC_TOKEN).extract(cas)
        self.assertTrue(lc > 0.0);
    
    def test_extract_student_dep_overlap(self):
        dkpro_ts = load_dkpro_core_typesystem()
        isaac_ts_file = "features/isaac-type-system.xml"
        with open(isaac_ts_file, 'rb') as f:
            isaac_ts = load_typesystem(f)

        final_ts = merge_typesystems(dkpro_ts,isaac_ts)
        file_to_read = "data/xmi/1ET5_7_0.xmi"
        with open(file_to_read, 'rb') as f:
            cas = load_cas_from_xmi(f, typesystem=final_ts)
        
        sdep = TripleOverlap(FeatureExtractor.STUDENT_ANSWER_VIEW).extract(cas)
        self.assertTrue(sdep >= 0.0);
    
    def test_extract_variety(self):
        dkpro_ts = load_dkpro_core_typesystem()
        isaac_ts_file = "features/isaac-type-system.xml"
        with open(isaac_ts_file, 'rb') as f:
            isaac_ts = load_typesystem(f)

        final_ts = merge_typesystems(dkpro_ts,isaac_ts)
        file_to_read = "data/xmi/1ET5_7_0.xmi"
        with open(file_to_read, 'rb') as f:
            cas = load_cas_from_xmi(f, typesystem=final_ts)
        
        variety = Variety().extract(cas)
        print(variety)
        self.assertTrue(variety > 0.0);
        
if __name__ == '__main__':
    unittest.main()
