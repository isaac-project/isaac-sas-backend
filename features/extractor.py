from abc import ABC,abstractmethod
from cassis import Cas
from enum import Enum, auto
from cassis.typesystem import Type
from typing import List
from features import uima
from collections import OrderedDict
import math

class AlignmentLabel(Enum):
    LC_TOKEN = auto()
    TOKEN = auto()
    SEMTYPE = auto()
    LEMMA = auto()
    SPELLING = auto()
    SYNONYM = auto()
    SIMILARITY = auto()
    REFERENCE = auto()

class FeatureExtractor(ABC):
    # some constants common to feature extractors
    STUDENT_ANSWER_VIEW = "studentAnswer"
    TARGET_ANSWER_VIEW = "_InitialView"
    QUESTION_VIEW = "question"
    ANSWER_TYPE = "de.sfs.isaac.server.nlp.types.Answer"
    MAPPABLE_TYPE = "de.sfs.isaac.server.nlp.types.MappableAnnotation"
    TOKEN_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"
    KEYWORD_TYPE = "de.sfs.isaac.server.nlp.types.Keyword"
    DEPENDENCY_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency"
    CHUNK_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.chunk.Chunk"
    
    @abstractmethod
    def extract(self, cas: Cas) -> (str,float):
        return ("", 0.0)
        
    def get_mappable_ann(self, cas: Cas, t: Type):
        return next(cas.select_covered(FeatureExtractor.MAPPABLE_TYPE, t))
    
    def get_perc_of_mapping_type(self, cas: Cas, alignment: AlignmentLabel) -> float:
        overallMatchesCount = 0
        itemsOfGivenTypeCount = 0
        for t in cas.select(FeatureExtractor.TOKEN_TYPE):

            item = self.get_mappable_ann(cas, t)

            # check for matches/alignment
            if item.match is not None and item.match.target is not None:
                overallMatchesCount += 1
                # check for types
                if item.match.label == alignment.name:
                    itemsOfGivenTypeCount += 1

        # if nothing has been matched at all, the result is 0
        if overallMatchesCount == 0:
            return 0.0;
        else:
            return itemsOfGivenTypeCount / overallMatchesCount

class Outcome(FeatureExtractor):
    LABEL2INT = {"correct" : 1, "incorrect" : 0, "true" : 1, "false" : 0}
    
    def extract(self, cas: Cas) -> (str,float):
        studentView = cas.get_view(FeatureExtractor.STUDENT_ANSWER_VIEW)
        answer = next(studentView.select(FeatureExtractor.ANSWER_TYPE))
        score = answer.contentScore
        
        if score in Outcome.LABEL2INT:
            return ("Outcome", Outcome.LABEL2INT[score])
        else:
            try:
                outcome = float(score)
            except ValueError:
                outcome = math.nan
            return ("Outcome", outcome)
            #todo:changed this to float()   File "/home/akrnshva/git/isaac-ml-service/features/extractor.py", line 69, in extract
            # return ("Outcome", int(answer.contentScore))
            # ValueError: invalid literal for int() with base 10: '-98.0'

class Diagnosis(FeatureExtractor):
    
    def extract(self, cas:Cas)->(str,float):
        studentView = cas.get_view(FeatureExtractor.STUDENT_ANSWER_VIEW)
        answer = next(studentView.select(FeatureExtractor.ANSWER_TYPE))
        return ("Diagnosis", int(answer.contentDiagnosis))

class PercentOfMappingType(FeatureExtractor):
    def __init__(self, alignment: AlignmentLabel):
        self.alignment = alignment
        
    def extract(self, cas:Cas)->(str,float):
        studentView = cas.get_view(FeatureExtractor.STUDENT_ANSWER_VIEW)
        return (self.alignment.name + "-Match", self.get_perc_of_mapping_type(studentView, self.alignment))
    
class MatchedAnnotation(FeatureExtractor):
    def __init__(self, view_name: str, ann_type: str):
        self.view_name = view_name
        self.ann_type = ann_type
    
    def extract(self, cas:Cas)->(str,float):
        view = cas.get_view(self.view_name)
        matched_ann = 0
        all_ann = 0
        for a in view.select(self.ann_type):
            all_ann += 1
            mappable = self.get_mappable_ann(view, a)
        
            if mappable.match:
                matched_ann += 1
    
        ret = -1.0
        if not all_ann:
            ret = 0.0
        else:
            ret = matched_ann / all_ann
        
        return (self.view_name + "-" + uima.simple_type_name(self.ann_type) + "-Overlap", ret)

class TripleOverlap(FeatureExtractor):
    DEP_MAPPING_TYPE = "de.sfs.isaac.server.nlp.types.DepMapping"
    
    def __init__(self, view_name: str):
        self.view_name = view_name
        self.english_arg_rels = set(
            ["dep",
            "arg",
            "subj",
            "nsubj",
            "nsubjpass",
            "csubj",
            "comp",
            "obj",
            "dobj",
            "iobj",
            "pobj",
            "attr",
            "ccomp",
            "xcomp",
            "compl",
            "mark",
            "rel",
            "acomp",
            "agent"])
    
    def extract(self, cas:Cas)->(str,float):
        view = cas.get_view(self.view_name)
        dep_matches = len(list(view.select(TripleOverlap.DEP_MAPPING_TYPE)))
        dep_rels = 0
        
        for d in view.select(FeatureExtractor.DEPENDENCY_TYPE):
            if d.DependencyType in self.english_arg_rels:
                dep_rels += 1
        
        ret = -1.0
        if not dep_rels:
            ret = 0.0
        else:
            ret = dep_matches / dep_rels
        
        return (self.view_name + "-Triple-Overlap", ret)
    
class Variety(FeatureExtractor):
    
    def extract(self, cas:Cas)->(str,float):
        studentView = cas.get_view(FeatureExtractor.STUDENT_ANSWER_VIEW)
        
        variety = 0.0
        for al in AlignmentLabel:
            variety += self.get_perc_of_mapping_type(studentView, al)
        
        return ("Variety", variety);

class FeatureExtraction():
    
    def __init__(self, extractors: List[FeatureExtractor] = None):
        if extractors:
            self.extractors = extractors
        else:
            self.extractors = [
                MatchedAnnotation(FeatureExtractor.TARGET_ANSWER_VIEW, FeatureExtractor.KEYWORD_TYPE),
                MatchedAnnotation(FeatureExtractor.TARGET_ANSWER_VIEW, FeatureExtractor.TOKEN_TYPE),
                MatchedAnnotation(FeatureExtractor.STUDENT_ANSWER_VIEW, FeatureExtractor.TOKEN_TYPE),
                MatchedAnnotation(FeatureExtractor.TARGET_ANSWER_VIEW, FeatureExtractor.CHUNK_TYPE),
                MatchedAnnotation(FeatureExtractor.STUDENT_ANSWER_VIEW, FeatureExtractor.CHUNK_TYPE),
                TripleOverlap(FeatureExtractor.TARGET_ANSWER_VIEW),
                TripleOverlap(FeatureExtractor.STUDENT_ANSWER_VIEW),
                PercentOfMappingType(AlignmentLabel.LC_TOKEN),
                PercentOfMappingType(AlignmentLabel.LEMMA),
                PercentOfMappingType(AlignmentLabel.SYNONYM),
                Variety(),
                Outcome()
            ]
    
    def from_cases(self, cases: List[Cas]) -> OrderedDict:
        ret = OrderedDict()
        
        for cas in cases:
            for x in self.extractors:
                f = x.extract(cas)
                ret.setdefault(f[0], []).append(f[1])
                
        return ret
                
                
