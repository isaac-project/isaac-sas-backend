from abc import ABC,abstractmethod
from cassis import Cas
from enum import Enum, auto

class AlignmentLabel(Enum):
    LC_TOKEN = auto()
    TOKEN = auto()
    DEPTRIPLE = auto()
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
    
    @abstractmethod
    def extract(self, cas: Cas) -> float:
        pass
    
    def getPercOfMappingType(self, cas: Cas, alignment: AlignmentLabel) -> float:
        overallMatchesCount = 0
        itemsOfGivenTypeCount = 0
        for t in cas.select(self.TOKEN_TYPE):

            item = next(cas.select_covered(self.MAPPABLE_TYPE, t));

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
    
    def extract(self, cas: Cas) -> float:
        studentView = cas.get_view(self.STUDENT_ANSWER_VIEW)
        answer = next(studentView.select(self.ANSWER_TYPE))
        score = answer.contentScore
        
        if score in self.LABEL2INT:
            return self.LABEL2INT[score]
        else:
            return int(answer.contentScore)

class Diagnosis(FeatureExtractor):
    
    def extract(self, cas:Cas)->float:
        studentView = cas.get_view(self.STUDENT_ANSWER_VIEW)
        answer = next(studentView.select(self.ANSWER_TYPE))
        return int(answer.contentDiagnosis)

class KeywordOverlap(FeatureExtractor):
    KEYWORD_TYPE = "de.sfs.isaac.server.nlp.types.Keyword"
    
    def extract(self, cas:Cas)->float:
        targetView = cas.get_view(self.TARGET_ANSWER_VIEW)
        cas.select
        matchedKeywords = 0
        allKeywords = 0
        for k in targetView.select(self.KEYWORD_TYPE):
            allKeywords += 1
            mappable = next(targetView.select_covered(self.MAPPABLE_TYPE, k))
            if mappable.match:
                matchedKeywords += 1
        
        if not allKeywords:
            return 0.0
        else:
            return matchedKeywords / allKeywords
    
class LC_TokenMatch(FeatureExtractor):
    
    def extract(self, cas:Cas)->float:
        studentView = cas.get_view(self.STUDENT_ANSWER_VIEW)
        return self.getPercOfMappingType(studentView, AlignmentLabel.LC_TOKEN)
