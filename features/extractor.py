from abc import ABC,abstractmethod
from cassis import Cas

class FeatureExtractor(ABC):
    STUDENT_ANSWER_VIEW = "studentAnswer"
    TARGET_ANSWER_VIEW = "_initialView"
    QUESTION_VIEW = "question"

    @abstractmethod
    def extract(self, cas: Cas) -> float:
        pass