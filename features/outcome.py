from features.extractor import FeatureExtractor
from cassis import Cas

class Outcome(FeatureExtractor):
    ANSWER_TYPE = "de.sfs.isaac.server.nlp.types.Answer"

    def extract(self, cas: Cas):
        studentView = cas.get_view(self.STUDENT_ANSWER_VIEW)
        answer = next(studentView.select(self.ANSWER_TYPE))
        return int(answer.contentScore)
