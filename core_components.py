import nltk

class SentenceProcessor:
    def __init__(self):
        nltk.download('punkt')  # Ensure that the sentence tokenizer is downloaded

    def process_sentence(self, text):
        """
        Split text into sentences using NLTK's sentence tokenizer.
        """
        sentences = nltk.sent_tokenize(text)
        return [sent.strip() for sent in sentences if sent.strip()]
    
class AutomotivePartFinder:
    def find_automotive_part(self, sentence_tokens):
        """
        Detect automotive parts in the sentence.
        """
        automotive_part = None
        for token in sentence_tokens:
            if token['label'] == 'AUTOMOTIVE_PART':
                automotive_part = token['word']
                break

        return automotive_part