import os
from typing import Dict, Set

from dotenv import load_dotenv

try:
    import spacy
except ImportError:
    spacy = None

try:
    import openai
except ImportError:
    openai = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

# Load environment variables
load_dotenv()


class ContentAnalyzer:
    def __init__(self, methods: Set[str] = {'nlp'}):
        """Initialize the ContentAnalyzer with specified methods.
        
        Args:
            methods: Set of methods to initialize ('nlp', 'openai', 'gemmi')
        """
        self.available_methods = set()  # Start with empty set

        # Check for required packages before initialization
        if 'nlp' in methods:
            if spacy is None:
                raise ImportError("spacy is not installed, try: pip install spacy")
            try:
                self.nlp = spacy.load("en_core_web_md")
                self.available_methods.add('nlp')
            except Exception as e:
                print(f"Error loading spaCy model: {e}")
                self.nlp = None

        if 'openai' in methods:
            if openai is None:
                raise ImportError("openai is not installed, try: pip install openai")
            try:
                self.openai_client = openai.OpenAI(
                    api_key=os.getenv('OPENAI_API_KEY')
                )
                self.available_methods.add('openai')
            except Exception as e:
                print(f"Error initializing OpenAI client: {e}")
                self.openai_client = None

        if 'gemmi' in methods:
            if genai is None or types is None:
                raise ImportError("google.generativeai is not installed, try: pip install google-generativeai")
            try:
                genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
                self.gemmi_client = genai.GenerativeModel(
                    'gemini-1.5-flash',
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=2000,
                        temperature=0.1,
                    ))
                self.available_methods.add('gemmi')
            except Exception as e:
                print(f"Error initializing Gemmi client: {e}")
                self.gemmi_client = None

        # Task-specific context
        self.task_context = {
            'document_text': """
            """,
            'key_phrases': [
                "system requirements",
                "user interface", 
                "data processing",
                "performance metrics",
                # Add more task-specific multi-word phrases
            ],
            'keywords': [
                "database", "frontend", "backend", "API",
                "testing", "deployment", "documentation", 
                # Add more individual task-specific keywords
            ]
        }

        # Construct-specific context
        self.construct_context = {
            'description': """
            """,
            'key_phrases': [
                "shared understanding",
                "team coordination",
                "knowledge transfer",
                "collaborative problem solving",
                # Add more construct-specific multi-word phrases
            ],
            'keywords': [
                "collaboration", "coordination", "communication",
                "agreement", "discussion", "understanding",
                # Add more individual construct-specific keywords
            ]
        }

    def analyze_with_nlp(self, text: str, threshold: float = 0.45) -> int:
        """
        Analyze text relevance using NLP (spaCy)
        Returns:
            0: Not relevant
            1: Task-relevant
            2: Construct-relevant
        """
        if self.nlp is None:
            raise ValueError("NLP method not initialized")
            
        # Convert input text to spaCy doc
        doc = self.nlp(text.lower())
        
        # Calculate similarities
        task_similarity = self._calculate_context_similarity(doc, self.task_context)
        construct_similarity = self._calculate_context_similarity(doc, self.construct_context)
        print(task_similarity, construct_similarity)    

        if task_similarity >= construct_similarity and task_similarity >= threshold:
            return 1
        elif construct_similarity >= task_similarity and construct_similarity >= threshold:
            return 2
        return 0

    def _calculate_context_similarity(self, doc, context: Dict) -> float:
        """Calculate similarity between text and a context"""
        similarities = []
        
        # Compare with document text
        if 'document_text' in context:
            doc_similarity = doc.similarity(self.nlp(context['document_text'])) 
            similarities.append(doc_similarity)
        
        # Compare with key phrases
        for phrase in context.get('key_phrases', []):
            phrase_doc = self.nlp(phrase)
            similarity = doc.similarity(phrase_doc)
            similarities.append(similarity)
            
        # Compare with individual keywords
        if context.get('keywords'):
            keywords_text = " ".join(context['keywords'])
            keywords_doc = self.nlp(keywords_text)
            similarity = doc.similarity(keywords_doc)
            similarities.append(similarity)
        
        average_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        return average_similarity

    def analyze_with_openai(self, text: str) -> bool:
        """Analyze using OpenAI's language model."""
        if self.openai_client is None:
            raise ValueError("OpenAI method not initialized")

        prompt = f"""
        Analyze if the following text is relevant to a professional discussion.
        Consider topics like project updates, technical discussions, or team coordination.
        Respond with only 'true' if relevant or 'false' if not relevant.
        
        Text: "{text}"
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes text relevance."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            return response.choices[0].message.content.strip().lower() == 'true'
        except Exception as e:
            print(f"Error in LLM analysis: {e}")
            return False

    def analyze_with_gemmi(self, text: str) -> bool:
        """Analyze using google's language model."""
        if self.gemmi_client is None:
            raise ValueError("Gemmi method not initialized")

        prompt = f"""
        Analyze if the following text is relevant to a professional discussion.
        Consider topics like project updates, technical discussions, or team coordination.
        Respond with only 'true' if relevant or 'false' if not relevant.

        Text: "{text}"
        """
        self.gemmi_client.generate_content(prompt)
        return True


# Example usage
if __name__ == "__main__":
    # Test data
    test_transcription = [
        {
            'speaker': '1',
            'text': 'We need to implement the database schema according to the requirements',
            'time': 1734094853.0  # Should return 1 (task-relevant)
        },
        {
            'speaker': '2',
            'text': 'Let\'s make sure everyone understands and agrees with the approach',
            'time': 1734094855.0  # Should return 2 (construct-relevant)
        },
        {
            'speaker': '3',
            'text': 'I think we should use the resistor to wiring the circuit',
            'time': 1734094857.0  # Should return 0 (not relevant)
        }
    ]
    
    analyzer = ContentAnalyzer(methods={'nlp'})
    
    # Test the analysis
    for record in test_transcription:
        status = analyzer.analyze_with_nlp(record['text'])
        print(f"Text: {record['text']}")
        print(f"Status: {status}\n")
