import os
from typing import Dict, Set
from typing import Dict
import spacy
from collections import Counter
from spacy.matcher import Matcher
import re
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
    def __init__(self, methods: Set[str] = {'nlp'}, task_context: Dict = {}, construct_context: Dict = {}):
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
        self.task_context = task_context
        self.construct_context = construct_context

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

def extract_context(text: str, nlp=None) -> Dict:
    """
    Extract document text, key phrases and keywords from input text.
    
    Args:
        text: Input text to process
        nlp: Optional spaCy model (will load if not provided)
        
    Returns:
        Dictionary containing document_text, key_phrases, and keywords
    """
    # Initialize spaCy if not provided
    if nlp is None:
        nlp = spacy.load("en_core_web_md")
    
    # Process the text
    doc = nlp(text)
    
    # Extract key phrases using noun chunks and verb phrases
    matcher = Matcher(nlp.vocab)
    
    # Define patterns for key phrases
    patterns = [
        # Noun phrase patterns
        [{"POS": "ADJ"}, {"POS": "NOUN"}],  # e.g., "digital read"
        [{"POS": "NOUN"}, {"POS": "NOUN"}],  # e.g., "circuit connections"
        # Verb phrase patterns
        [{"POS": "VERB"}, {"POS": "NOUN"}],  # e.g., "press button"
        # Technical term patterns
        [{"ORTH": "RGB"}, {"ORTH": "LED"}],  # e.g., "RGB LED"
    ]
    
    # Add patterns to matcher
    for i, pattern in enumerate(patterns):
        matcher.add(f"pattern_{i}", [pattern])
    
    # Find matches
    matches = matcher(doc)
    key_phrases = []
    for match_id, start, end in matches:
        phrase = doc[start:end].text.lower()
        if len(phrase.split()) > 1:  # Only keep multi-word phrases
            key_phrases.append(phrase)
    
    # Extract keywords (important nouns, verbs, and adjectives)
    keywords = []
    for token in doc:
        # Check if token is a relevant part of speech and not a stopword
        if (token.pos_ in ['NOUN', 'VERB', 'ADJ'] and 
            not token.is_stop and 
            len(token.text) > 2):  # Avoid very short words
            keywords.append(token.text.lower())
    
    # Count frequencies and get most common
    key_phrases = list(set(key_phrases))  # Remove duplicates
    keywords = [word for word, count in Counter(keywords).most_common(20)]  # Top 20 keywords
    
    return {
        'document_text': text,
        'key_phrases': key_phrases,
        'keywords': keywords
    }

# Example usage
if __name__ == "__main__":
    task_text = """
    Inventors Kit Experiment 10 - Using An RGB LED:
    This experiment puts the RGB LED to use. An RGB LED is a special LED that contains three separate LEDs in one package. As you might have guessed the three LEDs are Red, Green and Blue. The light from these LEDs can be mixed together to allow for the creation of many colours. We can use the PWM function of the BBC micro:bit to have a very fine control over the colours and shades The RGB LED included in this Inventor’s pack is a common cathode LED which means all three LEDs inside the package share the same negative leg

    Tutorial:
    in this experiment as you can see there are a lot of connections. here we have three switches there used in conjunction with resistors. so there's one resistor for each switch and number of connections that feedback to the BBC micro-bits. these connect to imports p0 p1 and p2. these inputs are used to read when these switches are pressed. we then also have an RGB LED so this has a red green and blue element. each element has its own independent resistor and again it's connected back to the BBC micro bits. the way it works is that when we press one of these buttons it will increase the corresponding LED element. so one button controls the red element, one controls the blue element from one controls the green element pressing the button that corresponds to that element will make that element with the LED brighter. I can show you this now. now for the benefit of the camera I've got a little piece of perspex just so you can see the colour that's be an output clearer. so when I press this we can see the green element is getting brighter. so it's doing a digital read and there's a variable that's held the hole was the brightness of the green element. so each time we press the button the brightness is increased and it's controlling this brightness by using a pulse width modulated output from the BBC marker bit you see the green areas getting brighter we can then add in some of the red you can see the cycle background it goes back to zero just irrelevant on its own and here we have the P element so each press here is increasing the corresponding variable value that corresponds to the brightness here for this experiment obviously is a large number of connections and it is crucial that we try make sure everything is lined up one thing I would take a lot care with is the LED legs on the RGB LED if I take this out you'll see it's got a number of legs and they vary in length the longest leg is their common this case of common cathode to connector to a negative connection on the microbrew on the other legs correspond to the other colors with an LED the red green and blue and these are indicated by the various lens so make sure you take care to get this correct.

    Units:
    1 x Perspex Mounting Plate.
    1 x Potentiometer & Finger Adjust Spindle.
    2 x Plastic Spacer 10mm.
    1 x Sticky Fixer for Battery Pack.
    1 x Small Prototype Breadboard.
    1 x Terminal Connector.
    4 x Push Switch.
    1 x Motor.
    1 x Transistor.
    2 x Red 5mm LED.
    2 x Orange 5mm LED.
    2 x Yellow 5mm LED.
    2 x Green 5mm LED.
    1 x RGB 5mm LED.
    1 x Fan Blade.
    5 x 2.2KΩ Resistor.
    5 x 10KΩ Resistor.
    5 x 47Ω Resistor.
    1 x Edge Connector Breakout Board for BBC micro:bit.
    1 x Miniature LDR.
    10 x Male to Male Jumper Wires.
    10 x Male to Female Jumper Wires.
    4 x Self-adhesive Rubber Feet.
    1 x 470uF Electrolytic Capacitor.
    1 x Piezo Element Buzzer.
    4 x Pan Head M3 Machine Screw.
    """

    construct_text = """
    CPS Constrcut Definition:

    We define the constructing shared knowledge facet as actively disseminating one's own ideas/knowledge and understanding others' ideas/knowledge. It contains two sub-facets—sharing understanding and establishing common ground. Sharing understanding refers to group members contributing their expertise and ideas regarding the constraints of particular problems, as well as ideas toward specific solutions. Examples of associated behavioral indicators include proposing specific solutions and talking about the givens and constraints of the task. Sharing ideas/information is a critical step towards establishing common ground among team members. To further build common ground, team members should acknowledge others' ideas and expertise, confirm each other's understanding, and clarify misunderstanding when necessary. Seeking confirmation from other group members about current understanding is effective in reducing uncertainty, whereas clarifying misunderstanding provides a learning opportunity, where members explicate their knowledge, which may be implicit, to other group members. We include interruption as a negative indicator because interrupting while others are speaking can impede CPS, although some interruption can help rectify misunderstandings immediately.

    The second CPS facet is negotiation and coordination, an iterative process where team members achieve an agreed upon solution that is ready for execution. The goal of negotiation and coordination is to clearly specify shared goals, divide labor as warranted, manage synchrony among members, and produce a joint work product. Negotiation should reduce uncertainties, resolve conflict through integrating different perspectives, and contribute to a collective solution via joint expertise. Two relevant sub-facets include responding to others' questions/ ideas and monitoring execution. Responding to others' ideas and questions is a key part of coordination and should contribute to reciprocal and balanced exchanges of ideas and knowledge. As such, team members should provide feedback regarding others' ideas, offer reasons to support or refute certain claims, negotiate when disagreements occur, and implement consensual solutions after discussion. Hence, providing reasons to support a potential solution is a positive indicator of this sub-facet while failing to respond when spoken to by others is a negative indicator. The second sub-facet, monitoring execution, requires team members to appraise whether the solution plan works as expected, and their progress toward task completion. Members should also be able to evaluate their own and others' actions, knowledge, and skills towards task completion. When things go awry, team members need to be able to modify or overhaul the solution if necessary. Thus, talking about progress and results is the key indicator of monitoring execution. Team members may occasionally propose to quit certain tasks, captured in the reverse coded “brings up giving up the challenge” indicator.

    It is important that all team members are aware of being part of a team and realize that individual behaviors affect team success. Successful teams iteratively talk about team organization. Thus, the third main facet of CPS involves maintaining team function. Maintaining a positive and effective team requires members to take distributed responsibility to contribute to the quantity and quality of the collaborative venture. Consequently, one sub-facet of maintaining team function pertains to each member performing his or her own roles/responsibilities within the team. The roles may be assigned by an external source, or more naturalistically evolve during collaboration. Either way, team members should stay focused on the task and on what is needed in their roles, while not distracting themselves or others. The second sub-facet – taking initiative to advance CPS processes – includes asking questions, acknowledging others' contributions, and helping to maintain team organization. Taking initiative predicts productive collaboration. In effective teams, team members not only regulate their own activities but also evaluate the activities of other members. Because one of the ground rules for collaboration is that group members encourage each other to speak, relevant indicators include asking if others have suggestions and complimenting or encouraging their team members.

    CPS Constrcut Details
    The facet and subfacets are latent varibales, while the indicators are observables. The counter indicators mean it will impede the facet.
    1.
    Facet:
    Constructing shared knowledge—expresses one's own ideas and attempts to understand others' ideas.

    Subfacets:
    Shares understanding of problems and solutions
    Establishes common ground

    Indicators:
    Talks about specific topics/concepts and ideas on problem solving: Proposes specific solutions, Talks about givens and constraints of a specific task, Builds on others' ideas to improve solutions 
    Recognizes and verifies understanding of others' ideas: Confirms understanding by asking questions/ paraphrasing, Repairs misunderstandings

    Reversed-indicators:
    Interrupts or talks over others as intrusion


    2.
    Facet:
    Negotiation/Coordination—achieves an agreed solution plan ready to execute

    Subfacets:
    Responds to others' questions/ideas
    Monitors execution

    Inidcators:
    Provides reasons to support/refute a potential solution
    Makes an attempt after discussion
    Talks about results

    Reversed-indicators:
    Does not respond when spoken to by others
    Makes fun of, criticizes, or is rude to others
    Brings up giving up the challenge


    3.
    Facet:
    Maintaining team function—sustains the team dynamics

    Subfacets:
    Fulfills individual roles on the team
    Takes initiatives to advance collaboration processes

    Indicators:
    Asks if others have suggestions
    Asks to take action before anyone on the team asks for help
    Compliments or encourages others

    Reversed-indicators:
    Not visibly focused on tasks and assigned roles
    Initiates off-topic conversation
    Joins off-topic conversation

    Examples
    Player C: “What if you grabbed it upwards. And then drew a pendulum, knocked it out. But you drew like farther out, the pendulum” (Proposes specific solution) 

    Player A: “I have an idea. Wait, which direction should I swing?” (Confirms understanding by asking questions/paraphrasing) 

    Player C: “Swing from here to here.” (Proposes specific solution) 

    Player A: “Nope, then it would just fly to the spider.” (Provides reason to support/refute a potential solution)





    """

    
    # Extract context
    task_context = extract_context(task_text)
    construct_context = extract_context(construct_text)

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
            'text': 'What is your name?',
            'time': 1734094857.0  # Should return 0 (not relevant)
        }
    ]
    
    analyzer = ContentAnalyzer(methods={'nlp'}, task_context=task_context, construct_context=construct_context)
    
    # Test the analysis
    for record in test_transcription:
        status = analyzer.analyze_with_nlp(record['text'])
        print(f"Text: {record['text']}")
        print(f"Status: {status}\n")
