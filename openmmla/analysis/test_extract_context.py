from collections import Counter
from typing import Dict

import spacy
from spacy.matcher import Matcher


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
    # Test text
    test_text = """
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

    # Extract context
    context = extract_context(test_text)

    # Print results
    print("Document Text:")
    print(context['document_text'][:100] + "...\n")

    print("Key Phrases:")
    for phrase in context['key_phrases']:
        print(f"- {phrase}")
    print()

    print("Keywords:")
    for word in context['keywords']:
        print(f"- {word}")
