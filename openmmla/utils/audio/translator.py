import json

from googletrans import Translator as GoogleTranslator


class Translator:
    def __init__(self, src='auto', dst='en'):
        self.translator = GoogleTranslator()
        self.src = 'auto'
        self.dst = 'en'

    def translate_text(self, text):
        try:
            # Check if text is not null
            if text is not None:
                translated = self.translator.translate(text, src=self.src, dest=self.dst)
                return translated.text
            else:
                return None  # return None if original text is None
        except Exception as e:
            print(f"An error occurred: {e}")
            return text  # return original text if translation fails

    def translate_json_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        for entry in json_data:
            original_text = entry.get('text', None)  # use dict.get() to handle missing keys
            print(original_text)
            translated_text = self.translate_text(original_text)
            print(translated_text)
            entry['text_translated'] = translated_text

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

# if __name__ == "__main__":
#     import os
#
#     log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../logs")
#     json_name = "session_2023-09-09T14:21:42Z_speaker_transcription.json"
#     json_path = os.path.join(log_dir, json_name)
#     translator = Translator()
#     translator.translate_json_file(json_path)
