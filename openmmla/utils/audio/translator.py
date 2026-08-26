"""Translator class to translate text using Google Translate API.

Example:
    translator = Translator()
    translator.translate_json_file("path/to/json/file.json")
"""

import json

from googletrans import Translator as GoogleTranslator


class Translator:
    def __init__(self, src='auto', dst='en'):
        self.translator = GoogleTranslator()
        self.src = 'auto'
        self.dst = 'en'

    def translate_text(self, text):
        try:
            if text is not None:
                translated = self.translator.translate(text, src=self.src, dest=self.dst)
                return translated.text
            else:
                return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return text

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
