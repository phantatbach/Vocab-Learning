import argparse
import logging
import time
import pandas as pd
from googletrans import Translator
import google.generativeai as genai

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class VocabPipeline:
    def __init__(self,
                 input_file: str,
                 output_file: str,
                 src_lang: str,
                 dest_lang: str,
                 prompt_template: str,
                 api_key: str,
                 model_name: str,
                 vocab_column: str = 'Vocab'):

        # File + column setup
        self.input_file   = input_file
        self.output_file  = output_file
        self.vocab_column = vocab_column

        # Langs + prompt
        self.src_lang        = src_lang
        self.dest_lang       = dest_lang
        self.prompt_template = prompt_template

        # Gemini setup
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

        # Translator
        self.translator = Translator()

    def load_csv(self):
        self.df = pd.read_csv(self.input_file)
        # Ensure all four cols exist
        for col in ['Examples', 'Translated_Vocab', 'Translated_Examples']:
            if col not in self.df.columns:
                self.df[col] = pd.NA

        if self.vocab_column not in self.df.columns:
            raise ValueError(f"Column '{self.vocab_column}' missing.")

    def generate_examples(self):
        def ask_model(word):
            prompt = self.prompt_template.format(word=word)
            resp   = self.model.generate_content(prompt)
            return getattr(resp, 'text', None) or \
                   (resp.candidates[0].content.parts[0] if resp.candidates else None)

        for idx, word in self.df[self.vocab_column].items():
            # skip if already have an example
            if pd.notna(self.df.at[idx, 'Examples']) and self.df.at[idx, 'Examples'].strip():
                continue
            if not isinstance(word, str) or not word.strip():
                self.df.at[idx, 'Examples'] = pd.NA
                continue

            try:
                text = ask_model(word).strip() or pd.NA
                self.df.at[idx, 'Examples'] = text
            except Exception as e:
                logging.error(f"Gen error @ '{word}': {e}")
                self.df.at[idx, 'Examples'] = pd.NA
            time.sleep(1)

    def translate_columns(self):
        # Translate Vocab
        for idx, word in self.df[self.vocab_column].items():
            if pd.notna(self.df.at[idx, 'Translated_Vocab']):
                continue
            if not isinstance(word, str) or not word.strip():
                self.df.at[idx, 'Translated_Vocab'] = pd.NA
                continue
            try:
                self.df.at[idx, 'Translated_Vocab'] = \
                    self.translator.translate(word,
                                              src=self.src_lang,
                                              dest=self.dest_lang).text
            except Exception as e:
                logging.error(f"Translate vocab fail '{word}': {e}")
                self.df.at[idx, 'Translated_Vocab'] = pd.NA

        # Translate Examples
        for idx, ex in self.df['Examples'].items():
            if pd.notna(self.df.at[idx, 'Translated_Examples']):
                continue
            if not isinstance(ex, str) or not ex.strip():
                self.df.at[idx, 'Translated_Examples'] = pd.NA
                continue
            try:
                self.df.at[idx, 'Translated_Examples'] = \
                    self.translator.translate(ex,
                                              src=self.src_lang,
                                              dest=self.dest_lang).text
            except Exception as e:
                logging.error(f"Translate example fail '{ex}': {e}")
                self.df.at[idx, 'Translated_Examples'] = pd.NA

    def save_csv(self):
        cols = [
            self.vocab_column,
            'Translated_Vocab',
            'Examples',
            'Translated_Example'
        ]
        self.df.to_csv(self.output_file, columns=cols, index=False, encoding='utf-8-sig')
        logging.info(f"Output saved to {self.output_file}")

    def run(self):
        self.load_csv()
        self.generate_examples()
        self.translate_columns()
        self.save_csv()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('--src_lang',     default='nl')
    parser.add_argument('--dest_lang',    default='en')
    parser.add_argument('--prompt',       required=True,
                        help="Use '{word}' placeholder")
    parser.add_argument('--api_key',      required=True)
    parser.add_argument('--model_name',   default='gemini-1.5-flash')
    parser.add_argument('--vocab_column', default='Vocab')
    args = parser.parse_args()

    pipeline = VocabPipeline(
        input_file      = args.input_file,
        output_file     = args.output_file,
        src_lang        = args.src_lang,
        dest_lang       = args.dest_lang,
        prompt_template = args.prompt,
        api_key         = args.api_key,
        model_name      = args.model_name,
        vocab_column    = args.vocab_column
    )
    pipeline.run()
