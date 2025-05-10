import argparse
import logging
import time
import os
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
        # File & column settings
        self.input_file   = input_file
        self.output_file  = output_file
        self.vocab_column = vocab_column

        # Languages & prompt
        self.src_lang        = src_lang
        self.dest_lang       = dest_lang
        self.prompt_template = prompt_template

        # Configure Google Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

        # Translator client
        self.translator = Translator()

        # DataFrame placeholder
        self.df = None

    def load_csv(self):
        self.df = pd.read_csv(self.input_file)
        if self.vocab_column not in self.df.columns:
            raise ValueError(f"Column '{self.vocab_column}' not found in {self.input_file}")

    def generate_examples(self):
        words = self.df[self.vocab_column].fillna('').astype(str).tolist()
        examples = {}

        def ask_model(w):
            prompt = self.prompt_template.format(word=w)
            resp = self.model.generate_content(prompt)
            # adapt if SDK shape differs
            return getattr(resp, 'text', None) or \
                   (resp.candidates[0].content.parts[0] if resp.candidates else None)

        # First pass
        for w in words:
            if not w:
                examples[w] = None
                continue
            try:
                examples[w] = ask_model(w).strip()
            except Exception as e:
                logging.warning(f"1st pass failed '{w}': {e}")
                examples[w] = None
            time.sleep(1)

        # Retry any misses
        missing = [w for w, ex in examples.items() if not ex]
        if missing:
            logging.info(f"Retrying {len(missing)} missing examples...")
            for w in missing:
                try:
                    examples[w] = ask_model(w).strip() or "—"
                except Exception as e:
                    logging.error(f"Retry failed '{w}': {e}")
                    examples[w] = "—"
                time.sleep(1)

        self.df['Examples'] = self.df[self.vocab_column].map(examples)

    def translate_columns(self):
        # Vocab translations
        vocab_list = self.df[self.vocab_column].tolist()
        try:
            trans_objs = self.translator.translate(
                vocab_list, src=self.src_lang, dest=self.dest_lang
            )
            self.df['Translated_Vocab'] = [t.text for t in trans_objs]
        except Exception:
            logging.warning("Batch vocab translation failed; falling back to singles.")
            self.df['Translated_Vocab'] = [
                self.translator.translate(w, src=self.src_lang, dest=self.dest_lang).text
                if w else None
                for w in vocab_list
            ]

        # Example translations
        ex_list = self.df['Examples'].fillna('').tolist()
        try:
            ex_objs = self.translator.translate(
                ex_list, src=self.src_lang, dest=self.dest_lang
            )
            self.df['Translated_Example'] = [t.text for t in ex_objs]
        except Exception:
            logging.warning("Batch example translation failed; falling back to singles.")
            self.df['Translated_Example'] = [
                self.translator.translate(ex, src=self.src_lang, dest=self.dest_lang).text
                if ex else None
                for ex in ex_list
            ]

    def save_csv(self):
        cols = [self.vocab_column, 'Translated_Vocab', 'Examples', 'Translated_Example']
        self.df.to_csv(self.output_file, columns=cols, index=False)
        logging.info(f"Saved output to {self.output_file}")

    def run(self):
        self.load_csv()
        self.generate_examples()
        self.translate_columns()
        self.save_csv()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generalised vocab-to-examples pipeline"
    )
    parser.add_argument('input_file',  help="Path to input CSV")
    parser.add_argument('output_file', help="Path to output CSV")
    parser.add_argument('--src_lang',     required=True, help='Specify your source language!')
    parser.add_argument('--dest_lang',    default='en')
    parser.add_argument('--prompt',       required=True,
                        help="Prompt template, use '{word}' as placeholder")
    parser.add_argument('--api_key',      required=True,
                        help="Your Gemini API key")
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