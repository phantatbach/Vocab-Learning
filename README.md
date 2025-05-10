# Vocab-Learning
Generate Examples for a list of Vocab then translate both the Vocab and the Examples to another language.

````markdown
# VocabPipeline

A lightweight, end-to-end Python pipeline that takes a CSV of source-language vocabulary, generates example sentences with Google Gemini AI, translates both the words and examples via Google Translate, and exports a combined CSV with four columns:

- **Vocab**  
- **Translated_Vocab**  
- **Examples**  
- **Translated_Examples**  

---

## 🌍 Big Picture

1. **Load** your list of words from a CSV.  
2. **Generate** simple, level-appropriate example sentences in the source language using Gemini.  
3. **Translate** both the vocabulary and the generated examples into your target language.  
4. **Save** the result as a new CSV for easy review or integration into learning tools.

---

## 📋 Features

- **Retry logic** for any missing or failed Gemini generations  
- **Batch + fallback** for fast, robust Google Translate calls  
- **Fully parameterised**: swap in any `src_lang`, `dest_lang`, prompt template, API key, or Gemini model  
- **CLI & programmatic** interfaces  

---

## ⚙️ Requirements

- Python 3.8+  
- A Google Gemini API key (via `genai`)  
- Dependencies:
  ```bash
  pip install argparse logging time os pandas googletrans google.generativeai
````

---
## 🚀 Installation
---

## 💡 Usage

### 1. Command-line

```bash
python vocab_pipeline.py \
  path/to/input.csv \
  path/to/output.csv \
  --src_lang nl \
  --dest_lang en \
  --prompt "Geef één eenvoudig voorbeeldzin in het Nederlands voor het woord '{word}'. De zin moet geschikt zijn voor een A1-A2 taalniveau en het woord '{word}' bevatten." \
  --api_key YOUR_GEMINI_API_KEY \
  --model_name gemini-1.5-flash
```

* **`input.csv`**: your source-language vocab list (one column, default header `Vocab`)
* **`output.csv`**: path where the 4-column result will be written
* **`--prompt`**: must include `'{word}'` exactly where the model should insert each vocab item
* **`--api_key`**: your Gemini API key
* **`--model_name`**: e.g. `gemini-1.5-flash`

### 2. Programmatic

```python
from vocab import VocabPipeline

pipeline = VocabPipeline(
    input_file      = r'D:\Data\vocab_copy.csv',
    output_file     = 'out.csv',
    src_lang        = 'nl',
    dest_lang       = 'en',
    prompt_template = (
        "Geef één eenvoudig voorbeeldzin in het Nederlands voor het woord "
        "'{word}'. De zin moet geschikt zijn voor een A1-A2 taalniveau en "
        "het woord '{word}' bevatten."
    ),
    api_key         = 'YOUR_GEMINI_API_KEY',
    model_name      = 'gemini-1.5-flash'
)
pipeline.run()
```

---

## 🔧 Configuration & Extension

* **Prompt template**: swap out for any task—just include `'{word}'` as placeholder.
* **Languages**: use any ISO-language codes for `src_lang`/`dest_lang`.
* **Retry & rate limits**: adjust `time.sleep(1)` in `generate_examples()` as needed.
* **Translator**: swap `googletrans` for another library if preferred.
* **Model**: point `model_name` at any Gemini/GenAI endpoint you have access to.

---

## ✅ Summary

* **One-stop script**: no manual loops or per-word juggling.
* **Robust**: retries, fallbacks, and clear logging for smooth runs.
* **Flexible**: fully parameterised for other languages, prompts, and models.

Happy vocabulary-building! 🚀
---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

```
