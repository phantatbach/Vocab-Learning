---

# 📚 Vocab-Learning

A lightweight Python pipeline for generating example sentences for vocabulary lists, translating them, and optionally clustering the vocab semantically using word embeddings — ideal for learners, teachers, or language apps.

---

## 🚀 Overview

The pipeline performs the following steps:

1. **Load** vocabulary from a CSV
2. **Generate** example sentences using **Google Gemini**
3. **Translate** vocab and examples using **Google Translate**
4. **(Optional)** Cluster vocabulary into topics using pretrained embeddings
5. **Export** a CSV with four or five columns:

   * `Vocab`
   * `Translated_Vocab`
   * `Examples`
   * `Translated_Examples`
   * `Group` *(optional, added if clustering is enabled)*

---

## ✨ Features

* 🧠 **Example generation** via Gemini with retry logic
* 🌐 **Batch translation** with fallback using `googletrans`
* 📊 **Optional clustering** using K-Medoids and pretrained word embeddings
* 🔧 **Fully configurable** via CLI or Python
* 🧩 **Modular**: use generation, translation, or clustering independently

---

## 📦 Requirements

* Python ≥ 3.8
* A Google Gemini API key
* Pretrained word embeddings (`.txt` or `.bin`, e.g., FastText)
* Install dependencies:

```bash
pip install argparse pandas googletrans google.generativeai gensim scikit-learn scikit-learn-extra
```

---

## 🛠️ Usage

### ✅ 1. Generate + Translate from CLI

```bash
python vocab_pipeline.py \
  path/to/input.csv \
  path/to/output.csv \
  --src_lang nl \
  --dest_lang en \
  --prompt "Geef één eenvoudig voorbeeldzin in het Nederlands voor het woord '{word}'..." \
  --api_key YOUR_GEMINI_API_KEY \
  --model_name gemini-1.5-flash
```

---

### ✅ 2. Cluster Vocab into Semantic Groups

```bash
python vocab_cluster.py \
  path/to/input.csv \
  path/to/output.csv \
  15 \
  --embedding_path Embeddings/nl/model.txt \
  --lang nl \
  --vocab_column Vocab
```

#### Required args:

| Arg                | Description                                     |
| ------------------ | ----------------------------------------------- |
| `input.csv`        | CSV with a `Vocab` column                       |
| `output.csv`       | Destination CSV with added `Group` column       |
| `15`               | Number of clusters                              |
| `--embedding_path` | Path to `.bin` or `.txt` pretrained embedding   |
| `--lang`           | Source language (e.g., `nl`)                    |
| `--vocab_column`   | Column name containing vocab (default: `Vocab`) |

---

### ✅ 3. Programmatic Example

```python
from vocab import VocabPipeline
from vocab_cl import VocabCluster

# Example generation + translation
pipeline = VocabPipeline(
    input_file='out.csv',
    output_file='out.csv',
    src_lang='nl',
    dest_lang='vi',
    prompt_template= "Geef één eenvoudig voorbeeldzin in het Nederlands voor het woord '{word}'. De zin moet geschikt zijn voor een A1-A2 taalniveau. De zin moet het woord '{word}' bevatten.",
    api_key='',
    model_name='gemini-1.5-flash'
)
pipeline.run()

# Clustering
cluster = VocabCluster(
    embedding_path='Embeddings/nl/model.txt',
    n_clusters=15,
    input_file='output.csv',
    output_file='output_with_clusters.csv',
    lang='nl',
    vocab_column='Vocab'
)
cluster.run()
```

---

## ✅ Output Format (after full pipeline)

| Vocab | Translated\_Vocab | Examples          | Translated\_Examples | Group |
| ----- | ----------------- | ----------------- | -------------------- | ----- |
| appel | apple             | Ik eet een appel. | I eat an apple.      | 3     |
| deur  | door              | De deur is open.  | The door is open.    | 1     |

---

## 🧠 Summary

* ⚡ One-stop solution for vocab enrichment
* 🤖 Combines LLM power with classical embedding clustering
* 🔧 Fully parameterized, reusable, and multilingual

---

## 📄 License

MIT License — use freely with attribution.

---
