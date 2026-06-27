<<<<<<< HEAD
Audience-Aware Adaptation of Curatorial Texts with Large Language Models

This repository contains the full pipeline for a bachelor's thesis investigating whether large language models can adapt museum curatorial texts for non-expert audiences while preserving curatorial voice and meaning. The pipeline takes a dataset of 699 original curatorial texts, generates four adaptation variants for each, and evaluates them across readability, voice, semantic, LLM-judge, and human dimensions.
Overview
Each original text is adapted using four strategies:
StrategyTarget readerTestsLexicalMinimal-intervention vocabulary swapEffect of vocabulary complexity alone on curatorial voicePersona 1Native English, ~13–15 yrs, no art knowledgeCombined lexical + syntactic simplification, maximum readabilityPersona 2Non-native (B2–C1), deep art expertiseSyntactic simplification while preserving disciplinary signalsPersona 3Native, university-level, no art expertiseDomain-specific (jargon) simplification for the general visitor
Adaptations are produced with DeepSeek (deepseek-chat); the LLM-as-a-Judge evaluation uses OpenAI (gpt-4o-mini).

Repository structure
.
├── Config/
│   └── api_keys.env                 # DEEPSEEK_API_KEY and OPENAI_API_KEY
├── Prompts/
│   ├── lexical_prompt.txt
│   ├── persona1_prompt.txt
│   ├── persona2_prompt.txt
│   ├── persona3_prompt.txt
│   └── llm_as_a_judge.txt
├── Data/
│   ├── Original/
│   │   └── Dataset_Curatorial_Text.xlsx
│   ├── adapted/                     # per-strategy outputs + checkpoints
│   ├── Merging_Files/               # combined master dataset
│   ├── Readability_Results/
│   ├── Voice_Metrics/
│   ├── Semantic_Metrics/
│   ├── Judge_Evaluation/
│   └── Human_Evaluation/
├── Results/
│   ├── Tables/
│   └── Figures/
└── Scripts/  
                       # the numbered pipeline below
Prerequisites
Python 3.9+
API keys for DeepSeek and OpenAI
A stable internet connection (long DeepSeek runs failed under weak Wi-Fi during development)

Installation
bashpip install pandas numpy openpyxl python-dotenv tqdm httpx openai tenacity \
            textstat nltk sentence-transformers scikit-learn bert-score spacy \
            matplotlib seaborn scipy requests
python -m spacy download en_core_web_sm
API keys
Create Config/api_keys.env:
DEEPSEEK_API_KEY=your_deepseek_key_here
OPENAI_API_KEY=your_openai_key_here
This file should be listed in .gitignore and never committed.
Input dataset
Data/Original/Dataset_Curatorial_Text.xlsx must contain at least these columns: ID, ReferenceNumber, Museum, NameOfTheObject, TextType, CuratorialText, WordCountCuratorialText. The reference dataset holds 699 texts (mean ~102 words) across four text types (ObjectLabel, SectionText, ExhibitionText, WallText).
Running the pipeline
Run scripts in numerical order. Each stage reads the output of earlier stages.
0 — Validation

0_validate_the_dataset.py — checks required columns, flags empty texts, prints length/type distributions. Run this first.

1–4 — Adaptation (DeepSeek)

0_1_run_adaptation.py — interactive menu to run one strategy or all four in sequence.
1_lexical_adaptation.py, 2_persona_1_adaptation.py, 3_persona_2_adaptation.py, 4_persona_3_adaptation.py — generate each variant. All use temperature 0.3, exponential-backoff retries (up to 8), checkpoint every 5 rows, and a second pass over failed rows. Outputs are written to Data/adapted/<strategy>/ in utf-8-sig (this encoding fixed corrupted-character issues seen with plain utf-8).

5 — Combine

5_combine_adaptations.py — merges the original dataset and all four adaptations into Data/Merging_Files/dataset_all_adaptations.csv, standardizing missing values.

6–8 — Computational metrics

6_calculate_readability.py — Flesch-Kincaid grade and SMOG index per text version → Data/Readability_Results/.
7_calculate_voice_metrics.py — hedging, attribution, and art-terminology density → Data/Voice_Metrics/.
8_evaluate_semantic_metrics.py — semantic similarity (MiniLM cosine), entity preservation (spaCy), and BERTScore F1 → Data/Semantic_Metrics/.

9 — LLM-as-a-Judge (OpenAI)

9_llm_as_a_judge_evaluation.py — rates each adaptation on voice preservation, meaning preservation, and accessibility (parallelized with ThreadPoolExecutor, 3 workers). Prompts you for how many texts to evaluate. → Data/Judge_Evaluation/judge_evaluations.csv.
9.1.parse_llm_judge_evaluation.py - extracts all the sub-dimensions of the voice preservation, meaning preservation, and accessability
9_2_revluatate_failed_judge_scores.py — re-runs only rows where parsing failed or scores are missing, and patches them back into the same file.


10 — Human evaluation

10_human_eval.py — generates a formatted Excel sample (per strategy) with blank rating columns for manual human evaluation.
10_1_human_eval_parging.py — parses the completed Excel files and writes per-strategy human means.

11–14 — Analysis

11_descriptive_statistics.py — means/SDs/medians for main dimensions and subdimensions, plus bar/distribution figures.
12_rq1_best_metrics.py — RQ1: which computational metrics align with judge scores (Pearson correlations + heatmap).
13_rq2_alignment_analysis.py — RQ2: alignment between judge dimensions and linguistic metrics, including Kendall's tau ranking agreement.
14_main_rq_tradeoff.py — main RQ: the accessibility-vs-preservation trade-off, efficiency ranking, and acceptability thresholds.

Utilities

check_api.py — quick OpenAI connectivity / rate-limit check. Load the key from the env file rather than hard-coding it.

Notes on reproducibility

All text I/O uses utf-8-sig encoding.
Adaptation runs are checkpointed, so an interrupted run can resume from the last checkpoint.
Judge calls use temperature=0 for determinism; adaptation calls use temperature=0.3.
10_human_eval.py seeds its random sample from the current timestamp, so set a fixed seed if you need reproducible sampling.
=======
# Thesis-
>>>>>>> fe6c106c7541100de484815a9b44df3cdcd61cd5
