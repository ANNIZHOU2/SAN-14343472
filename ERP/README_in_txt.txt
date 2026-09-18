# Corporate AI Discourse Analysis
This repository contains the code and firm metadata used to analyse changes in corporate AI discourse before and after the diffusion of generative AI. The workflow analyses earnings-call transcripts through preprocessing, keyword screening, semantic similarity review, BERTopic topic discovery and quantitative text analysis.
The original earnings-call transcripts are not included in this repository. They must be obtained from Wharton Research Data Services (WRDS). The detailed data structure, coding rules, model specifications and output schemas are documented in the technical appendices.

## Reproducibility Notes
- The study analyses 45 U.S.-listed technology-intensive firms and earnings calls from 2021 to 2024.
- The pre-GenAI period covers calls through 2022Q3.
- 2022Q4 is treated as a transition period and excluded from the main pre/post regressions.
- The post-GenAI period begins in 2023Q1.
- To comply with WRDS data-use and confidentiality restrictions, transcript-derived text fields  have been cleared, where applicable, while the associated identifiers and coding fields have been retained for reproducibility purposes.

## Repository Structure
```text
repository-root\
├── transcripts\              # WRDS transcripts used as preprocessing input
├── 1_preprocessing.py
├── 2_semantic_screening.py
├── 3_BERTopic.py
├── 4_qta.py
├── earnings_call_metadata.csv
├── requirements.txt
├── README.md
├── results\
└── qta_results\
```
## Repository Contents
| File | Purpose |
| --- | --- |
| `1_preprocessing.py` | Cleans transcript paragraphs and performs preliminary AI keyword screening. |
| `2_semantic_screening.py` | Checks whether excluded paragraphs may be semantically related to verified AI excerpts. |
| `3_BERTopic.py` | Discovers latent topics in manually reviewed AI excerpts. |
| `4_qta.py` | Produces descriptive statistics, regressions, robustness checks, tables and figures. |
| `earnings_call_metadata.csv` | Provides firm identifiers and AI value-chain group classifications. |
| `requirements.txt` | Lists the required Python packages. |

## Environment
The scripts require Python 3.10 or later. Install the required packages from the repository directory:
```powershell
python -m pip install -r requirements.txt
```
`2_semantic_screening.py` requires an OpenAI-compatible endpoint serving the `bge-m3` embedding model:
```powershell
$env:OPENAI_API_KEY = "your-api-key"
$env:OPENAI_BASE_URL = "your-openai-compatible-base-url"
```
Do not commit API keys, raw transcripts or other confidential data to the repository.

## Reproduction Workflow
After obtaining the transcripts from Wharton Research Data Services (WRDS), run the scripts from the directory containing the repository files, or provide the relevant input and output paths through the command-line options. The scripts do not depend on a fixed local path; when paths are omitted, they use the current working directory and its default subdirectories.

### 1. Preprocessing and keyword screening
```powershell
python 1_preprocessing.py --input path\to\raw_transcripts.csv --output-dir results
```
Main outputs:
```text
results/all_paragraphs.csv
results/ai_screening_results.csv
```
The second file must be manually reviewed. Substantive AI disclosures should be retained, and the fields `ai_relevant` and `ai_excerpt` should be completed according to Stage 1 of the manual screening code book in Appendix B.
### 2. Semantic similarity review
After manual screening, run:
```powershell
python 2_semantic_screening.py
```
The script reads the two files in `results` and produces:
```text
results/excluded_top_matches.csv
```
This output is used to manually check potentially relevant paragraphs missed by keyword screening. It does not automatically change the formal AI corpus.
### 3. Topic discovery
Run:
```powershell
python 3_BERTopic.py
```
The script uses the manually reviewed `ai_excerpt` values and produces:
```text
results/BERTopic_summary.xlsx
```
The automatically discovered topics must be manually reviewed and consolidated into the final thematic framework described in Stage 2 of Appendix B.
### 4. Quantitative text analysis
After final theme and attitude coding (more details in Stage 2 and Stage 3 of Appendix B), save the encoded corpus as:
```text
ai_paragraphs_semantically_encoded.csv
```
Then run:
```powershell
python 4_qta.py
```
The script uses the encoded corpus, `results/all_paragraphs.csv` and `earnings_call_metadata.csv`. Its results are written to:
```text
qta_results
```
The output includes the tables, figures, descriptive statistics, regressions and robustness checks used in the report and appendices.

## Manual Processing
Manual screening, excerpt extraction, theme consolidation and attitude coding are researcher-led stages. The repository supports these stages but does not automatically reproduce the associated decisions. The relevant rules, parameter settings and output documentation are provided in Appendix B and Appendix D onwards.
