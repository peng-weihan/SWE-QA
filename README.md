# SWE-QA: Software Engineering Q&A Benchmark

A comprehensive benchmark and evaluation framework for Software Engineering question-answering systems. This repository contains benchmark datasets, evaluation tools, and experimental scripts for assessing the performance of various QA approaches on real-world software engineering questions.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Benchmark Datasets](#benchmark-datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

SWE-QA is a benchmark designed to evaluate question-answering systems on software engineering tasks. It covers 15 popular open-source Python projects, including frameworks, libraries, and tools such as Django, Flask, Requests, Pytest, and more.

The benchmark includes:
- **15 repositories** with specific commit versions
- **Multiple question categories**: architecture, design rationale, API usage, performance, etc.
- **Reference answers** for each question
- **Evaluation tools** using LLM-as-a-Judge methodology

## 📁 Project Structure

```
SWE-QA/
├── Benchmark/                    # Benchmark dataset files (JSONL format)
│   ├── astropy.jsonl
│   ├── django.jsonl
│   ├── flask.jsonl
│   └── ...
├── Benchmark construction/       # Tools for building benchmarks
│   ├── repo_parser/             # Repository parsing utilities
│   ├── qa_generator/            # Question-answer generation
│   ├── issue_analyzer/          # Extract questions from GitHub issues
│   └── score/                   # Scoring scripts
├── Experiment/                   # Experimental QA approaches
│   └── Script/
│       ├── SWE-agent_QA/        # SWE-agent based QA
│       ├── OpenHands_QA/        # OpenHands agent QA
│       ├── Cursor-Agent_QA/     # Cursor agent QA
│       ├── rag_sliding_window/  # RAG with sliding windows
│       ├── rag_function_chunk/  # RAG with function chunks
│       └── llm_direct/          # Direct LLM approach
├── Seed_question/                # Question templates by category
├── llm-as-a-judge.py            # LLM-based evaluation tool
├── clone_repos.sh                # Script to clone repositories
├── repo_with_version.txt         # Repository URLs and commit hashes
└── requirements.txt              # Python dependencies
```

## 📊 Benchmark Datasets

The benchmark includes questions from the following repositories:

1. **astropy** - Astronomy and astrophysics Python library
2. **django** - High-level Python web framework
3. **flask** - Lightweight WSGI web application framework
4. **matplotlib** - Python plotting library
5. **pylint** - Python code analyzer
6. **pytest** - Python testing framework
7. **requests** - HTTP library for Python
8. **scikit-learn** - Machine learning library
9. **sphinx** - Documentation generator
10. **sqlfluff** - SQL linter
11. **sympy** - Symbolic mathematics library
12. **xarray** - N-dimensional labeled arrays
13. **conan** - C/C++ package manager
14. **reflex** - Python web framework
15. **streamlink** - Streamlink is a command-line utility

Each dataset file (`.jsonl`) contains questions and reference answers in JSON Lines format:

```json
{"question": "How to...", "answer": "Reference answer..."}
```

## 🔧 Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd SWE-QA
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Clone benchmark repositories (optional):**
```bash
bash clone_repos.sh
```

4. **Set up environment variables for evaluation:**
Create a `.env` file or export the following variables:
```bash
export EVAL_LLM_BASE_URL="your-azure-openai-endpoint"
export EVAL_LLM_API_VERSION="your-api-version"
export EVAL_LLM_API_KEY="your-api-key"
export EVAL_LLM_MODEL_NAME="your-model-name"
```

## 🚀 Usage

### Running Experiments

Each experimental approach has its own script in the `Experiment/Script/` directory. Refer to the specific README or documentation in each subdirectory.

### Generating Questions and Answers

Use the benchmark construction tools to generate new questions and answers:

```bash
cd "Benchmark construction"
# Follow the instructions in the respective modules
```

### Cloning Specific Repository Versions

To clone repositories at the exact versions used in the benchmark:

```bash
bash clone_repos.sh
```

Make sure to set the `TARGET_DIR` variable in the script to specify where repositories should be cloned.

## 📝 Evaluation

### LLM-as-a-Judge Evaluation

The `llm-as-a-judge.py` script evaluates candidate answers against reference answers using an LLM judge. It scores answers on five dimensions:

- **Correctness** (1-20): Factual accuracy
- **Completeness** (1-20): Coverage of key points
- **Relevance** (1-20): Focus on the question topic
- **Clarity** (1-20): Expression clarity and fluency
- **Reasoning** (1-20): Logical structure and argumentation

**Total Score: 100 points**

### Single File Evaluation

```bash
export EVAL_CANDIDATE_PATH="path/to/candidate_answers.jsonl"
export EVAL_REFERENCE_PATH="path/to/reference_answers.jsonl"
export EVAL_OUTPUT_PATH="path/to/output_scores.jsonl"
export EVAL_LLM_MODEL_NAME="gpt-4"
python llm-as-a-judge.py
```

### Batch Evaluation (Multiple Files)

```bash
export EVAL_CANDIDATE_PATHS="file1.jsonl,file2.jsonl,file3.jsonl"
export EVAL_REFERENCE_PATH="path/to/reference_answers.jsonl"
export EVAL_OUTPUT_DIR="path/to/output_dir"
export EVAL_LLM_MODEL_NAME="gpt-4"
python llm-as-a-judge.py
```

### Directory Mode (Auto-discover Files)

```bash
export EVAL_CANDIDATE_DIR="path/to/candidate_answers_directory"
export EVAL_REFERENCE_PATH="path/to/reference_answers"
export EVAL_OUTPUT_DIR="path/to/output_dir"
export EVAL_LLM_MODEL_NAME="gpt-4"
export EVAL_REPO_FILTER="requests,flask,pytest"  # Optional: filter specific repos
python llm-as-a-judge.py
```

### Configuration Options

- `EVAL_MAX_WORKERS`: Maximum parallel threads (default: 16 for single file, 48 for batch)
- `EVAL_REPO_FILTER`: Comma-separated list of repository names to process (directory mode only)

The evaluation tool supports:
- ✅ Parallel processing for faster evaluation
- ✅ Resume from checkpoint (skips already processed questions)
- ✅ Thread-safe file writing
- ✅ Progress tracking

## 📂 Seed Questions

The `Seed_question/` directory contains question templates organized by category:

- `architecture.txt` - Architecture-related questions
- `design-rationale.txt` - Design decisions and rationale
- `api-framework.txt` - API and framework usage
- `performance.txt` - Performance-related questions
- `purpose.txt` - Purpose and usage questions
- And more...

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

See [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{swe-qa,
  title={SWE-QA: Software Engineering Question-Answering Benchmark},
  author={...},
  year={2024},
  url={...}
}
```

## 🔗 Related Resources

- Repository versions are specified in `repo_with_version.txt`
- Each repository is pinned to a specific commit for reproducibility
- Reference answers are provided for all questions in the benchmark

## 📧 Contact

For questions or issues, please open an issue on GitHub.

