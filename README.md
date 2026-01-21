# SWE-QA

**SWE-QA: Repository-Level Code Question-Answering Benchmark**

A comprehensive benchmark and evaluation framework for Software Engineering question-answering systems. This repository contains benchmark datasets, evaluation tools, and experimental scripts for assessing the performance of various QA approaches on real-world software engineering questions.

**📦 Complete Dataset**: The full benchmark dataset (576 questions across 12 repositories) is available on [Hugging Face](https://huggingface.co/datasets/swe-qa/SWE-QA-Benchmark). You can download it using:

```python
from datasets import load_dataset
dataset = load_dataset("swe-qa/SWE-QA-Benchmark")
```

## 📋 Table of Contents

- [Overview](#overview)
- [Benchmark Datasets](#benchmark-datasets)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Quick Start](#quick-start)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

SWE-QA is a benchmark designed to evaluate question-answering systems on software engineering tasks. It covers 15 popular open-source Python projects, including frameworks, libraries, and tools such as Django, Flask, Requests, Pytest, and more.

The benchmark includes:
- **15 repositories** with specific commit versions
- **Multiple question categories**: architecture, design rationale, API usage, performance, etc.
- **Reference answers** for each question
- **Evaluation tools** using LLM-as-a-Judge methodology

### Benchmark Construction Workflow

The following diagram illustrates the workflow for constructing the SWE-QA benchmark:

![Benchmark Construction Workflow](assets/workflow.png)

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

**Note**: For the complete benchmark dataset with all questions and reference answers, please download from [Hugging Face](https://huggingface.co/datasets/swe-qa/SWE-QA-Benchmark). The dataset includes 576 questions across 12 repositories, with each repository containing 48 questions.

### Benchmark Example

The following example shows the structure and format of questions in the benchmark:

![Benchmark Example](assets/example.png)

## 📁 Repository Structure

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
├── assets/                       # Images and resources
│   ├── workflow.png            # Benchmark construction workflow
│   └── example.png             # Benchmark example
├── llm-as-a-judge.py            # LLM-based evaluation tool
├── clone_repos.sh                # Script to clone repositories
├── repo_with_version.txt         # Repository URLs and commit hashes
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Environment Setup

### Prerequisites

- Python 3.11+
- pip or conda for package management
- OpenAI API access (required for evaluation and some methods)
- Voyage AI API access (required for RAG-based methods)
- Cursor API access (required for Cursor-Agent_QA method)

### Installation

**1. Clone the repository:**
```bash
git clone <repository-url>
cd SWE-QA
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Clone benchmark repositories:**
```bash
# Edit clone_repos.sh to set TARGET_DIR
bash clone_repos.sh
```

**4. Set up environment variables:**

Create a `.env` file in the root directory or export the following variables:

```bash
# For evaluation (llm-as-a-judge.py)
export EVAL_LLM_BASE_URL="your-openai-endpoint"
export EVAL_LLM_API_VERSION="your-api-version"
export EVAL_LLM_API_KEY="your-api-key"
export EVAL_LLM_MODEL_NAME="your-model-name"

# For Cursor-Agent_QA
export CURSOR_API_KEY="your-cursor-api-key"
export CURSOR_AGENT_PATH="/path/to/cursor-agent"  # Optional
```

## ⚡ Quick Start

### 1. Direct LLM Evaluation

Before executing, configure the environment variables by creating a `.env` file in the `Experiment/Script/llm_direct` directory:

```bash
OPENAI_BASE_URL=your_openai_base_url
OPENAI_API_KEY=your_api_key
MODEL=your_model_name
```

Evaluate language models directly on repository-level questions:

```bash
cd Experiment/Script/llm_direct
python main.py
```

This method will:
- Load questions from the benchmark dataset
- Send questions directly to the LLM
- Generate answers without additional context
- Save results to output directory

### 2. RAG with Function Chunking

Before executing, configure the environment variables by creating a `.env` file in the `Experiment/Script/rag_function_chunk` directory:

```bash
# Voyage AI Configuration
VOYAGE_API_KEY=your_voyage_api_key
VOYAGE_MODEL=voyage-code-3  # recommended

# OpenAI Configuration
OPENAI_BASE_URL=your_openai_base_url
OPENAI_API_KEY=your_api_key
MODEL=your_model_name
```

Use RAG with function-level code chunking:

```bash
cd Experiment/Script/rag_function_chunk
python main.py
```

This method will:
- Parse code into function-level chunks
- Build vector embeddings for code chunks
- Retrieve relevant code context for each question
- Generate answers using retrieved context

### 3. RAG with Sliding Window

Before executing, configure the environment variables by creating a `.env` file in the `Experiment/Script/rag_sliding_window` directory:

```bash
# Voyage AI Configuration
VOYAGE_API_KEY=your_voyage_api_key
VOYAGE_MODEL=voyage-code-3  # recommended

# OpenAI Configuration
OPENAI_URL=your_openai_url
OPENAI_KEY=your_openai_key
MODEL=your_model_name
```

Use RAG with sliding window text chunking:

```bash
cd Experiment/Script/rag_sliding_window
python main.py
```

This method will:
- Split code into overlapping text windows
- Create embeddings for text chunks
- Retrieve relevant chunks for each question
- Generate contextual answers

### 4. SWE-agent QA

For detailed setup and usage instructions, see the [SWE-agent QA README](Experiment/Script/SWE-agent_QA/README.md).

### 5. OpenHands QA

For detailed setup and usage instructions, see the [OpenHands QA README](Experiment/Script/OpenHands_QA/README.md).

### 6. Cursor-Agent QA

Before executing, set the required environment variables:

```bash
export CURSOR_API_KEY="your-cursor-api-key"
export CURSOR_AGENT_PATH="/path/to/cursor-agent"  # Optional
```

For detailed setup and usage instructions, see the [Cursor-Agent QA README](Experiment/Script/Cursor-Agent_QA/README.md).

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

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or issues, please open an issue on GitHub.
