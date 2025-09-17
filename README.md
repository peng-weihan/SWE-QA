# SWE-QA

**SWE-QA: Can Language Models Answer Repository-level Code Questions?**

This repository contains code and data for the SWE-QA benchmark, which evaluates language models' ability to answer repository-level code questions.

## 📝 Prompts

The detailed prompt templates used in the paper are in the `appendix/` directory.

## 📊 Dataset

The benchmark dataset is available on Hugging Face:
- **Dataset**: [SWE-QA-Benchmark](https://huggingface.co/datasets/swe-qa/SWE-QA-Benchmark)

## 📖 Paper

For more details about the methodology and results, please refer to the paper:
- **Paper**: "SWE-QA: Can Language Models Answer Repository-level Code Questions?"

## 📁 Repository Structure

```
SWE-QA/
├── appendix/                  # Supplementary materials
│   ├── prompt.pdf            # Prompt   
│   ├── prompt1.md            # Issue extraction prompt template
│   ├── prompt2.md            # Question generation prompt template
│   └── prompt3.md            # Answer evaluation prompt template
├── SWE-QA/                    # Main package directory
│   ├── datasets/              # Dataset files and repositories
│   │   ├── questions/         # Question datasets (JSONL format)
│   │   │   ├── astropy.jsonl  # Project-specific datasets
│   │   │   ├── django.jsonl
│   │   │   ├── flask.jsonl
│   │   │   ├── matplotlib.jsonl
│   │   │   ├── pylint.jsonl
│   │   │   ├── pytest.jsonl
│   │   │   ├── requests.jsonl
│   │   │   ├── scikit-learn.jsonl
│   │   │   ├── sphinx.jsonl
│   │   │   ├── sqlfluff.jsonl
│   │   │   ├── sympy.jsonl
│   │   │   └── xarray.jsonl
│   │   ├── answers/           # Answer datasets
│   │   ├── faiss/             # FAISS index files
│   │   └── repos/             # Repository data
│   ├── issue_analyzer/        # GitHub issue analysis
│   │   ├── get_question_from_issue.py
│   │   └── pull_issues.py
│   ├── methods/               # Evaluation methods
│   │   ├── llm_direct/        # Direct LLM evaluation
│   │   ├── rag_function_chunk/ # RAG with function chunking
│   │   ├── rag_sliding_window/ # RAG with sliding window
│   │   ├── utils/             # Agent-based evaluation
│   │   │   ├── agent.py       # Main agent implementation
│   │   │   ├── config.py      # Configuration utilities
│   │   │   ├── history.py     # Conversation history management
│   │   │   ├── prompts/       # Agent prompts
│   │   │   │   └── react_prompt.txt
│   │   │   └── tools/         # Agent tools
│   │   │       ├── repo_rag.py
│   │   │       └── repo_read.py
│   │   ├── code_formatting.py
│   │   └── data_models.py
│   ├── qa_generator/          # Question-answer generation
│   │   ├── core/
│   │   │   └── generator.py
│   │   ├── generate_question.py
│   │   └── qa_generator.py
│   ├── repo_parser/           # Repository parsing utilities
│   │   ├── parse_repo.py
│   │   └── repo_parser.py
│   ├── score/                 # Scoring utilities
│   │   └── llm-score.py       # LLM-as-a-judge evaluation
│   ├── models/                # Data models
│   │   └── data_models.py
│   └── utils/                 # Utility functions
├── docs/                      # Documentation
│   └── README.md
├── LICENSE                    # License file
└── README.md                  # This file

## 📝 Citation

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.