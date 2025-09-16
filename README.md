# SWE-QA

**SWE-QA: Can Language Models Answer Repository-level Code Questions?**

This repository contains code and data for the SWE-QA benchmark, which evaluates language models' ability to answer repository-level code questions.

## 📊 Dataset

The benchmark dataset is available on Hugging Face:
- **Dataset**: [SWE-QA-Benchmark](https://huggingface.co/datasets/swe-qa/SWE-QA-Benchmark)

## 📖 Paper

For more details about the methodology and results, please refer to the paper:
- **Paper**: "SWE-QA: Can Language Models Answer Repository-level Code Questions?"

## 📁 Repository Structure

```
SWE-QA/
├── SWE-QA/                    # Main package directory
│   ├── datasets/              # Dataset files and repositories
│   │   ├── questions/         # Question datasets (JSONL format)
│   │   │   └── *.jsonl        # 12 project datasets
│   │   ├── answers/           # Answer datasets
│   │   ├── faiss/             # FAISS index files
│   │   └── repos/             # Repository data
│   ├── methods/               # Evaluation methods
│   │   ├── llm_direct/        # Direct LLM evaluation
│   │   ├── rag_function_chunk/ # RAG with function chunking
│   │   ├── rag_sliding_window/ # RAG with sliding window
│   │   ├── swe_qa_agent/      # SWE-QA agent implementation
│   │   │   ├── tools/         # Agent tools
│   │   │   └── prompts/       # Agent prompts
│   │   ├── code_formatting.py
│   │   └── data_models.py
│   ├── qa_generator/          # Question-answer generation
│   │   ├── core/
│   │   ├── generate_question.py
│   │   └── qa_generator.py
│   ├── repo_parser/           # Repository parsing utilities
│   │   ├── parse_repo.py
│   │   └── repo_parser.py
│   ├── score/                 # Scoring utilities
│   │   └── llm-score.py       # LLM-as-a-judge Evaluation
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