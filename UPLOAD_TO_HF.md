# Uploading SWE-QA Dataset to Hugging Face

This guide explains how to upload the SWE-QA benchmark dataset to Hugging Face Hub.

## Prerequisites

1. **Install required packages:**
   ```bash
   pip install huggingface_hub datasets
   ```

2. **Login to Hugging Face:**
   ```bash
   huggingface-cli login
   ```
   You'll be prompted to enter your Hugging Face token. You can get your token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

   Alternatively, you can set the token as an environment variable:
   ```bash
   export HF_TOKEN="your_token_here"
   ```

3. **Create a dataset repository on Hugging Face:**
   - Go to [https://huggingface.co/new-dataset](https://huggingface.co/new-dataset)
   - Create a dataset named "SWE-QA-Benchmark" under your organization/username
   - Or use an existing repository

## Upload Process

### Method 1: Using the Upload Script

1. **Run the upload script:**
   ```bash
   python upload_to_hf.py --benchmark-dir ./Benchmark --repo-id swe-qa/SWE-QA-Benchmark
   ```

2. **Options:**
   - `--benchmark-dir`: Directory containing JSONL files (default: `./Benchmark`)
   - `--repo-id`: Hugging Face dataset repository ID (default: `swe-qa/SWE-QA-Benchmark`)
   - `--private`: Make the dataset private (optional)

### Method 2: Manual Upload Using Python

```python
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
import json
from pathlib import Path

# Load JSONL files
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

# Load all benchmark files
benchmark_dir = Path("./Benchmark")
all_data = []
repo_datasets = {}

for jsonl_file in benchmark_dir.glob("*.jsonl"):
    repo_name = jsonl_file.stem
    data = load_jsonl(jsonl_file)
    all_data.extend(data)
    repo_datasets[repo_name] = Dataset.from_list(data)

# Create dataset with default and per-repo splits
dataset_dict = DatasetDict({
    "default": Dataset.from_list(all_data),
    **repo_datasets
})

# Upload to Hugging Face
dataset_dict.push_to_hub(
    repo_id="swe-qa/SWE-QA-Benchmark",
    private=False,
    commit_message="Upload SWE-QA benchmark dataset"
)
```

### Method 3: Using Hugging Face CLI

1. **Install git-lfs:**
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install git-lfs
   
   # On macOS
   brew install git-lfs
   ```

2. **Clone the dataset repository:**
   ```bash
   git clone https://huggingface.co/datasets/swe-qa/SWE-QA-Benchmark
   cd SWE-QA-Benchmark
   git lfs install
   ```

3. **Copy JSONL files to the repository:**
   ```bash
   cp ../Benchmark/*.jsonl .
   ```

4. **Create a dataset loading script** (`SWE_QA_Benchmark.py`):
   ```python
   import json
   from datasets import Dataset, DatasetDict
   from pathlib import Path

   def load_jsonl(file_path):
       data = []
       with open(file_path, 'r', encoding='utf-8') as f:
           for line in f:
               if line.strip():
                   data.append(json.loads(line))
       return data

   def load_dataset():
       benchmark_dir = Path(__file__).parent
       all_data = []
       repo_datasets = {}
       
       for jsonl_file in benchmark_dir.glob("*.jsonl"):
           repo_name = jsonl_file.stem
           data = load_jsonl(jsonl_file)
           all_data.extend(data)
           repo_datasets[repo_name] = Dataset.from_list(data)
       
       return DatasetDict({
           "default": Dataset.from_list(all_data),
           **repo_datasets
       })
   ```

5. **Commit and push:**
   ```bash
   git add .
   git commit -m "Add SWE-QA benchmark dataset"
   git push
   ```

## Dataset Structure

The uploaded dataset will have the following structure:

- **Default split**: All questions combined (576 questions)
- **Per-repository splits**: Individual splits for each repository (48 questions each)
  - `astropy`
  - `django`
  - `flask`
  - `matplotlib`
  - `pylint`
  - `pytest`
  - `requests`
  - `scikit-learn`
  - `sphinx`
  - `sqlfluff`
  - `sympy`
  - `xarray`

## Verification

After uploading, verify the dataset:

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("swe-qa/SWE-QA-Benchmark")

# Check default split
print(f"Total questions: {len(dataset['default'])}")

# Check a specific repository
print(f"Flask questions: {len(dataset['flask'])}")
print(dataset['flask'][0])
```

## Troubleshooting

1. **Authentication errors:**
   - Make sure you're logged in: `huggingface-cli login`
   - Check your token has write permissions

2. **Repository not found:**
   - Create the repository on Hugging Face first
   - Check the repository ID is correct (format: `username/dataset-name`)

3. **Large file uploads:**
   - Hugging Face automatically uses Git LFS for large files
   - Make sure git-lfs is installed and initialized

4. **Permission errors:**
   - Ensure you have write access to the repository
   - For organization repositories, check your role/permissions

## Updating the Dataset

To update an existing dataset:

```bash
python upload_to_hf.py --benchmark-dir ./Benchmark --repo-id swe-qa/SWE-QA-Benchmark
```

The script will automatically push updates to the existing repository.

