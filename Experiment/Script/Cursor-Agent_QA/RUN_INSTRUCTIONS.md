# 运行 batch_script.py 说明

## 依赖安装

```bash
pip install pydantic python-dotenv rich tqdm
```

## 配置方式

### 方式1：使用配置文件 (推荐)

创建 `config.json` 文件：

```json
{
  "repo_base_dir": "/path/to/repositories",
  "question_dir": "/path/to/questions",
  "output_dir": "/path/to/output",
  "model": "auto",
  "max_concurrency": 4,
  "output_format": "json",
  "cursor_agent_path": "/home/ugproj/.local/share/cursor-agent/versions/2025.11.25-d5b3271/cursor-agent"
}
```

然后运行：
```bash
python batch_script.py
```

### 方式2：使用命令行参数

```bash
python batch_script.py \
  --repo-base-dir /path/to/repositories \
  --question-dir /path/to/questions \
  --output-dir /path/to/output \
  --model auto \
  --max-concurrency 4 \
  --output-format json
```

## 参数说明

- `--config`: 配置文件路径（默认: `config.json`）
- `--repo-base-dir`: 仓库根目录路径
- `--question-dir`: 包含问题文件（.jsonl格式）的目录
- `--output-dir`: 输出结果目录
- `--model`: 使用的模型（默认: `auto`）
- `--max-concurrency`: 最大并发数（默认: 4）
- `--repo-filter`: 只处理特定仓库（例如: `--repo-filter conan`）
- `--output-format`: 输出格式，可选 `text`, `json`, `stream-json`（默认: `json`）
- `--cursor-agent-path`: cursor-agent 可执行文件的完整路径

## 问题文件格式

问题文件应为 `.jsonl` 格式，每行一个 JSON 对象：

```json
{"question": "问题内容", "ground_truth": "标准答案"}
{"question": "另一个问题", "ground_truth": "另一个答案"}
```

文件名格式：`{repo_name}.jsonl`（例如: `conan.jsonl`）

## 输出格式

输出文件为 `{repo_name}.jsonl`，每行包含一个结果：
- `question`: 问题
- `answer`: 回答
- `trajectory`: 完整轨迹
- `latency`: 延迟（秒）
- `input_tokens`: 输入token数
- `output_tokens`: 输出token数
- `total_tokens`: 总token数

## 示例

```bash
# 处理所有仓库
python batch_script.py --config config.json

# 只处理特定仓库
python batch_script.py --repo-filter conan --question-dir ./questions --output-dir ./results

# 自定义并发数
python batch_script.py --max-concurrency 8 --question-dir ./questions --output-dir ./results
```

## 注意事项

- 脚本会自动跳过已处理的问题（基于输出文件）
- 需要确保 cursor-agent 已安装并在 PATH 中，或通过 `--cursor-agent-path` 指定
- API key 已硬编码在脚本中（第311行），如需修改请编辑脚本

