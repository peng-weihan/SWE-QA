# Cursor-Agent QA

The experiment uses cursor-agent version: **cursor-agent-2025.09.04-fc40cd1**

python batch_script.py \
  --repo-base-dir /home/ugproj/raymone/swe-repos \
  --question-dir /home/ugproj/raymone/questions \
  --output-dir /home/ugproj/raymone/answer/cursor-agent \
  --model auto \
  --max-concurrency 4 \
  --output-format json \
  --repo-filter sympy \
  --max-tool-calls 50 \
  --force