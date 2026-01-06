#!/usr/bin/env python3
"""
批量重命名 JSONL 文件中的字段：
1. 原来的 question 字段改名为 origin_question
2. rewritten_question 字段改名为 question 并放到第一个字段位置
"""

import json
import os
from pathlib import Path

def process_jsonl_file(file_path):
    """处理单个 JSONL 文件"""
    output_lines = []
    modified = False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                file_modified = False
                
                # 处理逻辑：
                # 1. 如果存在 question 字段，且不存在 origin_question，将其改名为 origin_question
                # 2. 如果存在 rewritten_question 字段，将其改名为 question 并放到第一个位置
                # 3. 如果不存在 rewritten_question，但 question 已经是重写后的问题（且存在 origin_question），保持不变
                
                # 步骤1: 处理原始 question -> origin_question
                if 'question' in data and 'origin_question' not in data:
                    # 将 question 改名为 origin_question
                    data['origin_question'] = data.pop('question')
                    file_modified = True
                
                # 步骤2: 处理 rewritten_question -> question（放到第一个位置）
                if 'rewritten_question' in data:
                    rewritten_value = data.pop('rewritten_question')
                    # 创建新字典，question 放在第一个位置
                    new_data = {'question': rewritten_value}
                    # 添加其他字段
                    for key, value in data.items():
                        new_data[key] = value
                    data = new_data
                    file_modified = True
                elif 'question' not in data and 'origin_question' in data:
                    # 如果只有 origin_question 没有 question，说明 question 已经被改名为 origin_question
                    # 这种情况下，如果没有 rewritten_question，可能需要从其他地方获取
                    # 但根据当前文件结构，这种情况应该不会发生
                    pass
                
                # 确保 question 字段在第一个位置（如果存在）
                if 'question' in data:
                    question_value = data.pop('question')
                    new_data = {'question': question_value}
                    new_data.update(data)
                    data = new_data
                    file_modified = True
                
                if file_modified:
                    modified = True
                
                output_lines.append(json.dumps(data, ensure_ascii=False))
                
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line in {file_path}: {e}")
                output_lines.append(line)  # 保留原始行
    
    # 如果文件被修改，写回文件
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in output_lines:
                f.write(line + '\n')
        return True
    
    return False

def main():
    base_dir = Path("/home/ugproj/raymone/GIT_workspace/ICLR/DeepRepoQA/Script/Cursor-Agent_QA/unidentified_question")
    
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return
    
    jsonl_files = list(base_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"No JSONL files found in {base_dir}")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files to process")
    
    for file_path in jsonl_files:
        print(f"Processing {file_path.name}...")
        try:
            modified = process_jsonl_file(file_path)
            if modified:
                print(f"  ✓ Modified {file_path.name}")
            else:
                print(f"  - No changes needed for {file_path.name}")
        except Exception as e:
            print(f"  ✗ Error processing {file_path.name}: {e}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()








