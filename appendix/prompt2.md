You are an expert software research assistant.

Given:
1. A class description extracted from a software repository.
2. A list of seed questions from the "{CATEGORY}" category.

Task:
1. Based on the seed questions and the class description, generate **one single question** that is:
   - As difficult and complex as possible,
   - Requires multi-hop reasoning or deep technical understanding,
   - Not answerable by simple retrieval or direct lookup (i.e., not solvable by basic RAG methods),
   - Clearly related to the class/module description,
   - Technically precise and detailed,
   - Reflects the style and intent of the original seed questions but goes significantly deeper.
   - **Must not be a compound question** (e.g., no use of "and", "or", or comma-based subquestions),
   - **Must be not too long and syntactically simple**
   - **Must be specific to the "{CATEGORY}" category**

2. The question should encourage advanced analysis, integration of multiple concepts, or insight beyond surface-level information.

3. Output only the single refined question without additional explanation or commentary.


Class Description:
    {Class_DESCRIPTION}

Seed Questions from {CATEGORY}:
    {SEEDS}

Output JSON format:
    {{
        "question": "..."
    }}
  

