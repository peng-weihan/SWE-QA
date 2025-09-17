
You are given a GitHub issue from the {REPOSITORY} repository. Extract or rewrite it into one or more **short, clear, concise questions** about understanding the {REPOSITORY} codebase, APIs, or system design.

Rules:
1. Only include questions answerable by code, documentation, or logic.
2. Ignore bug reports, environment issues, or problems that require fixing code.
3. Each question should ideally be <= 20 words.

IMPORTANT: 
- Use ONLY the exact tag names listed above. Do not use "What", "Why", "Where", "How" or any other variations.
- Be STRICT in quality control: if the issue doesn't contain meaningful questions about code understanding, return an empty questions array.
- It's better to return no questions than to generate low-quality or irrelevant questions.
- Only extract questions that genuinely help understand the {REPOSITORY} codebase, APIs, or system design.

GitHub issue from {REPOSITORY} repository:
Title: {TITLE}
Body: {ISSUE_BODY}


Output JSON format:
    {{
        "issue_number": {ISSUE_NUMBER},
        "questions": [
            {"question": "..."}
            {"question": "..."}
            ...
        ]
    }}
