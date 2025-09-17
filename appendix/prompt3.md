You are a professional evaluator. Please rate the candidate answer against the reference answer based on five criteria.
Evaluation Criteria and Scoring Guidelines (each scored 0 to 10):
    1. Correctness:
        9-10 — Completely correct; core points and details are accurate with no ambiguity.
        7-8 — Mostly correct; only minor details are slightly inaccurate or loosely expressed.
        5-6 — Partially correct; some errors or omissions, but main points are generally accurate.
        3-4 — Several errors or ambiguities that affect understanding of the core information.
        0-2 — Serious errors; misleading or fails to convey key information.
    2. Completeness:
        9-10 — Covers all key points from the reference answer without omission.
        7-8 — Covers most key points; only minor non-critical information missing.
        5-6 — Missing several key points; content is somewhat incomplete.
        3-4 — Important information largely missing; content is one-sided.
        0-2 — Covers very little or irrelevant information; seriously incomplete.
    3. Relevance:
        9-10 — Content fully focused on the question topic; no irrelevant information.
        7-8 — Mostly focused; only minor irrelevant or peripheral information.
        5-6 — Topic not sufficiently focused; contains considerable off-topic content.
        3-4 — Content deviates from topic; includes excessive irrelevant information.
        0-2 — Majority of content irrelevant to the question.
    4. Clarity:
        9-10 — Fluent language; clear and precise expression; easy to understand.
        7-8 — Mostly fluent; some expressions slightly unclear or not concise.
        5-6 — Expression somewhat awkward; some ambiguity or lack of fluency.
        3-4 — Language obscure; sentences are not smooth; hinders understanding.
        0-2 — Expression confusing; very difficult to understand.
    5. Reasoning:
        9-10 — Reasoning is clear, logical, and well-structured; argumentation is solid.
        7-8 — Reasoning generally reasonable; mostly clear logic; minor jumps.
        5-6 — Reasoning is average; some logical jumps or organization issues.
        3-4 — Reasoning unclear; lacks logical order; difficult to follow.
        0-2 — No clear reasoning; logic is chaotic.
INPUT:
    Question:{question}
    Reference Answer:{reference}
    Candidate Answer:{candidate}

Please output ONLY a JSON object with 5 integer fields in the range [0,10], corresponding to the evaluation scores:
Output JSON format:
    {{
        "correctness": <0-10>,
        "completeness": <0-10>,
        "relevance": <0-10>,
        "clarity": <0-10>,
        "reasoning": <0-10>
    }}

REQUIREMENT:
No explanation, no extra text, no formatting other than valid JSON