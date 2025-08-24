# src/prompts.py
from langchain.prompts import ChatPromptTemplate

ATS_PROMPT = ChatPromptTemplate.from_template("""
Using only the CONTEXT, draft a 1-page ATS-friendly resume (Markdown).
- Bullet-heavy, keyword-rich, tailored to JD.
- Cite inline with [S#] when referencing experience.
CONTEXT:
{context}
End with: Sources mapping [S#].
""".strip())
# ... similarly for COVER_LETTER_PROMPT, EMAIL_PROMPT, DM_PROMPT
