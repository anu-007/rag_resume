from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate

score_prompt = PromptTemplate(
    input_variables=["jd", "resume"],
    template="""
You are an ATS software. Compare the Job Description and Resume.

Job Description:
{jd}

Resume Section:
{resume}

Tasks:
1. Give an ATS Match Score (0–100).
2. List missing skills, frameworks, or keywords in the resume.
3. Suggest improvements in phrasing and alignment.
Output in a structured JSON format with keys: score, missing, improvements.
"""
)                                   

rewrite_prompt = PromptTemplate(
    input_variables=["jd", "resume"],
    template="""
You are a resume optimization assistant.
Rewrite the following resume to maximize ATS score for this Job Description.

Job Description:
{jd}

Resume:
{resume}

Output a new optimized resume with:
- Professional Summary (ATS friendly, highlighting Python, APIs, cloud, testing).
- Skills (aligned with JD, including Python, FastAPI, Flask, Django, Pytest, REST APIs, Docker, AWS).
- Work Experience (rewrite bullets to explicitly show alignment with JD).
- Education
- Achievements

Keep resume concise (max 2 pages).
"""
)

COVER_LETTER_PROMPT = ChatPromptTemplate.from_template("""
You are a professional job application assistant.  
Write a concise, tailored cover letter for the given job description, company information, and candidate professional summary.  

Context:
{context}

Guidelines:
- Keep it professional yet human-like.  
- Start with why the candidate is excited about THIS company + THIS role.  
- Highlight 2–3 key experiences/skills that match the job description.  
- Keep length between 200–250 words (one short page).  
- Avoid fluff, repetition, and generic phrases.  
- End with a polite call-to-action (e.g., looking forward to discussing further).  

Now, generate the tailored cover letter.

""".strip())

EMAIL_PROMPT = ChatPromptTemplate.from_template("""
You are helping a candidate reach out to a recruiter.  
Write a short, professional email tailored to the given job description, company information, and candidate professional summary.  

Context:
{context}

Guidelines:
- Keep it under 150 words.  
- Use a polite and professional tone.  
- Mention the role title explicitly.  
- Briefly connect candidate’s background to the role (1–2 sentences).  
- Attach that resume and cover letter (assume they are generated separately).  
- End with a call-to-action (e.g., happy to share more details, looking forward to connecting).  

Now, generate the recruiter email.
""".strip())

DM_PROMPT = ChatPromptTemplate.from_template("""
You are helping a candidate write a concise LinkedIn DM to a recruiter or hiring manager.  
The goal is to express interest in the role without sounding overly formal.  

Context:
{context}

Guidelines:
- Keep it under 80 words.  
- Use a friendly and professional tone.  
- Mention the role title directly.  
- Briefly (1 sentence) show why candidate is a strong fit.  
- End with a soft call-to-action (e.g., “Would love to connect” or “Happy to share more details”).  

Now, generate the LinkedIn DM.

""".strip())