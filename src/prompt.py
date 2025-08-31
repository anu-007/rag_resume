from langchain.prompts import ChatPromptTemplate

ENTITY_EXTRACTION_PROMPT = ChatPromptTemplate.from_template("""
You are an expert HR analyst. Your task is to extract key information from the following job description.

From the text provided, please extract the following entities:
- "technical_skills": Specific programming languages, software, or technical abilities mentioned.
- "soft_skills": Non-technical skills like communication, teamwork, etc.
- "required_qualifications": Education, certifications, or years of experience required.
- "primary_responsibilities": The main duties and tasks of the job.

Return the information in a JSON format, with each key containing a list of the extracted strings.

Example:
{{
  "technical_skills": ["Python", "SQL", "Pandas"],
  "soft_skills": ["Communication", "Teamwork"],
  "required_qualifications": ["Bachelor's degree in Computer Science", "3+ years of experience"],
  "primary_responsibilities": ["Develop and maintain data pipelines", "Collaborate with cross-functional teams"]
}}

Job Description:
{jd_text}
""".strip())

ATS_PROMPT = ChatPromptTemplate.from_template("""
You are an expert career coach and professional resume writer specializing in technology roles. 
Your task is to update my resume by keeping the structure and formatting of my existing markdown template, 
but rewriting and populating the **Summary**, **Work Experience**, and **Skills** sections using the provided context and job description.

**My Relevant Experience (Retrieved Context):**
{context}

**Existing Resume Template:**
# Anubhuti Verma

Abu Dhabi, UAE | +971 508224911 | anubhutiv1@gmail.com | anuworks.com | LinkedIn | GitHub
[SUMMARY]

## WORK EXPERIENCE

**Senior Software Engineer** (2019 - Present)
SenseHawk (Acquired by Reliance), Abu Dhabi, UAE
[WORK EXPERIENCE]
**Backend Developer** (2018 - 2019)
Shanghari Global, New Delhi, India
● Developed blockchain tokenization platform (“Reneum”), processing 1.2 million daily power
transactions and enabling transparent renewable energy trading.

## SKILLS
[SKILLS]

## EDUCATION

B.Tech, Computer Science and Engineering - Ajay Kumar Garg Engineering College, AKTU (GPA: 9/10)

## ACHIEVEMENTS

● Google Summer of Code 2017 – FossAsia: Developed frontend modules for Open Event platform.
● Google India Challenge Scholar (Mobile Web): Selected for innovative web mobile solutions
development.

---

### Instructions:
1. **Keep the format and structure** of the provided resume template exactly the same (markdown headings, sections, order).
2. **Update only these sections**:
   - **Summary** → Rewrite to highlight alignment with the target job description, incorporating role-specific keywords naturally. Keep it concise (3–4 lines max).
   - **Work Experience** → Rewrite or replace bullet points under each role using the retrieved context.  
     - Use **STAR method** where applicable.  
     - Prioritize quantifiable achievements (%/$/time saved/scale).  
     - Seamlessly integrate keywords from the job description.  
     - Each bullet must start with a **strong action verb**.
   - **Skills** → Reorder and update to emphasize those most relevant to the job description. Remove irrelevant ones if necessary, but keep concise.
3. Do **not alter**: Contact Info, Education, Achievements, or overall section order/format.
4. Ensure the output is **ATS-friendly, concise (1 page), and professional**.
5. The final output must be a **complete updated resume in markdown** format, with no extra commentary.

---

### Output:
Return the **full updated resume in markdown**.
""".strip())

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