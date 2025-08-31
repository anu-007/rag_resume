import json
import os
from .prompt import ENTITY_EXTRACTION_PROMPT
from langchain_openai import ChatOpenAI

def extract_entities(jd_text, model="gpt-3.5-turbo"):
    llm = ChatOpenAI(model=model, temperature=0, api_key=os.environ.get("OPENAI_API_KEY"))
    prompt = ENTITY_EXTRACTION_PROMPT.format_messages(jd_text=jd_text)
    response = llm(prompt)
    try:
        entities = json.loads(response.content)
    except Exception:
        entities = response.content
    return entities