from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from SAR.prompt import SAR_PROMPT
from dotenv import load_dotenv

load_dotenv()

def build_sar_chain():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2
    )
    chain = SAR_PROMPT | llm | StrOutputParser()
    
    return chain
