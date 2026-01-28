from langchain_core.prompts import PromptTemplate

SAR_PROMPT = PromptTemplate(
    input_variables=["case_summary"],
    template="""
You are a financial crime compliance analyst.

Using only the factual information below, write a Suspicious Activity Report narrative.
Do not accuse the subject of wrongdoing.
Do not speculate beyond the provided facts.
Use formal regulatory language.

Facts:
{case_summary}

SAR Narrative:
"""
)
