from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from dotenv import load_dotenv
import os
load_dotenv()

os.environ.pop("GOOGLE_API_KEY", None)

# Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

#1stAgent
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt="""You are a web research agent.

Your job is to research the user's topic using the web_search tool.

IMPORTANT:
1. Always use the web_search tool.
2. Search for recent and reliable information.
3. Return the TITLE, URL and SNIPPET for each useful source.
4. Do not hide or remove URLs.
5. Do not replace the search results with your own summary.
"""
        
    )

#2ndAgent
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt="""You are a web reading agent.

You receive search results containing URLs.

Your job is:
1. Identify the most relevant URL.
2. Call the scrape_url tool using that URL.
3. Read the scraped content.
4. Return the scraped content.

IMPORTANT:
You MUST actually call scrape_url.
Do not simply say that a URL was not provided if a URL exists in the search results.
"""
    )

#writerChain
writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research writer. "
        "Write clear, structured, insightful, factual and professional reports."
    ),
    (
        "human",
        """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:

# Introduction

# Key Findings
Provide at least 3 well-explained findings.

# Challenges / Limitations

# Conclusion

# Sources
List the actual URLs found in the research.

IMPORTANT RULES:
- Do not invent URLs.
- Only include URLs that actually appear in the research.
- Do not create fake citations.
- Use the scraped content when explaining detailed findings.
- Use the search results to provide additional context.
- Clearly distinguish factual information from unsupported claims.
- Be detailed, factual and professional.
"""
    ),
])

writer_chain = writer_prompt | llm | StrOutputParser()

#criticChain

critic_prompt = ChatPromptTemplate.from_messages([
    ("system","You are a sharp and constructive research critic. Be honest and specific."),
    ("human","""Review the research report below and evaluate it strictly.
     Report:
     {report}
     Respond in this exact format:
     
     Score: X/10
     
     Strengths:
     - ...
     - ...
     
     Areas to Improve:
     - ...
     - ...
     
     Onel line verdict:
     ..."""),
])

critic_chain = critic_prompt |llm|StrOutputParser()