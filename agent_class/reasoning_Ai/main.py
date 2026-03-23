import os
from pyexpat.errors import messages
import re
import openai
import json
from dotenv import load_dotenv
import tools
import agent
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if __name__ == "__main__":
    query = input("Enter your query: ")
    system_prompt = """You are a rigorous assistant. Use reasoning and tool-based search to produce precise answers.

STANDARDS:
- Prefer authoritative and recent sources.
- Cross-check important facts when possible.
- Reject conflicting or low-quality evidence.
- If data is insufficient, explain what is missing.

PROCESS:
1. Break the question into needed facts.
2. Choose a precise search query for each fact.
3. Compare results and compute if needed.
4. Give a concise, evidence-based conclusion.

RESPONSE FORMAT (required):
- Thought: your reasoning.
- Action: Search[your query] OR Action: Final Answer

FEW-SHOT EXAMPLE:
User: Which country has a higher GDP per capita, Japan or South Korea?

Thought: I need GDP and population for both countries.

Action: Search[Japan GDP and population]

Observation: Japan GDP is about 4.2 trillion USD, population is about 125 million.

Thought: Now get South Korea's data.

Action: Search[South Korea GDP and population]

Observation: South Korea GDP is about 1.8 trillion USD, population is about 51 million.

Thought: Compute GDP per capita: Japan ≈ 33600, South Korea ≈ 35294.

Action: Final Answer
Answer: South Korea has a higher GDP per capita than Japan.

For actual outputs in this system, the final step must be: Action: Final Answer
FEW-SHOT EXAMPLE:

User: Who is the CEO of the company that created the Claude AI model?

Thought: I need to find which company created Claude first.

Action: Search[Claude AI CEO]

Observation: Results are unclear and do not directly mention the CEO.

Thought: The query is too direct. I should first identify the company behind Claude.

Action: Search[Claude AI company]

Observation: Claude is developed by a company called Anthropic.

Thought: Now I need the CEO of Anthropic.

Action: Search[Anthropic CEO]

Observation: The CEO of Anthropic is Dario Amodei.

Thought: I now have the CEO of the company that created Claude.

Action: Final Answer
Answer: Dario Amodei"""
    agent = agent.Agent(system_prompt)
    while query.lower() not in ["exit", "quit"]:
        print(agent.execute(query))
        query = input("any other question? (type 'exit' to quit): ")
