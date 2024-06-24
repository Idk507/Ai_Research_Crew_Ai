import os
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
from langchain.tools import DuckDuckGoSearchRun

# Set up Gemini Pro as the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    verbose=True,
    temperature=0.5,
    google_api_key=""
)

# Create search tool
search_tool = DuckDuckGoSearchRun()

# Define Agents
researcher_agent = Agent(
    role="Senior Research Agent",
    goal="Uncover cutting-edge developments in AI and Data Science",
    backstory="""
You are an expert at a technology research group,
skilled in identifying trends and analyzing complex data""",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=llm
)

writer_agent = Agent(
    role="Tech Content Writer",
    goal="Create compelling content on tech advancements",
    backstory="""
You are a content strategist known for making complex tech topics interesting and easy to 
understand and analyze""",
    verbose=True,
    allow_delegation=True,
    llm=llm
)

# Define Tasks
task1 = Task(
    description="""
    Analyze 2024's AI advancements. Find major trends, new technologies, and their implications.
    """,
    expected_output="A detailed analysis of 2024's AI advancements including trends and new technologies.",
    agent=researcher_agent
)

task2 = Task(
    description="""
    Create a blog post about major AI advancements using insights. Make it interesting and clear.
    """,
    expected_output="A compelling blog post that is interesting, clear, and easy to understand.",
    agent=writer_agent
)

# Create a Crew
crew = Crew(
    agents=[researcher_agent, writer_agent],
    tasks=[task1, task2],
    verbose=True,
    process=Process.sequential
)

# Execute the workflow
print("Crew: Working on AI Advancements Task")
result = crew.kickoff()

print("**********")
print(result)
