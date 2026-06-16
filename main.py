from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from tavily import TavilyClient


from dotenv import load_dotenv

load_dotenv()

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    '''
        Tool that searches over the internet

        Args:
            query (str): The query to search for

        Returns:
            str: The search results
    '''

    print(f"Searching for: {query}")
    return tavily.search(query=query)

llm = ChatOpenAI(model="gpt-5.5")
tools = [search]
agent = create_agent(model=llm, tools=tools)

def main():
    result = agent.invoke(
        {"messages": [HumanMessage(content="What's the weather like in Tokyo?")]}
    )
    print(result)


if __name__ == "__main__":
    main()
