import os
from typing import TypedDict, Annotated
from langgraph. graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

#tools

search_tool = TavilySearch(max_results = 3)

tools = [search_tool]

#llms

writer_llm = ChatGoogleGenerativeAI(model = "chat-bison-001", temperature = 0.7)
writer_llm_with_tools = writer_llm.bind_tools(tools)

#reviewer

reviewer_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

#state building

class State(TypedDict):
    topic : str
    messages : Annotated [list, add_messages]
    draft : str
    review_feedback : str
    is_approved : bool
    attempt : int

#nodes 

WRITER_SYSTEM_PROMPT = (
    "You are an expert LinkedIn content writer. Your job is to write "
    "engaging, professional LinkedIn posts about the given topic. "
    "If the topic requires up-to-date information, statistics, or "
    "current trends, use the web search tool to gather fresh context "
    "before writing. If you have already received feedback on a "
    "previous draft, carefully address every point in the new draft. "
    "Rules for good LinkedIn posts: strong hook in the first line, "
    "1 clear takeaway, easy to skim (short paragraphs), around "
    "150–200 words, ends with a question or call-to-action to invite "
    "engagement. Do not use hashtags."
)


def writer_node(state : State) -> dict:
    """Writes (or rewrites) the LinkedIn post. Can call Tavily to search first."""
    attempt = state.get("attempt",0) + 1 
    topic = state["topic"]
    previous_feedback = state['review_feedback']

    if attempt == 1:
        user_message = (
            f"Write a LinkedIn post on this topic {topic}"
            f"if you need current info search the web first "
        )
    else:
        user_message = (
            f"your previous draft on '{topic}' was rejected"
            f"Here is the reviewer's feedback \n\n {previous_feedback}\n\n"
            f"Write a new, improved draft that fixes every issue mentiond"
            f"do not repeat the same mistake"
        )
    messages = [("system",WRITER_SYSTEM_PROMPT),("human",user_message)]
    response = writer_llm_with_tools.invoke(messages)

    return {
        "messages" : [("human",user_message),response],
        "attempt" : attempt
    }

tool_node = ToolNode(tools)