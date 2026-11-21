import os 
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages 
from langgraph.graph import StateGraph , START , END 
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv 
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

load_dotenv()


#Step 1 - Building the RAG retrievers 

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2" )



def build_retriver(pdf_path: str):

    pdf_path = BASE_DIR / pdf_path

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    documents = PyPDFLoader(str(pdf_path)).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError(f"No chunks generated from {pdf_path.name}")

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )


def build_fee_vectorstore(pdf_path: str):

    pdf_path = BASE_DIR / pdf_path

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    documents = PyPDFLoader(str(pdf_path)).load()

    for doc in documents:

        content = doc.page_content.lower()

        if "b.pharm" in content:
            programme_name = "B.Pharm"
        elif "b.tech" in content or "btech" in content:
            programme_name = "B.Tech"
        elif "bba" in content:
            programme_name = "BBA"
        elif "bca" in content:
            programme_name = "BCA"
        else:
            programme_name = "Other"

        if "lateral entry" in content:
            entry_type = "lateral"
        elif "1st year" in content or "first year" in content:
            entry_type = "first_year"
        else:
            entry_type = "other"

        doc.metadata["programme"] = programme_name
        doc.metadata["entry_type"] = entry_type

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No fee document chunks generated.")

    return FAISS.from_documents(chunks, embeddings)


# Academic and Fee Vectorstores

academic_retriever = build_retriver(
    "ABESIT Brochure 2026.pdf"
)

fee_vectorstore = build_fee_vectorstore(
    "ABESIT Fee Structure and Eligibility 2026-2027.pdf"
)

fee_retriever = fee_vectorstore.as_retriever(
    search_kwargs={"k": 10}
)


fee_documents = PyPDFLoader(
    str(BASE_DIR / "ABESIT Fee Structure and Eligibility 2026-2027.pdf")
).load()


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.4
)

#step2 - State 

class State(TypedDict):
    programme : str  
    messages : Annotated[list,add_messages]
    query_type : str 
    retrieved_context : str 

#Step 3 - Nodes generation 
    
def classifier_node(state : State) -> dict:
    """Look at the latest user message and decide which path to take."""

    last_message = state['messages'][-1].content

    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"
        "Use 'academic' for questions about attendance, exams, grading, credits, "
        "promotion, course structure, summer training, or degree requirements.\n"
        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic.\n"
        "Use 'general' for greetings, casual talk, or anything not related to "
        "the college rules or fee.\n\n"
        f"Query: {last_message}\n\n"
        "Return only one word: academic, fee, or general."
    )

    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    if "academic" in category:
        category = "academic"
    elif "fee" in category:
        category = "fee"
    else:
        category = "general"
    
    return {"query_type" : category}


def academic_rag_node(state: State) -> dict:

    programme = state["programme"]
    query = state["messages"][-1].content

    retrieval_query = (
        f"ABESIT academic information for {programme} student. "
        f"Find relevant details about courses, curriculum, "
        f"attendance, examinations, credits, training, "
        f"eligibility, degree requirements and college rules.\n"
        f"Student Question: {query}"
    )

    docs = academic_retriever.invoke(retrieval_query)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    return {"retrieved_context": context}



def fee_rag_node(state: State) -> dict:

    programme = state["programme"]
    query = state["messages"][-1].content

    query_lower = query.lower()

    if "lateral" in query_lower:
        entry_type = "lateral"

    elif any(word in query_lower for word in [
        "first year",
        "1st year",
        "regular",
        "not for lateral",
        "not lateral"
    ]):
        entry_type = "first_year"

    elif "lateral" in programme.lower():
        entry_type = "lateral"

    else:
        entry_type = "first_year"

    if "b.pharm" in programme.lower():
        programme_name = "B.Pharm"
    elif "b.tech" in programme.lower():
        programme_name = "B.Tech"
    else:
        programme_name = programme

    retrieval_query = (
        f"{programme_name} {entry_type} fee structure "
        f"academic session 2026-2027. "
        f"Tuition, registration, student activity, training, "
        f"entrepreneurial, examination, uniform, ERP, insurance, "
        f"security and admission charges. "
        f"Question: {query}"
    )

    docs = fee_vectorstore.similarity_search(
        retrieval_query,
        k=10,
        filter={
            "programme": programme_name,
            "entry_type": entry_type
        }
    )

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    return {
        "retrieved_context": context
        if context else "No matching fee information found."
    }


def general_node(state: State) -> dict:
    """Answers directly using the LLM's own knowledge, no retrieval needed."""
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}


def response_node(state: State) -> dict:
    """Generates the final answer, personalized using the student's programme."""
    query = state["messages"][-1].content
    programme = state.get("programme", "Unknown")
    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":
        prompt = (
            f"You are a friendly college assistant talking to a {programme} student. "
            f"Answer this question using your own general knowledge:\n\n{query}"
        )
    else:
        
     prompt = (
    f"You are an official ABESIT College Assistant helping a {programme} student.\n\n"

    f"STRICT INSTRUCTIONS:\n"
    f"1. Use ONLY the official document context provided below.\n"
    f"2. Never invent, assume, modify, or guess any information or fee amount.\n"
    f"3. Carefully inspect the ENTIRE retrieved context before answering.\n"
    f"4. For fee-related questions, identify and include EVERY relevant fee head "
    f"available in the retrieved context.\n"
    f"5. Do not skip Tuition Fee, Registration & Documentation Fee, "
    f"Student Activity Fee, Training & Personality Development Fee, "
    f"Entrepreneurial Skill Development Fee, Enrollment & Examination Fees, "
    f"Uniform Fabric, ERP Services, Medical Insurance, Academic Security, "
    f"Admission Process & Documentation Fees, or any other applicable charge.\n"
    f"6. Include Tuition Fee whenever it is available, even if it is subject "
    f"to change. Mention the applicable fee-change condition.\n"
    f"7. Clearly distinguish annual, one-time, refundable, non-refundable, "
    f"and directly payable charges wherever specified.\n"
    f"8. If different amounts are given for Counselling, Direct/Vacant, "
    f"or Fee Waiver categories, show them separately.\n"
    f"9. Use ONLY the fee structure belonging to {programme}. "
    f"Never mix figures from another programme.\n"
    f"10. Include all relevant additional charges, conditions, and payment details.\n"
    f"11. Preserve approximate amounts as approximate. Do not present them "
    f"as confirmed fixed amounts.\n"
    f"12. If a fee head or amount is absent from the retrieved context, "
    f"write 'Not specified in the retrieved document context'.\n"
    f"13. Do not calculate a grand total unless all required components "
    f"are available and their payment categories are clear.\n\n"

    f"OUTPUT FORMAT FOR FEE QUESTIONS:\n"
    f"Start with the programme name and academic session.\n"
    f"Present fee details in a table with these columns:\n"
    f"Fee Head | Amount | Payment Type / Remarks\n\n"
    f"Include every applicable fee head found in the retrieved context.\n"
    f"After the table, add important notes, additional charges, refund "
    f"conditions, and fee-change notices.\n\n"

    f"OUTPUT FORMAT FOR OTHER QUESTIONS:\n"
    f"Provide a clear, accurate, and well-structured answer using "
    f"only the relevant retrieved document information.\n\n"

    f"Student Programme: {programme}\n\n"
    f"Official Document Context:\n{context}\n\n"
    f"Student Question: {query}\n\n"

    f"Give a complete and accurate answer. For fee-related questions, "
    f"completeness and accuracy are more important than brevity."
)


    response = llm.invoke(prompt)
    return {"messages": [("ai", response.content.strip())]}


#step 4 - router function 

def route_query(state:State):
    if state['query_type'] == 'academic':
        return "academic_rag"
    elif state['query_type'] == "fee":
        return "fee_rag"
    else:
        return "general"


#step 5 - Building the graph 

graph = StateGraph(State)

graph.add_node("classifier",classifier_node)
graph.add_node("academic_rag",academic_rag_node)
graph.add_node("fee_rag",fee_rag_node)
graph.add_node("general",general_node)
graph.add_node("response",response_node)

#edges 

graph.add_edge(START,"classifier")

graph.add_conditional_edges(
    "classifier",route_query
)

graph.add_edge("academic_rag","response")
graph.add_edge("fee_rag","response")
graph.add_edge("general","response")

graph.add_edge("response",END)

app = graph.compile()
#step 6 - Run the code 

print("\n========== WELCOME TO ABESIT COLLEGE ASSISTANT ==========\n")

print("Select your programme:\n")
print("1. B.Tech")
print("2. BBA")
print("3. BCA")
print("4. B.Tech Lateral Entry")
print("5. B.Pharm Lateral Entry")

programme_map = {
    "1": "B.Tech",
    "2": "BBA",
    "3": "BCA",
    "4": "B.Tech Lateral Entry",
    "5": "B.Pharm Lateral Entry"
}

while True:
    choice = input("\nEnter your choice (1-5): ").strip()

    if choice in programme_map:
        student_programme = programme_map[choice]
        break

    print("Invalid choice! Please select between 1 and 5.")

print(f"\nGreat! You're enrolled in {student_programme}.\n")
print("You can now ask questions about academics, fees and eligibility.")
print("Type 'exit' or 'quit' to end the conversation.\n")

while True:
    user_query = input("You: ").strip()

    if user_query.lower() in ["exit", "quit"]:
        print("\nThank you for using ABESIT College Assistant!")
        break

    if not user_query:
        continue

    result = app.invoke({
        "programme": student_programme,
        "messages": [("human", user_query)]
    })

    print(f"\nAssistant: {result['messages'][-1].content}\n")