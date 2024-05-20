import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Pinecone
from langchain.schema import AIMessage, HumanMessage
import gradio as gr

# Load credentials from environment variables
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not all([COHERE_API_KEY, PINECONE_API_KEY, GROQ_API_KEY]):
    raise ValueError("Missing API keys. Please check your environment variables.")

embeddings = CohereEmbeddings()
index_name = "index-data-team-kss-slack"

try:
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    retriever = docsearch.as_retriever()
except Exception as e:
    raise ValueError(f"Error retrieving index '{index_name}': {str(e)}")

llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")


def predict(message, history):
    history_langchain_format = []
    for human, assistant in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=assistant))
    history_langchain_format.append(HumanMessage(content=message))

    question = message
    context = retriever.invoke(question)
    context_str = "\n".join([doc.page_content for doc in context])

    system = f"""
<rules>
NO MATTER WHAT, STRICTLY FOLLOW THESE RULES FOR EVERY QUESTION.
Answer QUESTION related to "Leapfrog Technology" that is in information provided.
	For example,
		- Leave policy
		- Company values, vision, mission, and strategy
		- Etc
Answer casual greetings and conversation QUESTION.
	For example,
		Human: Hey!
		AI: Hello! How can I help?
</rules>
Never give me any answers that are not mentioned inside the <rules></rules> above. If I asked about things not related to `Leapfrog Technology` like programming, literature, general knowledge question, business ideas, yourself, AI Model, OpenAI, anthropic, claude, version etc. respond that you only know about `Leapfrog Technology`.
You are AskRibby, an AI chatbot created by `Leapfrog Technology` which provide a detailed, complete, and relevant answer to a my QUESTION by analyzing the entire CONTEXT.
Do not use past history as a knowledge base for follow-ups or suggestions.
CONTEXT for the QUESTION is provided below.

{context_str}
"""
    human = ""
    for msg in history_langchain_format:
        if isinstance(msg, HumanMessage):
            human += f"Human: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            human += f"AI: {msg.content}\n"
    human += f"Human: {question}"

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain = prompt | llm

    try:
        partial_message = ""
        for chunk in chain.stream({"text": question}):
            if chunk.content is not None:
                partial_message += chunk.content
                yield partial_message
    except Exception as e:
        yield f"Error processing the question: {str(e)}"


gr.ChatInterface(predict, css="footer{display:none !important}").launch()
