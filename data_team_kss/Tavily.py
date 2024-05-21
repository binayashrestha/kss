from tavily import TavilyClient
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


def predict(message, history):
    """
    Generate a report based on the given message using Tavily search and OpenAI chat completion.
    """
    # Initialize OpenAI and Tavily clients
    client = OpenAI(api_key=OPENAI_API_KEY)
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

    # Perform Tavily search with the user's message
    search_results = tavily_client.search(message, search_depth="advanced", include_domains=['https://www.lftechnology.com/'])
    content = search_results["results"]

    # Define system and user messages for the OpenAI chat completion
    system_message = {
        "role": "system",
        "content": "You are AskRibby, an AI chatbot created by `Leapfrog Technology` also know as `LF Technology` which provide complete and relevant answer to a my QUESTION."
    }
    user_message = {
        "role": "user",
        "content": f'Information: """{content}"""\n\n'
        f"Using the above information, answer the following "
        f'query: "{message}" in detail.'
        f"Provide answers in markdown syntax and list the citations at last.",
    }

    prompt = [system_message, user_message]

    # Generate the response using OpenAI chat completion with streaming
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=prompt, temperature=0, stream=True
    )

    # Stream the response partial messages
    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            partial_message = partial_message + chunk.choices[0].delta.content
            yield partial_message


# Launch the Gradio chat interface with the predict function
gr.ChatInterface(predict, css="footer{display:none !important}").launch()
