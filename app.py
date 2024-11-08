from flask import Flask, request, jsonify
from flasgger import Swagger, swag_from
from langchain_community.vectorstores.faiss import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import logging
from duckduckgo_search import DDGS
from pprint import pprint
from langchain_huggingface import HuggingFaceEmbeddings
from flask_cors import CORS

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app and set static folder to serve Swagger UI assets
app = Flask(__name__)
CORS(app)  # Add CORS support to allow cross-origin requests

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
pp = pprint

# Initialize Swagger for API documentation
swagger = Swagger(app)

# Initialize the ChatGroq model using API key from environment variable
llm = ChatGroq(api_key=os.getenv('GROQ_API_KEY'), max_tokens=200)

# Load the FAISS vector store and embedding model using HuggingFaceEmbeddings
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    logger.debug("Attempting to load FAISS vector store with allow_dangerous_deserialization=True")
    faiss_index = FAISS.load_local("faiss_store", embeddings, allow_dangerous_deserialization=True)
    logger.info("FAISS vector store successfully loaded.")
except Exception as e:
    logger.error(f"Failed to load FAISS vector store: {e}")
    exit()

# Create a retriever from the FAISS index
retriever = faiss_index.as_retriever()

# DuckDuckGo search function using DDGS class
def duckduckgo_search(query):
    logger.info("Performing DuckDuckGo search...")
    ddgs = DDGS()  # Initialize the DuckDuckGo search object
    search_results = list(ddgs.text(query, max_results=3))  # Retrieve the top 3 results

    if search_results:
        logger.info("DuckDuckGo search successful.")
        pp(search_results)  # Pretty print the search results for debugging
        # Formatting the results to provide them as context
        return "\n".join([f"{result['title']}: {result['body']} (URL: {result['href']})" for result in search_results])
    else:
        logger.warning("No relevant results found from DuckDuckGo.")
        return "No relevant results found from DuckDuckGo."

# Function to determine whether the query is relevant to the knowledge base
def is_relevant_query(query):
    # List of keywords related to the data we have in FAISS (e.g., agriculture, food security, nutrition, FDI, GDP, etc.)
    relevant_keywords = [
        "agriculture", "food security", "nutrition", "hunger", "malnutrition", 
        "rice production", "fertilizer", "fishing", "food consumption", 
        "food affordability", "farming", "urban agriculture", "food systems", "FAO",
        "foreign direct investment", "FDI", "current account balance", "GDP", "economic growth", "debt"
    ]
    
    # Check if any relevant keyword appears in the query (case-insensitive match)
    for keyword in relevant_keywords:
        if keyword.lower() in query.lower():
            return True
    return False

# Function to query the system with simplified context management
# Function to query the system with enhanced context-handling strategies
def query_system(query):
    logger.info(f"Received query: {query}")

    # Determine if the query is relevant to the available context
    if is_relevant_query(query):
        # Retrieve relevant context using the FAISS retriever
        logger.info("Query is relevant to knowledge base. Retrieving context from FAISS...")
        relevant_texts = retriever.get_relevant_documents(query)
        
        # Limit the number of relevant chunks to reduce context size
        MAX_CHUNKS = 5
        if len(relevant_texts) > MAX_CHUNKS:
            logger.info(f"Retrieved {len(relevant_texts)} chunks. Limiting to {MAX_CHUNKS} most relevant chunks.")
            relevant_texts = relevant_texts[:MAX_CHUNKS]  # Keep only the top 5 relevant chunks

        # Combine the selected chunks into a single context
        context = "\n".join([text.page_content for text in relevant_texts])

        # If the combined context is still too long, truncate or summarize
        MAX_CONTEXT_LENGTH = 2000
        if len(context) > MAX_CONTEXT_LENGTH:
            logger.info(f"Context size ({len(context)}) exceeds {MAX_CONTEXT_LENGTH} characters. Truncating context.")
            context = context[:MAX_CONTEXT_LENGTH] + "... [truncated]"

        # Prepare the source information to guide the response
        source_info = "Note: The following information was retrieved internally from the FAISS index."
    
    else:
        # If the query is irrelevant to the internal knowledge base, no context is added
        logger.info("Query is not related to the knowledge base. Proceeding without FAISS context.")
        context = ""
        source_info = "Note: The following information is generated based on general knowledge."

    # Prepare the messages for ChatGroq invocation with updated prompt for styled and concise responses
    messages = [
    (
        "system",
        "You are a helpful assistant that provides answers in a well-structured, styled format. "
        "Use bullet points, lists, or sections to make the response easy to read. Keep your response concise, "
        "use context only if it's directly relevant, and only provide the most critical information needed to answer the user's question. "
        "While responding, don't explain too much unless the user specifically asks for more details. Keep your answers to the point."
        "\n\nContext (if applicable):\n{context}\n\n"
        "Note: The following information was retrieved from your data. Please clearly indicate if any part of the answer is based on external data."
        ),
        ("user", query)
    ]


    # Invoke the model with the provided messages
    logger.info("Invoking the ChatGroq model with the provided context and query...")
    response = llm.invoke(messages)

    # Print debug information about the type of data source used
    if "FAISS index" in source_info:
        logger.info("The answer is based on information retrieved from the FAISS index.")
    else:
        logger.info("The answer is based on general knowledge.")
    
    pp(response.content)  # Pretty print the final response for better debugging

    return response.content


# Flask endpoint for handling chat queries
@app.route('/chat', methods=['POST'])
@swag_from({
    'tags': ['Chat'],
    'summary': 'Chat Endpoint for Queries',
    'description': 'This endpoint takes a user query and provides a response using FAISS and ChatGroq. '
                   'If no relevant information is found in the FAISS index, it performs a DuckDuckGo search.',
    'parameters': [
        {
            'name': 'query',
            'in': 'body',
            'type': 'string',
            'required': True,
            'description': 'User query for the chatbot',
            'schema': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'example': 'What is the total rice production in Egypt and Saudi Arabia?'
                    }
                }
            }
        }
    ],
    'responses': {
        '200': {
            'description': 'Successful response',
            'schema': {
                'type': 'object',
                'properties': {
                    'response': {
                        'type': 'string',
                        'example': 'The total rice production in Egypt is X and in Saudi Arabia is Y.'
                    }
                }
            }
        },
        '400': {
            'description': 'Invalid input, missing query'
        },
        '500': {
            'description': 'Internal server error'
        }
    }
})
def handle_query():
    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    response = query_system(user_query)
    return jsonify({"response": response})

# Run the Flask app
if __name__ == "__main__":
    app.run(port=5000, debug=True)
