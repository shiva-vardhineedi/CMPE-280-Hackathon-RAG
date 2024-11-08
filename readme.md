# Flask Chatbot with FAISS and ChatGroq Integration

## Overview
This project is a Flask-based chatbot API designed to answer user queries using an integrated retrieval-augmented generation (RAG) approach. The chatbot utilizes the following components:

- **FAISS (Facebook AI Similarity Search)**: Used for fast similarity searches on preprocessed text data.
- **ChatGroq**: A language model integrated using Langchain's ChatGroq class to provide concise, structured responses.

The result is a chatbot that can answer questions using internal indexed data, providing flexible and detailed responses.

## Features
- **Retrieval-Augmented Generation (RAG)** using FAISS for efficient vector similarity search.
- **ChatGroq Language Model** for generating responses based on the context.
- **CORS Support** for allowing cross-origin requests to the Flask application.

## Setup Instructions

### Prerequisites
Ensure you have the following installed:
- Python 3.8 or higher
- `pip` for Python package management

Clone the repository to your local environment:
```sh
$ git clone <repository_url>
$ cd <repository_folder>
```

### Install Dependencies
Install the required packages using `pip`:

```sh
pip install -r requirements.txt
```

The `requirements.txt` file should include:

```
Flask
flasgger
langchain-community
langchain-groq
sentence-transformers
faiss-cpu
dotenv
flask-cors
```

### Data Preparation
There is a zip file (`Data.zip`) that contains the data required for this project. Unzip the file to get a folder named `CMPE280_DataSources_20241107_Hackathon` that contains several CSVs and PDFs. These files are used to create the knowledge store in FAISS.

To unzip the file, run the following command:

```sh
unzip Data.zip
```

Ensure the `CMPE280_DataSources_20241107_Hackathon` folder is in the root of your project directory. The content of these CSVs and PDFs will be processed to create the FAISS index.

### Environment Variables
Create a `.env` file in the project root to store API keys and configuration variables. Ensure that your `.env` file includes the following:

```
GROQ_API_KEY=your_groq_api_key_here
```

Replace `your_groq_api_key_here` with the actual API key from Groq.

### FAISS Index Creation
Before running the server, you need to create a FAISS index using the code in `rag.ipynb`:


### Running the Application
After the setup is complete, run the Flask application with the following command:

```sh
python3 app.py
```
The application will start and be available at `http://127.0.0.1:5000`.

## API Endpoints
### `/chat` (POST)
This endpoint takes a user query and provides a response using FAISS and ChatGroq.

- **URL**: `/chat`
- **Method**: POST
- **Request Body**:
  ```json
  {
    "query": "What is the total rice production in Egypt and Saudi Arabia?"
  }
  ```
- **Response**:
  - **200 OK**: Returns the response to the query in JSON format.
  - **400 Bad Request**: If no query is provided.
  - **500 Internal Server Error**: If any error occurs while processing.

### Example cURL Request
```sh
curl -X POST "http://127.0.0.1:5000/chat" -H "Content-Type: application/json" -d '{"query": "What is the total rice production in Egypt and Saudi Arabia?"}'
```

## Code Overview
- **`app.py`**: Main application file that sets up the Flask server, integrates the ChatGroq model, uses FAISS for vector similarity search.
- **FAISS Index**: Preprocessed text data is embedded and indexed for quick retrieval using `sentence-transformers/all-MiniLM-L6-v2` from HuggingFace.
- **ChatGroq Integration**: The language model is used to provide styled, concise responses based on the retrieved context.

## Known Issues & Troubleshooting
- **HTTP Errors**: If you get a `Payload Too Large` error, consider truncating or summarizing the context further to meet the model's input size requirements.

## Future Improvements
- **Production Server Deployment**: The current setup is for development purposes. For production, use a WSGI server like Gunicorn.
- **Batching Queries**: Implement batch processing for larger queries to handle more extensive context requirements without exceeding model input limits.
- **More Robust Error Handling**: Improve error handling in the case of API request failures or missing data.

## Contributing
Feel free to fork this project and make a pull request if you have any improvements in mind. Contributions are always welcome!

## License
This project is licensed under the MIT License. See the `LICENSE` file for more information.

## Contact
If you have questions or need further help, feel free to reach out:
- **Email**: your_email@example.com
- **GitHub**: [YourGitHubUsername](https://github.com/YourGitHubUsername)

---
Happy coding and have fun building intelligent chatbots!

