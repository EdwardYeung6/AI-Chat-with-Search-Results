# Chat with Search Results 🔍

A Streamlit application that allows users to search for topics and have interactive conversations about the search results using AI.

## Features

- Google Search integration to find relevant content
- AI-powered chat interface to discuss search results
- Clean and intuitive user interface
- Search result summaries displayed in sidebar
- Error handling and logging

## Installation

1. Clone this repository:

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a .env file in the root directory with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   GOOGLE_API_KEY=your_google_api_key
   GOOGLE_CSE_ID=your_google_cse_id
   BASE_API_PATH=your_base_api_path
   ```

## Run the Project
----------------------------

1. Navigate to the project directory:
   ```
   cd your-project-directory
   ```

2. Start the Streamlit application:
   ```
   streamlit run app.py
   ```

3. Open the displayed local URL in your browser (usually http://localhost:8501)

4. Search the information and start asking questions
