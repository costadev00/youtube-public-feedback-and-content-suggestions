# YouTube Comments Sentiment Analysis

This project is a web application developed in Python using Flask that extracts comments from YouTube videos and performs sentiment analysis on them. The application processes the comments, assigns labels such as **EXCELLENT**, **AVERAGE**, **BAD**, and **TERRIBLE** to each comment, and presents an overall conclusion of the audience’s reaction. Additionally, based on the aggregated analysis, the application generates creative content suggestions specifically for TikTok using the OpenAI ChatCompletion API (leveraging a specialized marketing RAG on top of OpenAI’s “4th model”).

## Features

- **Comments Extraction:**  
  Utilizes the YouTube API to collect up to 2000 comments from a video.

- **Sentiment Analysis:**  
  Uses a pre-trained sentiment analysis model ([nlptown/bert-base-multilingual-uncased-sentiment](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)) from Hugging Face to classify each comment from a range of 1 (very negative) to 5 (highly positive).  
  - **Custom Emoji Analysis:** Custom routine to better interpret comments mostly consisting of emojis.  
  - **Aggregation:** Computes an overall sentiment average from individual comment scores and generates a detailed analysis summary.

- **Content Suggestions (RAG + OpenAI):**  
  Harnesses a Retrieval-Augmented Generation workflow based on OpenAI’s “4th model,” which references a broad range of vectorized documents focusing on marketing sources. Generates at least three novel and creative content ideas tailored for social platforms such as TikTok, each with a brief explanation of how it addresses viewer sentiment.

- **Modern UI/UX:**  
  - A responsive web interface built with Flask and Bootstrap.  
  - A custom light-green toggle button to display or hide the analyzed comments.  
  - A results page showing an overall sentiment conclusion, creative content suggestions, and a collapsible section for individual comment analyses.  
  - Clear formatting of suggestions – newlines in the suggestions are converted to HTML `<br>` tags for improved readability.

## Requirements

- Python 3.7 or higher  
- Flask  
- google-api-python-client  
- transformers  
- torch  
- openai  

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/youtube-comments-analysis.git
   cd youtube-comments-analysis
   ```
2. Create and activate a virtual environment (optional, but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure your YouTube API Key:
   Create a file named `api_key.py` in the root of the project and insert:
   ```python
   API_KEY = "YOUR_API_KEY_HERE"
   OPENAI_API_KEY = "YOU_API_KEY_HERE"
   ```

## Usage

1. Run the application:
   ```bash
   python main.py
   ```
2. Access the application in your browser at http://127.0.0.1:5000/.
3. Enter the link of a YouTube video and submit the form to see the comment analysis and personalized conclusion.

## Architecture

The architecture of the YouTube Comments Sentiment Analysis application is illustrated in the following diagram:

```mermaid
flowchart TD
    A[User] -->|Submits YouTube Link| B1(Extract Video ID)
    A[User] -->|Inserts Content Theme| B2(YouTube Search)
    B1 --> C1[YouTube API - Retrieve Comments]
    B2 --> C2[Extract Comments from Top 5 Videos]
    
    C1 & C2 -->|All Comments| D[Sentiment Analysis Module]
    D --> D1[Pre-trained BERT Model - nlptown/bert-base-multilingual-uncased-sentiment]
    D --> D2[Custom Emoji Detection]
    
    D1 & D2 --> E[Rank Comments - 1-5 Scale]
    E --> F[Aggregate Sentiment Scores]
    F --> G[Generate Average Classification]
    
    G --> H[Build Context for RAG]
    H --> I[RAG System - Built on GPT-4]
    I --> I1[Context: Marketing Books & Content Creation Documents]
    
    I & I1 --> J[Generate Marketing Campaign Strategy]
    J --> K[Store Results - JSON Files]
    K --> L[Display Results in UI]
```

The main components of the architecture are:

1. **User Input Processing:**  
   Accepts either a direct YouTube link or a content theme, which initiates different processing paths.

2. **YouTube Data Collection:**  
   - For direct links: Extracts video ID and retrieves comments
   - For themes: Searches YouTube for relevant videos and extracts comments from multiple sources

3. **Sentiment Analysis Module:**  
   Processes the comments using a pre-trained BERT sentiment analysis model and custom emoji detection.

4. **Aggregation and Classification:**  
   Ranks comments on a 1-5 scale, aggregates scores, and generates an overall sentiment classification.

5. **Retrieval-Augmented Generation (RAG):**  
   Takes the sentiment analysis results and builds context for the RAG system built on GPT-4, which is enhanced with specialized marketing knowledge.

6. **Content Strategy Generation:**  
   Produces tailored marketing campaign strategies based on the sentiment analysis and specialized marketing context.

7. **Storage and Presentation:**  
   Stores analysis results in JSON files and presents them through a responsive user interface.
