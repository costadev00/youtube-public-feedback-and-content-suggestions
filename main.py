import re
import asyncio
from flask import Flask, render_template, request
from googleapiclient.discovery import build
from transformers import pipeline
from api_key import API_KEY          # YouTube API key
from api_key import OPENAI_API_KEY   # OpenAI API key
import openai

# Configure OpenAI with your API key
openai.api_key = OPENAI_API_KEY

app = Flask(__name__)

def get_content_suggestions(analysis_summary, average):
    """
    Receives a detailed analysis summary and the average sentiment,
    and returns tailored TikTok content ideas to engage sports video audiences,
    based on their feedback.
    """
    prompt = (
        f"Based on the analysis of the comments from this sports video, the following observations were made:\n"
        f"- The audience's overall sentiment is {average:.2f} stars.\n"
        f"- Analysis summary: {analysis_summary}\n\n"
        "Considering that this feedback comes from a sports-related content video, "
        "please suggest at least 3 innovative and engaging TikTok content ideas that build on this audience reaction. "
        "Each suggestion should include a brief explanation of how the idea taps into the current viewer sentiment and how it can boost audience engagement."
    )
    
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a creative assistant specializing in social media content production for sports."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=300
    )
    return completion.choices[0].message["content"]

def extract_video_id(url):
    """
    Extracts the video_id from the YouTube link.
    Supports standard URLs (with "v=") and shortened URLs (youtu.be).
    """
    video_id = None
    match = re.search(r"v=([A-Za-z0-9_-]{11})", url)
    if match:
        video_id = match.group(1)
    else:
        match = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", url)
        if match:
            video_id = match.group(1)
    return video_id

def get_comments(video_id, api_key, max_results=50):
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText",
        maxResults=100
    )
    while request and len(comments) < max_results:
        response = request.execute()
        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
            if len(comments) >= max_results:
                break
        request = youtube.commentThreads().list_next(request, response)
    return comments

def is_emoji_comment(comment):
    comment = comment.strip()
    emoji_pattern = re.compile("[" 
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictograms
        u"\U0001F680-\U0001F6FF"  # transports & maps
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002700-\U000027BF"  # miscellaneous symbols
        u"\U0001F900-\U0001F9FF"  # additional symbols
        "]+", flags=re.UNICODE)
    result = emoji_pattern.sub(r'', comment)
    return len(result) == 0

def custom_emoji_analysis(comment):
    emoji_mapping = {
        "🤢": 1,
        "😠": 1,
        "🤬": 1,
        "😡": 1,
        "😢": 1,
        "😭": 1,
        "😞": 1,
        "🙁": 2,
        "😐": 3,
        "🤔": 3,
        "🙂": 4,
        "😊": 5,
        "😀": 5,
        "😁": 5,
        "❤️": 5,
        "👏": 5,
        "👍": 5,
    }
    total = 0
    count = 0
    for char in comment:
        if char in emoji_mapping:
            total += emoji_mapping[char]
            count += 1
    if count == 0:
        return None
    avg = total / count
    if avg < 1.5:
        label = "TERRIBLE"
    elif avg < 2.5:
        label = "BAD"
    elif avg < 3.5:
        label = "AVERAGE"
    else:
        label = "EXCELLENT"
    return {"label": label, "score": 1.0}

def clean_comment(comment):
    return " ".join(comment.split())

def get_sentiment_pipeline():
    sentiment_pipeline = pipeline(
        task='text-classification',
        model='nlptown/bert-base-multilingual-uncased-sentiment',
        device=-1  # Force CPU
    )
    return sentiment_pipeline

def map_star_label(label):
    try:
        value = float(label.split()[0])
    except Exception:
        return label
    if value >= 4.50:
        return "EXCELLENT"
    elif value >= 3.00:
        return "AVERAGE"
    elif value >= 2.50:
        return "BAD"
    elif value >= 1.50:
        return "TERRIBLE"
    else:
        return "TERRIBLE"

def analyze_comments(comments):
    sentiment_pipeline = get_sentiment_pipeline()
    results = []
    for comment in comments:
        comment_clean = clean_comment(comment)
        if is_emoji_comment(comment_clean):
            result = custom_emoji_analysis(comment_clean)
            if result:
                results.append(result)
            else:
                res = sentiment_pipeline(comment_clean, truncation=True)[0]
                res['label'] = map_star_label(res['label'])
                results.append(res)
        else:
            res = sentiment_pipeline(comment_clean, truncation=True)[0]
            res['label'] = map_star_label(res['label'])
            results.append(res)
    return results

def summarize_sentiments(sentiments):
    total = 0
    count = 0
    for result in sentiments:
        label = result.get('label', '')
        try:
            stars = float(label.split()[0])
        except (ValueError, IndexError):
            mapping = {"EXCELLENT": 5, "AVERAGE": 4, "BAD": 3, "TERRIBLE": 2}
            stars = mapping.get(label.upper(), 0)
        total += stars
        count += 1
    if count == 0:
        return None
    average = total / count
    return average

def generate_conclusion(average):
    if average is None:
        return "Unfortunately, we could not gather enough data to reach a conclusion. Please try again later."
    
    if average >= 4.5:
        rating = "AMAZING"
        extra_message = (
            "The feedback was overwhelmingly positive, indicating that the content resonated strongly with the audience."
        )
    elif average >= 4.0:
        rating = "GOOD"
        extra_message = (
            "The feedback was generally positive, showing that the audience had a favorable response to the content."
        )
    elif average >= 3.5:
        rating = "AVERAGE"
        extra_message = (
            "The feedback was moderately positive, suggesting room for improvement. Consider refining the content or presentation."
        )
    elif average >= 3.0:
        rating = "BAD"
        extra_message = (
            "The feedback was below average, indicating that the content did not fully meet audience expectations."
        )
    else:
        rating = "TERRIBLE"
        extra_message = (
            "The feedback was extremely negative, indicating that the content did not meet expectations. "
            "Reevaluate the material and consider drastic changes."
        )
    
    conclusion = (
        f"Conclusion: The overall reaction from the audience was <strong>{rating}</strong> with an average of {average:.2f} stars. "
        f"{extra_message}"
    )
    return conclusion

@app.context_processor
def utility_processor():
    return dict(zip=zip)

def save_comments_to_file(video_id, comments, sentiments, average, conclusion):
    """
    Salva os comentários e suas análises em um arquivo JSON para uso posterior.
    
    Args:
        video_id (str): ID do vídeo do YouTube
        comments (list): Lista de comentários
        sentiments (list): Lista de análises de sentimento
        average (float): Média de sentimento
        conclusion (str): Conclusão gerada pela análise
    """
    # Cria pasta para armazenar os resultados se não existir
    if not os.path.exists('analysis_results'):
        os.makedirs('analysis_results')
    
    # Prepara os dados para salvar
    data = {
        'video_id': video_id,
        'timestamp': datetime.now().isoformat(),
        'average_sentiment': average,
        'conclusion': conclusion,
        'comments_analysis': [
            {'comment': comment, 'sentiment': sentiment.get('label', 'N/A')}
            for comment, sentiment in zip(comments, sentiments)
        ]
    }
    
    # Salva em um arquivo com nome baseado no ID do vídeo e timestamp
    filename = f"analysis_results/{video_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    return filename


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        youtube_link = request.form.get('youtube_link')
        video_id = extract_video_id(youtube_link)
        if not video_id:
            return render_template('index.html', error="Invalid link. Please enter a valid YouTube link.")
        comments = get_comments(video_id, API_KEY, max_results=50)
        sentiments = analyze_comments(comments)
        average = summarize_sentiments(sentiments)
        conclusion = generate_conclusion(average)

        # Salva os comentários e análises em um arquivo
        saved_file = save_comments_to_file(video_id, comments, sentiments, average, conclusion)
        
        # Generate content suggestions based on the analysis using OpenAI
        suggestions = get_content_suggestions(conclusion, average)
        
        return render_template(
            'result.html',
            conclusion=conclusion,
            comments=comments,
            sentiments=sentiments,
            suggestions=suggestions,
            zip=zip
        )
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
