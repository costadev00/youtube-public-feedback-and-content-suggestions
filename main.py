import re
import time
from urllib.parse import urlencode
from flask import Flask, render_template, request, jsonify, redirect
from googleapiclient.discovery import build
from transformers import pipeline
from api_key import API_KEY  # YouTube API key
from api_key import OPENAI_API_KEY  # OpenAI API key
import openai
import os
import json
from datetime import datetime
from flask_cors import CORS
from openai import OpenAI

# Configure OpenAI with your API key
openai.api_key = OPENAI_API_KEY

# Set your custom assistant model name
ASSISTANT_MODEL = (
    "asst_W1LW7q9EXFRVWaUn7xiSp0yR"  # Replace with your custom assistant model
)

client = OpenAI(api_key=OPENAI_API_KEY)


app = Flask(__name__, static_folder="static")
CORS(app, resources={r"/*": {"origins": "*"}})


def get_content_suggestions(analysis_summary, average, context, comments):
    """
    Receives a detailed analysis summary, the average sentiment,
    a context string, and the comments list, then returns tailored
    TikTok content marketing ideas using your custom assistant.
    """

    def strip_markdown(text):
        import re
        # Remove triple backticks and code blocks
        text = re.sub(r'```+[^`]+```+', '', text)
        # Remove inline backticks
        text = re.sub(r'`([^`]+)`', r'\1', text)
        # Remove bold/italic markers (** or __ or *)
        text = re.sub(r'(\*+)([^*]+)(\*+)', r'\2', text)
        text = re.sub(r'(_+)([^_]+)(_+)', r'\2', text)
        # Remove headings (#, `###### Heading`)
        text = re.sub(r'(^|\n)#{1,6}\s*(.*)', r'\2', text)
        return text

    thread = client.beta.threads.create()

    prompt = (
        f"Based on the analysis of the comments from these {context} video, the following observations were made:\n"
        f"- The audience's overall sentiment is {average:.2f} stars.\n"
        f"- Analysis summary: {analysis_summary}\n\n"
        "Given these insights and your expertise as a marketing expert, "
        f"please suggest in Portuguese from Brazil, some innovative and engaging social media content marketing ideas tailored for {context} content. "
        "For each suggestion, include a brief explanation of how the idea leverages the audience's feedback "
        "to drive engagement and boost brand growth."
        f"Consider these comments to get a plus reaction of the audience: {comments[:20] if isinstance(comments, list) else 'No comments available'}"
    )

    client.beta.threads.messages.create(
        thread_id=thread.id, 
        role="user", 
        content=prompt
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=ASSISTANT_MODEL,
        instructions="Please address the user as Jane Doe. The user has a premium account.",
    )

    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run.status == "completed":
            break
        elif run.status in ["failed", "cancelled", "expired"]:
            raise RuntimeError(f"Run failed with status: {run.status}")
        time.sleep(1)

    messages = client.beta.threads.messages.list(thread_id=thread.id)

    assistant_responses = []
    for msg in messages.data:
        if msg.role.lower() == "assistant":
            for content_block in msg.content:
                if hasattr(content_block, "text") and hasattr(content_block.text, "value"):
                    # Apply Markdown filtering here
                    filtered_text = strip_markdown(content_block.text.value)
                    assistant_responses.append(filtered_text)

    return "\n".join(assistant_responses)


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
    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    request = youtube.commentThreads().list(
        part="snippet", videoId=video_id, textFormat="plainText", maxResults=100
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
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictograms
        "\U0001f680-\U0001f6ff"  # transports & maps
        "\U0001f1e0-\U0001f1ff"  # flags
        "\U00002700-\U000027bf"  # miscellaneous symbols
        "\U0001f900-\U0001f9ff"  # additional symbols
        "]+",
        flags=re.UNICODE,
    )
    result = emoji_pattern.sub(r"", comment)
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
        task="text-classification",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        device=-1,  # Force CPU
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
                res["label"] = map_star_label(res["label"])
                results.append(res)
        else:
            res = sentiment_pipeline(comment_clean, truncation=True)[0]
            res["label"] = map_star_label(res["label"])
            results.append(res)
    return results


def summarize_sentiments(sentiments):
    total = 0
    count = 0
    for result in sentiments:
        label = result.get("label", "")
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
        return "Infelizmente, não conseguimos coletar dados suficientes para chegar a uma conclusão. Por favor, tente novamente mais tarde."

    if average >= 4.5:
        rating = "EXCELENTE"
        extra_message = "O feedback foi extremamente positivo, indicando que o conteúdo repercutiu fortemente com a audiência."
        english_rating = "AMAZING"
        english_extra_message = "The feedback was overwhelmingly positive, indicating that the content resonated strongly with the audience."
    elif average >= 4.0:
        rating = "BOM"
        extra_message = "O feedback foi geralmente positivo, mostrando que a audiência teve uma resposta favorável ao conteúdo."
        english_rating = "GOOD"
        english_extra_message = "The feedback was generally positive, showing that the audience had a favorable response to the content."
    elif average >= 3.5:
        rating = "MÉDIO"
        extra_message = "O feedback foi moderadamente positivo, sugerindo espaço para melhorias. Considere refinar o conteúdo ou a apresentação."
        english_rating = "AVERAGE"
        english_extra_message = "The feedback was moderately positive, suggesting room for improvement. Consider refining the content or presentation."
    elif average >= 3.0:
        rating = "RUIM"
        extra_message = "O feedback ficou abaixo da média, indicando que o conteúdo não atendeu completamente às expectativas da audiência."
        english_rating = "BAD"
        english_extra_message = "The feedback was below average, indicating that the content did not fully meet audience expectations."
    else:
        rating = "PÉSSIMO"
        extra_message = "O feedback foi extremamente negativo, indicando que o conteúdo não atendeu às expectativas. Reavalie o material e considere mudanças drásticas."
        english_rating = "TERRIBLE"
        english_extra_message = "The feedback was extremely negative, indicating that the content did not meet expectations. Reevaluate the material and consider drastic changes."

    conclusion = (
        f"Conclusão: A reação geral da audiência foi <strong>{rating}</strong> com uma média de {average:.2f} estrelas. "
        f"{extra_message}"
    )
    
    # Keep the English version for reference or other uses if needed
    english_conclusion = (
        f"Conclusion: The overall reaction from the audience was <strong>{english_rating}</strong> with an average of {average:.2f} stars. "
        f"{english_extra_message}"
    )
    
    return conclusion


@app.context_processor
def utility_processor():
    return dict(zip=zip)


def save_comments_to_file(video_id, comments, sentiments, average, conclusion):
    """
    Saves the comments and their analysis to a JSON file for later use.
    """
    if not os.path.exists("analysis_results"):
        os.makedirs("analysis_results")

    data = {
        "video_id": video_id,
        "timestamp": datetime.now().isoformat(),
        "average_sentiment": average,
        "conclusion": conclusion,
        "comments_analysis": [
            {"comment": comment, "sentiment": sentiment.get("label", "N/A")}
            for comment, sentiment in zip(comments, sentiments)
        ],
    }

    filename = (
        f"analysis_results/{video_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return filename


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        data = request.get_json()
        video_id = [extract_video_id(url) for url in data["videos"]]
        if not video_id:
            return render_template(
                "index.html", error="Invalid link. Please enter a valid YouTube link."
            )

        # Retrieve comments
        comments = get_comments(video_id[0], API_KEY, max_results=2000)
        sentiments = analyze_comments(comments)
        average = summarize_sentiments(sentiments)
        conclusion = generate_conclusion(average)

        # Save comments and analysis to file
        saved_file = save_comments_to_file(video_id, comments, sentiments, average, conclusion)

        # Now pass comments as an argument
        suggestions = get_content_suggestions(conclusion, average, context="video", comments=comments)

        return render_template(
            "result.html",
            conclusion=conclusion,
            comments=comments,
            sentiments=sentiments,
            suggestions=suggestions,
            zip=zip,
        )
    return render_template("index.html")


@app.route("/batch-analysis-ctx", methods=["POST"])
def batch_analysis():
    """
    Rota para análise em lote de múltiplos vídeos do YouTube.
    Recebe uma lista de vídeos com links e tags, analisa os comentários de todos eles
    e retorna uma análise agregada, salvando também os resultados em arquivos.
    Também faz a contagem de tokens para estimar custos.

    Exemplo de JSON esperado no request:
    {
        "videoCtx": [
            {
                "link": "https://www.youtube.com/watch?v=videoId",
                "tags": ["tag1", "tag2", "..."]
            },
            ...
        ],
        "max_comments_per_video": 50  # opcional, default 50
    }
    """
    try:
        # Importar o analisador de tokens

        data = request.get_json()

        if not data or "videoCtx" not in data:
            return jsonify(
                {
                    "error": "Invalid request. Please provide a list of videos with links and tags."
                }
            ), 400

        video_contexts = data.get("videoCtx", [])
        context_input = data.get("ctx", "")
        max_comments = data.get("max_comments_per_video", 20)

        # Verificar se há vídeos para analisar
        if not video_contexts or not isinstance(video_contexts, list):
            return jsonify({"error": "Please provide at least one video."}), 400

        batch_folder = f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        batch_path = os.path.join("analysis_results", batch_folder)

        if not os.path.exists("analysis_results"):
            os.makedirs("analysis_results")

        if not os.path.exists(batch_path):
            os.makedirs(batch_path)

        all_comments = []
        all_sentiments = []
        processed_videos = []
        failed_videos = []

        # Criar pasta para análise de tokens
        tokens_folder = os.path.join(batch_path, "token_analysis")
        if not os.path.exists(tokens_folder):
            os.makedirs(tokens_folder)

        # Processar cada vídeo
        for video_ctx in video_contexts:
            link = video_ctx.get("link", "")
            tags = video_ctx.get("tags", [])
            

            # Validar se o link existe
            if not link:
                failed_videos.append(
                    {"video_ctx": video_ctx, "reason": "Missing YouTube link"}
                )
                continue

            video_id = extract_video_id(link)
            if not video_id:
                failed_videos.append(
                    {"video_ctx": video_ctx, "reason": "Invalid YouTube link format"}
                )
                continue

            try:
                # Obter e analisar comentários
                comments = get_comments(video_id, API_KEY, max_results=max_comments)
                sentiments = analyze_comments(comments)

                if not comments or len(comments) == 0:
                    failed_videos.append(
                        {
                            "video_ctx": video_ctx,
                            "reason": "No comments found or unable to retrieve comments",
                        }
                    )
                    continue

                # Calcular média para este vídeo
                video_average = summarize_sentiments(sentiments)
                video_conclusion = generate_conclusion(video_average)

                # Preparar dados para salvar
                video_data = {
                    "video_id": video_id,
                    "link": link,
                    "tags": tags,
                    "timestamp": datetime.now().isoformat(),
                    "average_sentiment": video_average,
                    "conclusion": video_conclusion,
                    "comments_analysis": [
                        {"comment": comment, "sentiment": sentiment.get("label", "N/A")}
                        for comment, sentiment in zip(comments, sentiments)
                    ],
                }

                # Salvar em um arquivo com nome baseado no ID do vídeo e timestamp
                video_filename = (
                    f"{video_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                video_filepath = os.path.join(batch_path, video_filename)
                with open(video_filepath, "w", encoding="utf-8") as f:
                    json.dump(video_data, f, ensure_ascii=False, indent=4)

                all_comments.extend(comments)
                all_sentiments.extend(sentiments)

                processed_videos.append(
                    {
                        "video_id": video_id,
                        "link": link,
                        "tags": tags,
                        "comments_count": len(comments),
                        "average": video_average,
                        "conclusion": video_conclusion,
                    }
                )

            except Exception as e:
                failed_videos.append({"video_ctx": video_ctx, "reason": str(e)})
                print(f"Error processing video {link}: {str(e)}")

        if all_sentiments:
            aggregate_average = summarize_sentiments(all_sentiments)
            aggregate_conclusion = generate_conclusion(aggregate_average)
        else:
            return jsonify(
                {
                    "error": "Could not analyze any videos. Please check the logs for details.",
                    "failed_videos": failed_videos,
                }
            ), 400

        # Extrair todas as tags únicas de todos os vídeos processados
        all_tags = []
        for video in processed_videos:
            all_tags.extend(video.get("tags", []))
        unique_tags = list(set(all_tags))

        # Gerar sugestões baseadas na análise agregada
        suggestions = get_content_suggestions(aggregate_conclusion, aggregate_average, context=context_input, comments=all_comments)

        # Salvar análise agregada em um arquivo separado
        aggregate_data = {
            "timestamp": datetime.now().isoformat(),
            "videos_analyzed": len(processed_videos),
            "total_comments": len(all_comments),
            "aggregate_average": aggregate_average,
            "aggregate_conclusion": aggregate_conclusion,
            "content_suggestions": suggestions,
            "all_tags": unique_tags,
            "processed_videos": processed_videos,
            "failed_videos": failed_videos,
        }

        aggregate_filename = os.path.join(batch_path, "aggregate_analysis.json")
        with open(aggregate_filename, "w", encoding="utf-8") as f:
            json.dump(aggregate_data, f, ensure_ascii=False, indent=4)

        # return jsonify(
        #     {
        #         "status": "success",
        #         "videos_analyzed": len(processed_videos),
        #         "total_comments": len(all_comments),
        #         "aggregate_average": aggregate_average,
        #         "aggregate_conclusion": aggregate_conclusion,
        #         "content_suggestions": suggestions,
        #         "all_tags": unique_tags,
        #         "batch_folder": batch_folder,
        #         "processed_videos": processed_videos,
        #         "failed_videos": failed_videos,
        #     }
        # )

        return render_template(
            'result.html',
            conclusion=aggregate_conclusion,
            comments=all_comments,
            sentiments=all_sentiments,
            suggestions=suggestions,
            zip=zip
        )


    except Exception as e:
        print(f"Error in batch analysis: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)