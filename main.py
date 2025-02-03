import re
from flask import Flask, render_template, request
from googleapiclient.discovery import build
from transformers import pipeline
from api_key import API_KEY

app = Flask(__name__)

def extract_video_id(url):
    """
    Extrai o video_id a partir do link do YouTube.
    Suporta URLs no formato padrão (com "v=") e encurtadas (youtu.be).
    """
    video_id = None
    # Tenta extrair do parâmetro "v="
    match = re.search(r"v=([A-Za-z0-9_-]{11})", url)
    if match:
        video_id = match.group(1)
    else:
        # Tenta extrair do formato youtu.be
        match = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", url)
        if match:
            video_id = match.group(1)
    return video_id

def get_comments(video_id, api_key, max_results=500):
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
        u"\U0001F300-\U0001F5FF"  # símbolos e pictogramas
        u"\U0001F680-\U0001F6FF"  # transportes e mapas
        u"\U0001F1E0-\U0001F1FF"  # bandeiras
        u"\U00002700-\U000027BF"  # diversos símbolos
        u"\U0001F900-\U0001F9FF"  # símbolos adicionais
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
        # Adicione outros emojis conforme necessário
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
        label = "PÉSSIMO"
    elif avg < 2.5:
        label = "RUIM"
    elif avg < 3.5:
        label = "MEDIANO"
    else:
        label = "EXCELENTE"
    return {"label": label, "score": 1.0}

def clean_comment(comment):
    """
    Realiza uma limpeza básica no comentário (remoção de espaços extras, etc.).
    """
    return " ".join(comment.split())

def get_sentiment_pipeline():
    """
    Retorna um pipeline de análise de sentimentos otimizado para o português.
    Tenta utilizar o modelo 'pysentimiento/roberta-base-portuguese-sentiment';
    em caso de falha, utiliza o modelo multilíngue.
    """
    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="pysentimiento/roberta-base-portuguese-sentiment",
            truncation=True
        )
    except Exception as e:
        print("Falha ao carregar o modelo específico para português. Usando o modelo multilíngue.")
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            truncation=True
        )
    return sentiment_pipeline

def map_star_label(label):
    """
    Mapeia um rótulo no formato "X stars" para um rótulo textual baseado nos thresholds:
      - ≥ 4.50: EXCELENTE
      - ≥ 3.50: MEDIANO
      - ≥ 2.50: RUIM
      - ≥ 1.50: PÉSSIMO
    """
    try:
        value = float(label.split()[0])
    except Exception as e:
        return label
    if value >= 4.50:
        return "EXCELENTE"
    elif value >= 3.50:
        return "MEDIANO"
    elif value >= 2.50:
        return "RUIM"
    elif value >= 1.50:
        return "PÉSSIMO"
    else:
        return "PÉSSIMO"

def analyze_comments(comments):
    sentiment_pipeline = get_sentiment_pipeline()
    results = []
    for comment in comments:
        comment_clean = clean_comment(comment)
        # Se o comentário for composto apenas de emojis, usamos a análise customizada
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
        # Tenta converter os rótulos mapeados para valores numéricos
        # Supondo que os rótulos customizados não sejam convertíveis, vamos usar os valores numéricos originais
        try:
            # Se o rótulo for algo como "5 stars", extraímos o valor
            stars = float(label.split()[0])
        except (ValueError, IndexError):
            # Se falhar, podemos atribuir um valor médio com base no rótulo customizado
            mapping = {"EXCELENTE": 5, "MEDIANO": 4, "RUIM": 3, "PÉSSIMO": 2}
            stars = mapping.get(label.upper(), 0)
        total += stars
        count += 1
    if count == 0:
        return None
    average = total / count
    return average

def generate_conclusion(average):
    if average is None:
        return "Infelizmente, não conseguimos coletar dados suficientes para chegar a uma conclusão. Tente novamente mais tarde."
    
    if average >= 4.50:
        rating = "EXCELENTE"
        extra_message = (
            "O público demonstrou um entusiasmo excepcional, indicando que seu conteúdo é altamente apreciado. "
            "Continue produzindo esse ótimo trabalho!"
        )
    elif average>=4.00:
        rating = "ÓTIMO"
        extra_message = (
            "A reação geral foi positiva, sugerindo que o conteúdo é bem recebido. Continue assim!"
        )
    elif average >= 3.50:
        rating = "BOM"
        extra_message = (
            "Os comentários indicam uma reação razoável. Há espaço para aprimoramento, mas você está no caminho certo."
        )
    elif average >= 2.50:
        rating = "RUIM"
        extra_message = (
            "A reação geral foi desfavorável. Talvez seja interessante revisar o conteúdo e buscar melhorias significativas."
        )
    elif average >= 1.50:
        rating = "PÉSSIMO"
        extra_message = (
            "O feedback foi extremamente negativo, sugerindo que o conteúdo não atendeu às expectativas. "
            "Reavalie o material e considere mudanças drásticas."
        )
    else:
        rating = "PÉSSIMO"
        extra_message = (
            "O feedback foi extremamente negativo, sugerindo que o conteúdo não atendeu às expectativas. "
            "Reavalie o material e considere mudanças drásticas."
        )
    
    conclusion = (
        f"Conclusão: A reação geral do público foi <strong>{rating}</strong> com uma média de {average:.2f} estrelas. "
        f"{extra_message}"
    )
    return conclusion

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        youtube_link = request.form.get('youtube_link')
        video_id = extract_video_id(youtube_link)
        if not video_id:
            return render_template('index.html', error="Link inválido. Por favor, insira um link válido do YouTube.")
        comments = get_comments(video_id, API_KEY, max_results=2000)
        sentiments = analyze_comments(comments)
        average = summarize_sentiments(sentiments)
        conclusion = generate_conclusion(average)
        return render_template('result.html', conclusion=conclusion, comments=comments, sentiments=sentiments, zip=zip)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
