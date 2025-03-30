import os
import json
from datetime import datetime
from statistics import mean
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace


def create_simple_tokenizer():
    """
    Cria um tokenizador simples baseado em BPE.
    """
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def count_tokens(text, tokenizer=None):
    """
    Conta a quantidade de tokens em um texto usando um tokenizador.

    Args:
        text (str): O texto para contar tokens
        tokenizer: O tokenizador a ser usado

    Returns:
        int: O número de tokens no texto
    """
    if tokenizer is None:
        tokenizer = Tokenizer.from_pretrained("bert-base-cased")

    # print(tokenizer)
    print(text)
    encoding = tokenizer.encode(text)
    print(encoding)
    print(len(encoding.tokens))
    return len(encoding.tokens)


def analyze_tokens_in_comments(comments, output_file=None, verbose=True):
    """
    Analisa a quantidade de tokens em uma lista de comentários.

    Args:
        comments (list): Lista de comentários (strings)
        output_file (str, optional): Caminho para salvar o relatório
        verbose (bool): Se True, imprime resultados no console

    Returns:
        dict: Um dicionário com as estatísticas da análise
    """
    if not comments:
        return {"error": "No comments provided for analysis"}

    # Criar tokenizador uma vez para reutilizar
    tokenizer = Tokenizer.from_pretrained("bert-base-cased")

    # Contar tokens para cada comentário
    token_counts = [count_tokens(comment, tokenizer) for comment in comments]

    # Calcular estatísticas
    total_comments = len(comments)
    total_tokens = sum(token_counts)
    avg_tokens = mean(token_counts) if token_counts else 0
    max_tokens = max(token_counts) if token_counts else 0
    min_tokens = min(token_counts) if token_counts else 0

    # Distribuição de tokens
    distribution = {
        "1-10": len([c for c in token_counts if 1 <= c <= 10]),
        "11-25": len([c for c in token_counts if 11 <= c <= 25]),
        "26-50": len([c for c in token_counts if 26 <= c <= 50]),
        "51-100": len([c for c in token_counts if 51 <= c <= 100]),
        "101-250": len([c for c in token_counts if 101 <= c <= 250]),
        "250+": len([c for c in token_counts if c > 250]),
    }

    # Criar relatório
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_comments": total_comments,
        "total_tokens": total_tokens,
        "average_tokens_per_comment": avg_tokens,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "token_distribution": distribution,
        "cost_estimate": {
            "input_tokens_per_1000_comments": avg_tokens * 1000,
            "gpt-3.5-turbo_cost_per_1000_comments": (avg_tokens * 1000 * 0.0015)
            / 1000,  # $0.0015 por 1K tokens
            "gpt-4_cost_per_1000_comments": (avg_tokens * 1000 * 0.03)
            / 1000,  # $0.03 por 1K tokens
            "claude-3-sonnet_cost_per_1000_comments": (avg_tokens * 1000 * 0.03)
            / 1000,  # $0.03 por 1K tokens (estimativa)
        },
    }

    # Imprimir resultados se verbose
    if verbose:
        print("===== Token Analysis Report =====")
        print(f"Total Comments: {total_comments}")
        print(f"Total Tokens: {total_tokens}")
        print(f"Average Tokens per Comment: {avg_tokens:.2f}")
        print(f"Min Tokens: {min_tokens}")
        print(f"Max Tokens: {max_tokens}")
        print("Token Distribution:")
        for range_name, count in distribution.items():
            print(
                f"  {range_name}: {count} comments ({count / total_comments * 100:.1f}%)"
            )
        print("Cost Estimate:")
        print(
            f"  Input Tokens per 1000 Comments: {report['cost_estimate']['input_tokens_per_1000_comments']:.0f}"
        )
        print(
            f"  GPT-3.5 Turbo Cost per 1000 Comments: ${report['cost_estimate']['gpt-3.5-turbo_cost_per_1000_comments']:.2f}"
        )
        print(
            f"  GPT-4 Cost per 1000 Comments: ${report['cost_estimate']['gpt-4_cost_per_1000_comments']:.2f}"
        )
        print(
            f"  Claude 3 Sonnet Cost per 1000 Comments: ${report['cost_estimate']['claude-3-sonnet_cost_per_1000_comments']:.2f}"
        )

    # Salvar em arquivo
    if output_file:
        directory = os.path.dirname(output_file)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=4)

        if verbose:
            print(f"Report saved to: {output_file}")

    return report


def analyze_comments_from_analysis_files(
    directory="analysis_results", output_file=None
):
    """
    Analisa tokens de comentários de todos os arquivos de análise existentes.

    Args:
        directory (str): Diretório com os arquivos de análise
        output_file (str, optional): Caminho para salvar o relatório agregado

    Returns:
        dict: Um dicionário com as estatísticas da análise
    """
    if not os.path.exists(directory):
        return {"error": f"Directory {directory} not found"}

    all_comments = []

    # Percorrer todos os arquivos no diretório
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                try:
                    filepath = os.path.join(root, file)
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Extrair comentários do arquivo
                    if "comments_analysis" in data:
                        for item in data["comments_analysis"]:
                            if "comment" in item:
                                all_comments.append(item["comment"])
                except Exception as e:
                    print(f"Error processing file {file}: {e}")

    # Usar a função de análise para o conjunto agregado de comentários
    if not output_file:
        output_file = f"token_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    return analyze_tokens_in_comments(all_comments, output_file)
