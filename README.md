# YouTube Comments Analysis

Este projeto é uma aplicação web desenvolvida em Python utilizando Flask, que extrai comentários de vídeos do YouTube e realiza uma análise de sentimentos. A aplicação processa os comentários, atribui rótulos como **EXCELENTE**, **MEDIANO**, **RUIM** e **PÉSSIMO** para cada comentário, e apresenta uma conclusão geral da reação do público.

## Funcionalidades

- **Extração de Comentários:** Utiliza a API do YouTube para coletar até 2000 comentários de um vídeo.
- **Análise de Sentimentos:** Emprega modelos de análise de sentimentos (idealmente otimizados para o português) para classificar cada comentário.
- **Tratamento de Emojis:** Implementa uma análise customizada para comentários compostos predominantemente por emojis.
- **Interface Web:** Interface responsiva desenvolvida com Flask e Bootstrap, com um botão customizado para exibir ou ocultar os comentários analisados.
- **Conclusão Personalizada:** Gera uma mensagem final chamativa com base na média dos sentimentos dos comentários.

## Requisitos

- Python 3.7 ou superior
- Flask
- google-api-python-client
- transformers
- torch

## Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/youtube-comments-analysis.git
   cd youtube-comments-analysis
2. Crie e ative um ambiente virtual (opcional, mas recomendado):
   ´´´bash
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
3. Instale as dependências
   ´´´bash
   pip install -r requirements.txt
4. Configure sua API Key do YouTube:
   Crie um arquivo chamado api_key.py na raiz do projeto e insira:
   ```python
   API_KEY = "SUA_API_KEY_AQUI"

## Uso
1. Execute a aplicação
   ´´´bash
   python main.py
2. Acesse a aplicação no navegador através do endereço http://127.0.0.1:5000/.
3. Insira o link de um vídeo do YouTube e submeta o formulário para ver a análise dos comentários e a conclusão personalizada.