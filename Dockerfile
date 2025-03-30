FROM python:3.12-slim

# Configurar diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema necessárias
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"


# Copiar os arquivos de dependência primeiro para aproveitar o cache do Docker
COPY requirements.txt .

# Instalar as dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Crie e configure o diretório para armazenar os resultados das análises
RUN mkdir -p /app/analysis_results

# Copiar o código-fonte da aplicação
COPY . .

# Porta onde o Flask será executado
EXPOSE 5000

# Comando para iniciar a aplicação
CMD ["python", "main.py"]