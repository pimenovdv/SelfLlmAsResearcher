FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Установка рабочей директории агента
RUN mkdir -p /app/agent_workspace
COPY . .

# Ограничение прав
RUN useradd -m agentuser
RUN chown -R agentuser:agentuser /app/agent_workspace
USER agentuser

CMD ["python", "src/agent_loop.py"]
