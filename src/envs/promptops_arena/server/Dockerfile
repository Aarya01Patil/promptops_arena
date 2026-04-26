FROM python:3.11-slim

WORKDIR /app

COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r /app/server/requirements.txt

# Copy the env package
COPY . /app/promptops_arena
ENV PYTHONPATH=/app

EXPOSE 8000
ENV PROMPTOPS_LLM_BACKEND=transformers

CMD ["uvicorn", "promptops_arena.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
