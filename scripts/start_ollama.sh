#!/bin/bash
set -e
OLLAMA_PORT="${PORT:-11434}"
export OLLAMA_HOST="0.0.0.0:${OLLAMA_PORT}"
ollama serve &
SERVE_PID=$!
echo "Waiting for ollama to be ready on port ${OLLAMA_PORT}..."
for i in $(seq 1 30); do
    if curl -sf "http://localhost:${OLLAMA_PORT}/api/tags" > /dev/null 2>&1; then
        echo "Ollama ready."
        break
    fi
    sleep 2
done
if ! ollama list | grep -q "llama3"; then
    echo "Pulling llama3..."
    ollama pull llama3
    echo "llama3 ready."
else
    echo "llama3 already present."
fi
wait $SERVE_PID
