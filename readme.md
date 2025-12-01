A simple LangChain-based RAG system that allows you to talk with an LLM assistant about your papers.

Simply add your papers in `/papers`. Then pull the models:

```bash
ollama pull qwen3:0.6b
ollama pull model all-minilm
```

Now you're ready to run:

```bash
python main.py 
```