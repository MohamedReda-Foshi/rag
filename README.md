this is my project of jobingteck ai & machine learning
on FST Errachidia

# project:
#retrieval augmented generation

    Search first. Generate second.

tools:
llm: qwen2.5:3b  (local llm)
langChain
database:Chromabd
flash:webserver
## dependencies

### Create and activate virtual environment
python -m venv venv
source venv/bin/activate.fish    # Fish shell
### source venv/bin/activate        # Bash/Zsh

### Install dependencies
pip install langchain langchain_community langchain_chroma \
            chromadb pypdf flask ollama


### Pull models via Ollama
ollama pull qwen2.5:3b
ollama pull nomic-embed-text



# quit start add your pdf file to `data/pdf` 
then run
``python ingest.py``
and this step we add file and store to database (vectordb) 
to run it
``python core_rag.py``

for web
``server.py``


## llm problem
- cutoff problem (old data)
- hallucination ()
- Domain-specific expertise ()
- source attribution ()
- Private data handling (security)

  ### token:(token mean wold)
  the longer the context the more degradation you get 

# what rag come to fix

we split the data to give use Higher quality and more accurate

Large data causes hallucinations

before we give me file we split datafile and we call it chunks

to store those chunk(and main idea ) we need database (vector database) we use Chroma

vector database
what is vector embeddings:is a way to present data (text,image) as data points
to change it to array of number and keep the data meaning using unstructured  



vedata is way to store ton using number array that can ml process

explain
retrieval: is semantic search is mean  searching my the meaning of the word in prompt
augmented:

## problem solution
```source venv/bin/activate.fish                                   pip install langchain_community langchain_chroma pypdf```
