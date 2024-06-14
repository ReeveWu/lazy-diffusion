import pandas as pd
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from utils.Embeddings import EmbeddingModel

df = pd.read_csv('./data/diffusion_prompts.csv')

df = df.drop_duplicates(subset='prompt', keep='first')

documents = []
for index, row in df.iterrows():
    page_content = row['prompt']
    metadata = {
        'url': row['url'],
        'source_site': row['source_site']
    }
    
    document = Document(
        page_content=page_content,
        metadata=metadata
    )
    
    documents.append(document)


embeddings = EmbeddingModel()
db = FAISS.from_documents(documents, embeddings)
print(db.index.ntotal)

db.save_local("./data/prompt_index")