import os 
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine



load_dotenv()

my_api_key = os.environ.get("GOOGLE_API_KEY")

llm = GoogleGenAI("gemini-2.0-flash", api_key=my_api_key)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

Settings.llm = llm
Settings.embed_model = embed_model

documents_1 = SimpleDirectoryReader(input_files=["data/multi_query_data/origin.pdf"]).load_data()

documents_2 = SimpleDirectoryReader(input_files=["data/multi_query_data/survey.pdf"]).load_data()

splitter = SentenceSplitter(chunk_size=512,
                          chunk_overlap=20)

nodes_1 = splitter.get_nodes_from_documents(documents_1)
nodes_2 = splitter.get_nodes_from_documents(documents_2)

index_1 = VectorStoreIndex(nodes_1)
index_2 = VectorStoreIndex(nodes_2)

retriever = QueryFusionRetriever(
    retrievers=[index_1.as_retriever(), index_2.as_retriever()],
    similarity_top_k=2,
    num_queries=4,
    use_async=True,
    verbose=True
)

import nest_asyncio

nest_asyncio.apply()

synthesizer = get_response_synthesizer(streaming=True)

query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=synthesizer)

response = query_engine.query("tell me what is transformer in simplest words possible.")

response.print_response_stream()