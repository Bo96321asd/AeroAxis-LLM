from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import warnings
import logging
import random
import re
import Config

logging.getLogger("langchain_core").setLevel(logging.ERROR) 
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

loader = DirectoryLoader(
    path = "./Knowledge_base",
    glob = "**/*.txt", 
    loader_cls = TextLoader, 
    show_progress = True,
    loader_kwargs = {"encoding": "utf-8"},
    use_multithreading = True,
    silent_errors = True,
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100,
    add_start_index = True,
    strip_whitespace = True,
    separators = ["\n\n", "\n", " ", ""]
)
splits = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name = "all-MiniLM-L6-v2",
    encode_kwargs = {'normalize_embeddings': True} 
)

vector_database = FAISS.from_documents(
    # inputs
    # Begin
    documents = splits,
    embedding = embeddings,
    # End
    distance_strategy = DistanceStrategy.COSINE
)

retriever = vector_database.as_retriever(
    search_type = "similarity", 
    search_kwargs = {"k": 5}    
)

prompt = ChatPromptTemplate.from_template(Config.template)

LLM = ChatOllama(
    model = "llama3.1",
    temperature = 0,
    top_p = 0.9,
    num_predict = 512,
    repeat_penalty = 1.1
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | LLM
    | StrOutputParser()
)

last_prompt = None # In case LLM give the same question 
print("=" * 50) # Split log and performance 
print(f"{random.choice(Config.open_message)} :3")
while True:
    question = input()
    command = question.strip().lower()
    # None prompt
    if not command:
        print("I did not get anything. Could you type the question again? :<")
        continue
    # Stop
    if command in Config.stop_commands:
        print(f"{random.choice(Config.exit_message)} :3")
        break
    # Short prompt
    elif command in Config.short_commands:
        print(f"{random.choice(Config.ready_message)} :3")
        continue 
    # Avoid hallucination
    elif not re.search('[a-zA-Z0-9]', command):
        print("I did not get anything. Could you type the question again? :<")
    # Normal prompt
    elif command:
        try:
            answer = rag_chain.invoke(question)
            print(answer)
        except Exception as e:
            print("Error: Unable to process your request right now. :<")
        # Follow-up question
        next_prompt = f"{random.choice(Config.follow_up_question)} :3"
        while next_prompt == last_prompt:
            next_prompt = f"{random.choice(Config.follow_up_question)} :3"
        print(next_prompt)
        last_prompt = next_prompt
