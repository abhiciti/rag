import os
import torch
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- 1. Load and Parse PDF Documents ---
pdf_paths = ["AST-1.pdf", "AST-2.pdf"]
documents = []

print("Loading PDF Documents...")
for path in pdf_paths:
    if os.path.exists(path):
        loader = PyPDFLoader(path)
        documents.extend(loader.load())
    else:
        print(f"Warning: {path} not found in the root folder.")

# --- 2. Chunk Documents ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
splits = text_splitter.split_documents(documents)
print(f"Total chunks generated: {len(splits)}")

# --- 3. Initialize Embedding Model & Chroma Vector Store ---
print("Initializing embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

print("Building vector database...")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Configure retrievers
retriever_similarity = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
retriever_mmr = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 8})

# --- 4. Load Open-Source LLM Locally ---
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"Loading local LLM: {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
device_map = "auto" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    device_map=device_map
)

local_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.2,
    do_sample=True
)

def call_local_llm(prompt_text):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer the question directly using only the provided context."},
        {"role": "user", "content": prompt_text}
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    outputs = local_generator(formatted_prompt)
    generated_text = outputs[0]["generated_text"]
    response = generated_text[len(formatted_prompt):]
    return response.strip()

# --- 5. RAG Pipeline Logic ---
prompt_template = "\"\"\"Use the following pieces of retrieved context to answer the question at the end.
If you do not know the answer, say that you don't know.

Context:
{context}

Question: {question}"""
QA_PROMPT = PromptTemplate.from_template(prompt_template)

def rag_pipeline(query, search_method="similarity"):
    retriever = retriever_mmr if search_method == "mmr" else retriever_similarity
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    full_prompt = QA_PROMPT.format(context=context, question=query)
    llm_response = call_local_llm(full_prompt)
    return llm_response, docs

# --- 6. Gradio Interface Definition ---
def gradio_rag_interface(query, search_method):
    answer, retrieved_docs = rag_pipeline(query, search_method.lower())
    
    # Format source documents metadata for display
    sources_text = ""
    for i, doc in enumerate(retrieved_docs, 1):
        source_name = doc.metadata.get('source', 'Unknown source')
        page = doc.metadata.get('page', 'N/A')
        sources_text += f"**Source {i}:** {source_name} (Page {page})\n"
        sources_text += f"```\n{doc.page_content[:200]}...\n```\n\n"
        
    return answer, sources_text

demo = gr.Interface(
    fn=gradio_rag_interface,
    inputs=[
        gr.Textbox(label="Enter your query here:", placeholder="e.g., What is discussed in AST-1?"),
        gr.Radio(choices=["Similarity", "MMR"], value="Similarity", label="Retrieval Strategy")
    ],
    outputs=[
        gr.Textbox(label="RAG Answer", interactive=False),
        gr.Markdown(label="Retrieved Sources Context")
    ],
    title="PDF Knowledge Base RAG Assistant",
    description="Interact with the content of AST-1.pdf and AST-2.pdf using advanced retriever setups!"
)

# Launch app locally
if __name__ == "__main__":
    demo.launch()
