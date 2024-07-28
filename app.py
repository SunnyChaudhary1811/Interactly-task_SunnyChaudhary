import os
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import chainlit as cl
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load and preprocess data
df = pd.read_csv(r"D:\Interactly task_SunnyChaudhary\RecruterPilot candidate sample input dataset - Sheet1.csv")
df['combined'] = df.apply(lambda row: f"Name: {row['Name']}\nContact: {row['Contact Details']}\nLocation: {row['Location']}\nSkills: {row['Job Skills']}\nExperience: {row['Experience']}\nProjects: {row['Projects']}\nComments: {row['Comments']}", axis=1)

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings()

# Create FAISS index
texts = df['combined'].tolist()
faiss_index = FAISS.from_texts(texts, embeddings)

# Load model locally
model_id = "google/flan-t5-base"  # Using a smaller model for faster local inference
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# Create pipeline
pipe = pipeline(
    "text2text-generation",
    model=model, 
    tokenizer=tokenizer, 
    max_length=1024,
    temperature=0.5
)

# Initialize local Hugging Face model
llm = HuggingFacePipeline(pipeline=pipe)

# Define RAG prompt template
rag_template = """
You are an AI assistant tasked with matching job descriptions to candidate profiles.
Use the following candidate profiles to answer the question:
{context}

Question: Given this job description, rank the top 5 candidates and explain why they are suitable:
{question}

Answer: Provide your analysis and ranking in the following format:
1. [Candidate Name] - [Brief explanation of suitability]
2. [Candidate Name] - [Brief explanation of suitability]
...
"""

# Create RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=faiss_index.as_retriever(search_kwargs={"k": 10}),
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": PromptTemplate(
            input_variables=["context", "question"],
            template=rag_template
        )
    }
)

@cl.on_chat_start
def start():
    cl.user_session.set("rag_chain", rag_chain)

@cl.on_message
async def main(message: cl.Message):
    rag_chain = cl.user_session.get("rag_chain")

    # Generate response using the RAG chain
    result = rag_chain({"query": message.content})

    # Send the result
    await cl.Message(content=result["result"]).send()

    # Optionally, display the source documents used
    sources = [doc.page_content for doc in result["source_documents"]]
    await cl.Message(content=f"Sources used:\n\n" + "\n\n".join(sources)).send()

if __name__ == "__main__":
    cl.run()