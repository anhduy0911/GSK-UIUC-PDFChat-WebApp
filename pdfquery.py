import os
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFium2Loader, Docx2txtLoader, TextLoader
# from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
# from langchain.llms import OpenAI
import torch
from langchain import HuggingFacePipeline, PromptTemplate
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from accelerate import load_checkpoint_and_dispatch
# from langchain.chat_models import ChatOpenAI


class PDFQuery:
    def __init__(self, model_name = 'TheBloke/Llama-2-13B-chat-GPTQ') -> None:
        # self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        # os.environ["OPENAI_API_KEY"] = openai_api_key
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        # self.llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
        self.llm, self.prompt = self._setup_llm(model_name)
        self.embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                      model_kwargs={"device": "cuda"}, 
                                                      cache_folder='/scratch/bcgd/duyan2/.cache')
        
        self.chain = None
        self.db = None
    
    def _setup_llm(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        generation_config = GenerationConfig.from_pretrained(model_name)
        generation_config.max_new_tokens = 1024
        generation_config.temperature = 0.0001
        generation_config.top_p = 0.95
        generation_config.do_sample = True
        generation_config.repetition_penalty = 1.15

        model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    cache_dir='/scratch/bcgd/duyan2/.cache',
                    # disable_exllama=True
                )
        print('Loading model from huggingface hub')
        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            # max_new_tokens=512,
            # do_sample=True,
            # temperature=0.7,
            # top_p=0.95,
            # top_k=40,
            # repetition_penalty=1.1
        )

        template = """
<s>[INST] <<SYS>>
Act as a Machine Learning engineer. Extract the information as accurately as possible from the provided context. Provide nothing further than what is asked.
<</SYS>>

{context}
 
{question} [/INST]
"""
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )
        # return tokenizer, model, HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0}), prompt
        return HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0}), prompt
        

    def ask(self, question: str) -> str:
        if self.chain is None:
            response = "Please, add a document."
        else:
            # docs = self.db.get_relevant_documents(question)
            # response = self.chain.run(input_documents=docs, question=question)
            response = self.chain(question)["result"].strip()
        return response

    def ingest(self, file_path: os.PathLike, file_extension='pdf') -> None:
        print(file_extension)
        if file_extension == ".txt":
            loader = TextLoader(file_path)
        elif file_extension in [".doc", ".docx"]:
            loader = Docx2txtLoader(file_path)
        else:
            loader = PyPDFium2Loader(file_path)
        documents = loader.load()
        splitted_documents = self.text_splitter.split_documents(documents)
        self.db = Chroma.from_documents(splitted_documents, self.embeddings).as_retriever(search_kwargs={"k": 2})
        # self.db = Chroma.from_documents(splitted_documents, self.embeddings, persist_directory='db').as_retriever(search_kwargs={"k": 2})
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="stuff",
            retriever=self.db,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )
        
        return self.chain('Please summarize the given document in a precise language')["result"].strip()

    def forget(self) -> None:
        self.db = None
        self.chain = None

if __name__ == "__main__":
    pdf = PDFQuery(model_name='anhduy0911/LLM_Healthcare_Information_Extraction')
    pdf.ingest("db/test.docx")
    # import IPython; IPython.embed(); exit(1)