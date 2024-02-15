import os
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFium2Loader, Docx2txtLoader, TextLoader
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory


class PDFQuery:
    def __init__(self, model_name = 'TheBloke/Llama-2-13B-chat-GPTQ') -> None:
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        self.llm, self.prompt = self._setup_llm(model_name)
        self._contextualize_q_chain()
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
        generation_config.repetition_penalty = 1.3

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
        )

        template = """
<s>[INST] <<SYS>>
You are an Information Extractor. Extract the information as accurately as possible from the provided context. Provide NOTHING other than what is asked.
<</SYS>>

{context}
 
{question} [/INST]
"""
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )

        return HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0}), prompt
    
    def _contextualize_q_chain(self):
        history_packed_template = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Return the original question if it is not related to the chat history. \
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

        contextualize_q_prompt = PromptTemplate(
            template=history_packed_template,
            input_variables=["chat_history", "question"],
        )
        contextualize_q_chain = contextualize_q_prompt | self.llm | StrOutputParser()
        self.subchain = contextualize_q_chain
        self.chat_history = ChatMessageHistory(messages=[])

    def _contextualized_question(self, input: dict):
        if input.get("chat_history"):
            new_question = self.subchain.invoke(input)
            print('NEW QUESTION:', new_question, 'HISTORY LENGTH:', len(input["chat_history"]))
            return new_question
        else:
            return input["question"]
    
    def _test_subchain(self):
        return self.subchain.invoke(
            {
                "chat_history": [
                    HumanMessage(content="What does LLM stand for?"),
                    AIMessage(content="Large language model"),
                ],
                "question": "What is meant by large",
            }
        )
    
    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def ask(self, question: str) -> str:
        if self.chain is None:
            response = "Please, add a document."
        else:
            new_question = self._contextualized_question({'question': question, 'chat_history': self.chat_history.messages})
            response = self.chain.invoke({'query': new_question})["result"].strip()
            self.chat_history.add_messages([HumanMessage(content=question), AIMessage(content=response)])
        return response

    def ingest(self, file_path: os.PathLike, file_extension='pdf') -> None:
        if file_extension == ".txt":
            loader = TextLoader(file_path)
        elif file_extension in [".doc", ".docx"]:
            loader = Docx2txtLoader(file_path)
        else:
            loader = PyPDFium2Loader(file_path)
        documents = loader.load()
        splitted_documents = self.text_splitter.split_documents(documents)
        self.db = Chroma.from_documents(splitted_documents, self.embeddings).as_retriever(search_kwargs={"k": 2})
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="stuff",
            retriever=self.db,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )

        return self.chain.invoke({'query': 'Please summarize the given document in a precise language'})["result"].strip()

    def forget(self) -> None:
        self.db = None
        self.chain = None
        self.chat_history.clear()

if __name__ == "__main__":
    pdf = PDFQuery(model_name='anhduy0911/LLM_Healthcare_Information_Extraction')
    print(pdf.ingest("db/sample.pdf"))
    print('START____')
    import IPython; IPython.embed(); exit(0)