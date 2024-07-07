import os
import box
import yaml
import getpass
import json
import torch

from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFium2Loader, Docx2txtLoader, TextLoader
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, GenerationConfig, pipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import RetrievalQA
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory

with open('./cfg/llm_cfg.yml', 'r', encoding='utf8') as config:
    cfg = box.Box(yaml.safe_load(config))

class PDFQuery:
    def __init__(self, model_name = 'TheBloke/Llama-2-13B-chat-GPTQ') -> None:
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE, chunk_overlap=cfg.CHUNK_OVERLAP)
        self.llm, self.prompt = self._setup_llm(model_name)
        self._contextualize_q_chain()
        self.embeddings = HuggingFaceInstructEmbeddings(model_name=cfg.EMBEDDINGS_MODEL,
                                                      model_kwargs={"device": "cuda"}, 
                                                      cache_folder=cfg.CACHE_DIR)
        
        self.chain = None
        self.db = None
    
    def _setup_llm(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(cfg.LLM_MODEL, 
                                                #   gguf_file=cfg.GGUF_FILE, 
                                                  use_fast=True)

        generation_config = GenerationConfig()
        generation_config.max_new_tokens = 8192
        generation_config.temperature = 0.0001
        generation_config.top_p = 0.95
        generation_config.do_sample = True
        generation_config.repetition_penalty = 1.5

        quantize_config = BaseQuantizeConfig(
            bits=4,  # quantize model to 4-bit
            group_size=128,  # it is recommended to set the value to 128
            desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
        )
                
        model = AutoGPTQForCausalLM.from_quantized(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    cache_dir=cfg.CACHE_DIR,
                    quantization_config=quantize_config,
                )
        print('Loading model from huggingface hub')
        text_pipeline = pipeline(
            task = 'text-generation',
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
        )

        template = """
System: Provide information from the provided context and no yapping.

{context}

User: {question} 

Assistant:
"""
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )

        return HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0}), prompt
    
    def _contextualize_q_chain(self):
        history_packed_template = """System: Given a chat history and the follow up user question, \
formulate a standalone question which can be understood without the chat history. \
Return the original question if it is not related to the chat history. 
DO NOT add any external knowledge other than the chat history and the question provided.\n\

Chat History: {chat_history}
User: follow-up question: {question}
Assistant:"""
        contextualize_q_prompt = PromptTemplate(
            template=history_packed_template,
            input_variables=["chat_history", "question"],
        )
        self.subchain = contextualize_q_prompt | self.llm | StrOutputParser()
        self.chat_history = ChatMessageHistory(messages=[])

    def _contextualized_question(self, input: dict):
        if input.get("chat_history"):
            new_question = self.subchain.invoke({'chat_history': self._parse_chat_history(input["chat_history"]), 'question': input["question"]})
            print('NEW QUESTION:', new_question, 'HISTORY LENGTH:', len(input["chat_history"]))
            new_question = self._postprocess_ans(new_question)
            return new_question
        else:
            print('ORIGINAL QUESTION')
            return input["question"]

    def _parse_chat_history(self, chat_history: dict):
        return "\n".join([f"{message.type}: {message.content}" for message in chat_history])

    def _postprocess_ie(self, ie_output: str):
        ie_output = self._postprocess_ans(ie_output)
        print('JSON OBJ:', ie_output)
        is_json = False
        try:
            json_obj = json.loads(ie_output)
            is_json = True
        except:
            try:
                import IPython; IPython.embed();
                if '```' in ie_output:
                    ie_output = ie_output.split('```')[-2].strip()
                kv_s = ie_output.strip('{}').split(':')
                len_kv_s = len(kv_s) // 2 * 2
                
                json_obj = {}
                next_k = None
                for i in range(len_kv_s):
                    if i == 0:
                    # if i % 2 == 0:
                        key = kv_s[i].strip('\"')
                    else:
                        key = next_k

                    next_k = kv_s[i+1].strip(' ,\"').split(', ')[-1].strip(' ,\"')
                    val = kv_s[i+1].strip(' ,\"')[:-len(next_k)].strip(' ,\"')
                    json_obj[key] = val
                is_json = True
            except:
                print('return string')
        
        return str(json_obj) if is_json else ie_output

    def _postprocess_ans(self, ans_output: str):
        final_ans = ans_output.split('Assistant:')[-1].strip()
        for char in ['“', '”', '„']:
            final_ans = final_ans.replace(char, '\"')
        final_ans = final_ans.replace('\\', '')
        return final_ans

    def ask(self, question: str, is_extract: bool, use_chat_history: bool) -> str:
        if self.chain is None:
            response = "Please, add a document."
        else:
            if use_chat_history:
                print('USE CHAT HISTORY')
                new_question = self._contextualized_question({'question': question, 'chat_history': self.chat_history.messages, 'is_extract': is_extract})
            else:
                new_question = question
            # response = self.chain.invoke({'query': new_question})["result"][len(new_question):].strip()
            response = self.chain(new_question)["result"].strip()
            if is_extract:
                # response = self._postprocess_ie(response)
                response = self._postprocess_ans(response)
            else:
                response = self._postprocess_ans(response)
            
            if len(self.chat_history.messages) >= cfg.MAX_HIST_LEN:
                self.chat_history.clear()
                
            self.chat_history.add_messages([HumanMessage(content=question), AIMessage(content=response)])
        return response

    def ingest(self, file_path: os.PathLike, file_extension='pdf') -> None:
        if file_extension == ".txt":
            loader = TextLoader(file_path)
        elif file_extension in [".doc", ".docx"]:
            loader = Docx2txtLoader(file_path)
        else:
            loader = PyPDFium2Loader(file_path)
        
        def _format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        documents = loader.load()
        splitted_documents = self.text_splitter.split_documents(documents)
        # heuristically - 2 top chunks contains the most important information about the document
        self.ie_context = splitted_documents[0].page_content + splitted_documents[1].page_content
        
        db = FAISS.from_documents(splitted_documents, self.embeddings).as_retriever()
        
        # import IPython; IPython.embed(); exit(0)
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="stuff",
            retriever=db,
            # retriever=compression_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )
        query = 'Please summarize the given document in a precise language'
        return self._postprocess_ans(self.chain(query)["result"].strip())
        
    def forget(self) -> None:
        self.db = None
        self.chain = None
        self.chat_history.clear()

if __name__ == "__main__":
    # os.environ["LANGCHAIN_TRACING_V2"] = "true"
    # os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
    # os.environ["LANGCHAIN_API_KEY"] = 'ls__72a643ed0bf043068f465eb5b38b2578'
    # print('API KEY:', os.environ["LANGCHAIN_API_KEY"])

    pdf = PDFQuery(model_name=cfg.LLM_MODEL)
    print(pdf.ingest("db/sample.pdf"))
    print('START____')
    import IPython; IPython.embed(); exit(0)