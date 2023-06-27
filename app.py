import os
import time
from llama_index import SimpleDirectoryReader, LLMPredictor,ServiceContext,VectorStoreIndex,StorageContext,load_index_from_storage
from dotenv import load_dotenv
from langchain import OpenAI
import openai
from flask import Flask,request,render_template

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

class App:

    def __init__(self) -> None:
        self.llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003",max_tokens=1800))
        self.service_context = ServiceContext.from_defaults(llm_predictor=self.llm_predictor)

    # 查询本地索引
    def query_index(self,prompt):
        # rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
       
        # load index
        index = load_index_from_storage(storage_context)

        query_engine = index.as_query_engine()
       
        response = query_engine.query(prompt)

        return response

    # 建立本地索引
    def create_index(self):
       # 读取数据
       documents = SimpleDirectoryReader(input_dir=os.environ["DATA_DIR"], 
       required_exts=os.environ["DATA_FILTER"].split(','),
       recursive=True,exclude_hidden=True).load_data()

       # 建立索引
       index = VectorStoreIndex.from_documents(documents=documents,service_context=self.service_context)

       # 保存索引
       index.storage_context.persist()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/create_index', methods=['POST'])
def create_index():
    llama = App()
    llama.create_index()
    return 'create_index success'

@app.route('/query_index/', methods=['GET'])
def query_index():
    # 获取request参数prompt
    prompt = request.args.get('prompt')
    print('prompt:\n',prompt)

    llama = App()
    response = llama.query_index(prompt + '\n 最后基于我提的问题联想两个不同维度的问题，放在最后展示')

    print('response:\n',response)
    return response.response

if __name__ == '__main__':
    app.run()
