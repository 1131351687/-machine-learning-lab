from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from langsmith.schemas import DatasetShareSchema

from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatTongyi
from file_history_store import get_history



class RagService(object):
    def __init__(self):
        self.vector_service=VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )
        self.prompt_template=ChatPromptTemplate(
            [
                ("system","以我提供的资料为主,"
                 "简洁专业回答问题。参考资料:{context}"),
                ("system","并且我提供用户的对话历史记录，如下："),
                MessagesPlaceholder("history"),
                ('user',"请回答用户提问:{input}")

            ]
        )
        self.chat_model=ChatTongyi(model=config.chat_model_name)
        self.chain=self.__get_chain()

    def __get_chain(self):
        """
        获取最终的执行链
        """
        retriever=self.vector_service.get_retriever()

        def format_document(docs):
            """
            格式化检索到的文档
            """
            if not docs:
                return "没有相关资料"
            formatted_str=''
            for doc in docs:
                formatted_str+=f'文档片段:{doc.page_content}\n文档源数据:{doc.metadata}\n'
            return formatted_str

        def format_for_retriever(value:dict)-> str:
            """
            格式化输入，供检索器使用
            """
            return value['input']

        def format_for_prompt_template(value:dict)-> dict:
            """
            格式化输入，供prompt_template使用
            """
            return {
                'input':value['input']['input'],
                'context':value['context'],
                'history':value['input']['history']
            }

        chain=(
            {
                'input':RunnablePassthrough(),
                'context':RunnableLambda(format_for_retriever) | retriever | format_document
            }  | RunnableLambda(format_for_prompt_template) |
            self.prompt_template | self.chat_model |
            StrOutputParser()
        )

        conversation_chain=RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key='input',
            history_messages_key='history'
        )

        return conversation_chain

if __name__=="__main__":
    res=RagService().chain.invoke({'input':"人类如何养"}, config.session_config)
    print(res)