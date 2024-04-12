from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory, MomentoChatMessageHistory
from langchain.prompts import MessagesPlaceholder
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.tools import StructuredTool
from dotenv import load_dotenv
import os
import streamlit as st
import csv
from datetime import timedelta
import uuid

session_id = str(uuid.uuid4())


def create_tools(data, chat):

    # プロンプトテンプレートの定義
    template = """
        {user_question}
        以下は質問者の読書履歴データです。データに基づいて上記の質問に答えてください。

        {data}

        """

    prompt = ChatPromptTemplate.from_template(template)

    # LLMチェーンの作成
    # book_history = LLMChain(llm=chat, prompt=prompt)
    book_history = prompt | chat
    tools = load_tools(["bing-search"], llm=chat)
    return tools + [
        StructuredTool.from_function(
            func=lambda user_question: book_history.invoke({"user_question": user_question, "data": data}),
            name="book_history",
            description="読書履歴データを元に、本に関するユーザーの質問に答える",
        )
    ]


def main():
    load_dotenv()

    chat = ChatOpenAI(model="gpt-3.5-turo", temperature=0, streaming=True)

    # memory関連
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }
    history = MomentoChatMessageHistory.from_client_params(
        session_id,
        os.environ["MOMENTO_CACHE"],
        timedelta(hours=int(os.environ["MOMENTO_TTL"])),
    )
    memory = ConversationBufferMemory(chat_memory=history, memory_key="memory", return_messages=True)

    # タイトルとアップローダーの生成
    st.set_page_config(page_title="読書管理")
    st.header("読書管理 📖")

    csv_file = st.file_uploader(label="Choose a CSV file", type="csv")

    data = []

    if csv_file is not None:
        # UploadedFile オブジェクトをテキストモードで開く
        csv_file = csv_file.read().decode("utf-8")
        csv_data = csv.reader(csv_file.splitlines())

        next(csv_data)  # ヘッダー行をスキップ
        for i, row in enumerate(csv_data):
            if i >= 10:
                break
            book = {"title": row[0], "author": row[2], "publisher": row[4], "completed_date": row[6]}
            data.append(book)

        # data取得完了後toolsのセットアップとagentの初期化
        tools = create_tools(data, chat)
        agent_chain = initialize_agent(
                tools, chat, agent=AgentType.OPENAI_FUNCTIONS, agent_kwargs=agent_kwargs, memory=memory, verbose=True
            )

        user_question = st.text_input("Ask a question about your CSV: ")
        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                st.write(agent_chain.run("invoke book_history or bing-search only" + user_question))


if __name__ == "__main__":
    main()
