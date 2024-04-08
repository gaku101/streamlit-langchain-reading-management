from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import StructuredTool
from dotenv import load_dotenv
import os
import streamlit as st
import csv


def main():
    load_dotenv()

    # Load the OpenAI API key from the environment variable
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True)
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }
    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

    st.set_page_config(page_title="Ask your CSV")
    st.header("Ask your CSV 📈")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")

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

        print(data)

        # プロンプトテンプレートの定義
        template = """
        {user_question}
        以下は質問者の読書履歴データです。データに基づいて上記の質問に答えてください。


        {data}

        """

        prompt = PromptTemplate(
            input_variables=["user_question", "data"],
            template=template,
        )

        # LLMチェーンの作成
        book_history = LLMChain(llm=chat, prompt=prompt)

        tools = load_tools(["bing-search"], llm=chat)
        tools = tools + [
            StructuredTool.from_function(
                func=lambda user_question: book_history.run(user_question=user_question, data=data),
                name="book_history",
                description="読書履歴データを元に、本に関するユーザーの質問に答える",
            )
        ]

        agent_chain = initialize_agent(
            tools, chat, agent=AgentType.OPENAI_FUNCTIONS, agent_kwargs=agent_kwargs, memory=memory, verbose=True
        )

        user_question = st.text_input("Ask a question about your CSV: ")
        print("user_question:", user_question)

        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                st.write(agent_chain.run(user_question))


if __name__ == "__main__":
    main()
