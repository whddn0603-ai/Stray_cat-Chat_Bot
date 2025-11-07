import streamlit as st

st.title("🐈 길고양이 돌보미 챗봇")
st.write(
    "귀엽다고 함부로 만지지 마시오, 이것이 마지막 경고요.")
import os
import streamlit as st
import tempfile

from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor

# --------------------------------------------------------------------
# 1. Web Search Tool
# --------------------------------------------------------------------
def search_web():
    return TavilySearchResults(k=6, name="web_search")


# --------------------------------------------------------------------
# 2. PDF Tool
# --------------------------------------------------------------------
def load_pdf_files(uploaded_files):
    all_documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        all_documents.extend(documents)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(all_documents)

    vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    retriever = vector.as_retriever(search_kwargs={"k": 5})

    retriever_tool = create_retriever_tool(
        retriever,
        name="pdf_search",
        description="This tool gives you direct access to the uploaded PDF documents. "
                    "Always use this tool first when the question might be answered from the PDFs."
    )
    return retriever_tool


# --------------------------------------------------------------------
# 3. Agent + Prompt 구성
# --------------------------------------------------------------------
def build_agent(tools):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant for KIBO employees. "
         "First, always try `pdf_search`. "
         "If `pdf_search` returns no relevant results, immediately call ONLY `web_search`. "
         "Never mix the two tools. "
         "Answer in Korean with a professional and friendly tone, including emojis."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

    return agent_executor


# --------------------------------------------------------------------
# 4. Agent 실행 함수 (툴 사용 내역 제거)
# --------------------------------------------------------------------
def ask_agent(agent_executor, question: str):
    result = agent_executor.invoke({"input": question})
    answer = result["output"]

    # intermediate_steps에서 마지막만 가져오기
    if result.get("intermediate_steps"):
        last_action, _ = result["intermediate_steps"][-1]
        answer += f"\n\n출처:\n- Tool: {last_action.tool}, Query: {last_action.tool_input}"

    return f"답변:\n{answer}"


# --------------------------------------------------------------------
# 5. Streamlit 메인
# --------------------------------------------------------------------
def main():
    st.set_page_config(page_title="길고양이를 사랑한다면", layout="wide", page_icon="😺")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image('/workspaces/Stray_cat-Chat_Bot/data/cat.jpg', width=800)  # ← 들여쓰기

    st.markdown('---') 
    st.title("안녕하세요, 츄르 주시겠어요?:\n"
             "감사합니다.")

    with st.sidebar:
        openai_api = st.text_input("OPENAI API 키", type="password")
        tavily_api = st.text_input("TAVILY API 키", type="password")
        pdf_docs = st.file_uploader("PDF 파일 업로드", accept_multiple_files=True)

    if openai_api and tavily_api:
        os.environ['OPENAI_API_KEY'] = openai_api
        os.environ['TAVILY_API_KEY'] = tavily_api

        tools = [search_web()]
        if pdf_docs:
            tools.append(load_pdf_files(pdf_docs))

        agent_executor = build_agent(tools)

        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        user_input = st.chat_input("저에 대해 물어보세요")

        if user_input:
            response = ask_agent(agent_executor, user_input)
            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.session_state["messages"].append({"role": "assistant", "content": response})

        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

    else:
        st.warning("API 키 입력하라옹.")


if __name__ == "__main__":
    main()
