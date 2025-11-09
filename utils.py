"""
このファイルは、固定の文字列や数値などのデータを変数として一括管理するファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

# ✅ create_retrieval_chain と create_stuff_documents_chain の正しいimport
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter

import streamlit as st
import os
import constants as ct


############################################################
# ベクトルデータベース作成関数
############################################################
def create_vector_db(file_path, persist_directory):
    """
    ドキュメントをロードし、ベクトルデータベースを作成する。
    """
    ext = os.path.splitext(file_path)[1].lower()

    # --- ファイルタイプごとのローダー選択 ---
    if ext == ".pdf":
        loader = PyMuPDFLoader(file_path)
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(file_path)
    elif ext in [".txt", ".md"]:
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"対応していないファイル形式です: {ext}")

    # --- ドキュメントを分割 ---
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE,
        chunk_overlap=ct.CHUNK_OVERLAP
    )
    split_docs = text_splitter.split_documents(docs)

    # --- 埋め込みモデルとDB作成 ---
    embeddings = OpenAIEmbeddings(model=ct.EMBEDDING_MODEL)
    vectordb = Chroma.from_documents(
        split_docs,
        embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb


############################################################
# LLM応答生成関数（ページ番号付き・重複削除済）
############################################################
def get_llm_response(chat_message):
    """
    LLMからの回答を取得し、参照元情報（PDFページ番号付き）を整形して返す。
    """
    # --- LLM初期化 ---
    llm = ChatOpenAI(model_name=ct.MODEL, temperature=ct.TEMPERATURE)

    # --- 質問生成プロンプト ---
    question_generator_template = ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT
    question_generator_prompt = ChatPromptTemplate.from_messages([
        ("system", question_generator_template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # --- 回答生成プロンプト（モード別） ---
    if st.session_state.mode == ct.ANSWER_MODE_1:
        question_answer_template = ct.SYSTEM_PROMPT_DOC_SEARCH
    else:
        question_answer_template = ct.SYSTEM_PROMPT_INQUIRY

    question_answer_prompt = ChatPromptTemplate.from_messages([
        ("system", question_answer_template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # --- 履歴対応リトリーバー作成 ---
    history_aware_retriever = create_history_aware_retriever(
        llm, st.session_state.retriever, question_generator_prompt
    )

    # --- チェーン作成 ---
    question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)
    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # --- LLM呼び出し ---
    llm_response = chain.invoke({
        "input": chat_message,
        "chat_history": st.session_state.chat_history
    })

    # --- 参照情報整形（PDFページ番号付き） ---
    source_info_list = []
    if "context" in llm_response:
        for doc in llm_response["context"]:
            src = doc.metadata.get("source", "")
            page = doc.metadata.get("page", None)
            if src.endswith(".pdf") and page is not None:
                source_info_list.append(f"{os.path.basename(src)}（ページNo.{page}）")
            else:
                source_info_list.append(os.path.basename(src))
    elif "source_documents" in llm_response:
        for doc in llm_response["source_documents"]:
            src = doc.metadata.get("source", "")
            page = doc.metadata.get("page", None)
            if src.endswith(".pdf") and page is not None:
                source_info_list.append(f"{os.path.basename(src)}（ページNo.{page}）")
            else:
                source_info_list.append(os.path.basename(src))

    # --- 会話履歴更新 ---
    st.session_state.chat_history.extend([
        HumanMessage(content=chat_message),
        llm_response["answer"]
    ])

    # --- 整形済みの参照情報を格納 ---
    llm_response["source_info"] = list(set(source_info_list))
    return llm_response


############################################################
# セッション初期化
############################################################
def initialize_session():
    """
    Streamlitセッションの初期化
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "mode" not in st.session_state:
        st.session_state.mode = ct.ANSWER_MODE_1
    if "retriever" not in st.session_state:
        st.session_state.retriever = None