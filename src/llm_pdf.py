import hashlib
import os
import json

from elasticsearch import Elasticsearch
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from streamlit.runtime import get_instance
import threading

from src.logger import logger
from src.settings import Settings

LLM_MODEL = "gpt-3.5-turbo"

if not os.path.exists(Settings.temp_dir):
    os.makedirs(Settings.temp_dir)


def llm_pdf_app():
    if "es" not in st.session_state:
        st.session_state["es"] = Elasticsearch(Settings.elastic_url)

    logger.info("UI loop")

    icon = "💬"
    st.set_page_config(page_title=Settings.app_description, page_icon=icon, layout="centered")
    st.title(icon + " " + Settings.app_description, anchor="home")

    if "pdf_hash" not in st.session_state or not st.session_state["pdf_hash"]:
        st.subheader("👈 Please upload the pdf file first.")

    # Display chat messages from the conversation history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "sample_question" not in st.session_state:
        st.session_state["sample_question"] = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], dict):
                st.markdown(message["content"]["markdown"])

                def set_sample_question(question):
                    st.session_state.sample_question = question

                for question in message["content"]["sample_questions"]:
                    st.button(question, on_click=set_sample_question, args=[question])
            else:
                st.markdown(message["content"])

    # Show the control for user input
    if "input_disabled" not in st.session_state:
        st.session_state["input_disabled"] = True

    user_prompt = st.chat_input("Your question", disabled=st.session_state.input_disabled)

    # The sample question selected by the button overrides the user's input
    if st.session_state.sample_question:
        user_prompt = st.session_state.sample_question
        st.session_state.sample_question = None

    if "first_last_doc" not in st.session_state:
        st.session_state["first_last_doc"] = None

    if "document_facts" not in st.session_state:
        st.session_state["document_facts"] = None

    if user_prompt:
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            answer = st.write_stream(
                _stream_rag(
                    user_prompt,
                    st.session_state.messages,
                    st.session_state.first_last_doc,
                    st.session_state.document_facts,
                )
            )
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        st.session_state.messages.append({"role": "assistant", "content": answer})

    # Sidebar to load pdf
    st.sidebar.title("Your PDF")

    _remove_chunks_from_elastic_on_exit()
    uploaded_file = st.sidebar.file_uploader(
        "",
        accept_multiple_files=False,
    )
    st.sidebar.caption("""
                       We don't store your content, but process it with OpenAI services. Uploaded PDF, questions,
                       and answers will be deleted from our machine when the page is closed.
                       """)

    st.sidebar.markdown("""
                        # About the app

                        This is a Retrieval-Augmented Generation (RAG) application that slices the uploaded PDF,
                        searches for parts related to the question, and summarises them using the OpenAI API.

                        You can find the complete source code on [GitHub](https://github.com/IvanRublev/llm_pdf).

                        Ivan Rublev © 2024.
                        """)

    if uploaded_file:
        pdf_path, pdf_size, pdf_hash = _write_tmp_file(uploaded_file)

        if "pdf_hash" not in st.session_state or st.session_state["pdf_hash"] != pdf_hash:
            st.session_state.pdf_hash = pdf_hash
            st.session_state.document_facts = None
            st.session_state.first_last_doc = None
            st.session_state.messages = []
            st.cache_data.clear()
            st.session_state.input_disabled = True
            st.rerun()

        logger.info(f"A file was of size: {pdf_size} was written to: {pdf_path}")

        chunks = _read_text_chunks(pdf_path, pdf_hash, Settings.pdf_chunk_size, Settings.pdf_chunk_overlap)
        os.remove(pdf_path)
        chunks_count = len(chunks)
        logger.info(f"The file was splitted into: {chunks_count} chunks")

        if chunks_count == 0:
            st.error("PDF file has no parsable text.", icon="🚨")
            return

        pages_count = int(chunks[-1].metadata["page"]) + 1

        max_pages = Settings.pdf_max_pages
        if pages_count > max_pages:
            st.error(f"We don't support PDF files longer than {max_pages} pages.", icon="🚨")
            return

        st.session_state.document_facts = f"The document has {pages_count} pages."

        docs = _docs_from_chunks(chunks, pdf_hash)

        if docs:
            st.session_state.first_last_doc, docs = _split_docs(docs, pdf_hash)

            def on_finish_fn():
                # message will be displayed on the following rerun
                st.session_state.messages.append(
                    {"role": "assistant", "content": "Thank you for the uploading! How I can help you?"}
                )
                translated_questions = _translate_sample_questions(
                    st.session_state.first_last_doc, Settings.sample_questions
                )
                logger.info(f"Translated sample questions: {translated_questions}")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": {
                            "markdown": "Sample questions I can answer:",
                            "sample_questions": translated_questions,
                        },
                    }
                )

            _upload_docs_to_elastic(docs, pdf_hash, on_finish_fn)

        if st.session_state.input_disabled:
            st.session_state.input_disabled = False
            st.rerun()
    else:
        # no file uploaded, cleanup associated attributes
        if not st.session_state.input_disabled:
            st.session_state.input_disabled = True
            st.session_state.pdf_hash = None
            st.session_state.document_facts = None
            st.session_state.first_last_doc = None
            st.session_state.messages = []
            st.cache_data.clear()
            st.rerun()


def _write_tmp_file(uploaded_file):
    pdf_path = None
    pdf_bytes_hash = None
    pdf_size = None

    if uploaded_file:
        pdf_size = uploaded_file.size
        pdf_bytes = uploaded_file.getvalue() if uploaded_file else None
        pdf_bytes_hash = hashlib.sha256(pdf_bytes).hexdigest()
        pdf_path = _absolute_path(Settings.temp_dir, f"file_{pdf_bytes_hash}.pdf")
        with open(pdf_path, "wb") as file:
            file.write(pdf_bytes)

    return pdf_path, pdf_size, pdf_bytes_hash


@st.cache_data
def _read_text_chunks(pdf_path, cache_data_pdf_hash, chunk_size, chunk_overlap):
    loader = PyPDFLoader(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return loader.load_and_split(splitter)


@st.cache_data
def _docs_from_chunks(_chunks, cache_data_pdf_hash):
    chunks = _chunks

    # Keep only printable characters and add page number as first element
    clean_chunks = []

    for chunk in chunks:
        text = re.sub(Settings.non_printable_chars_regex, " ", chunk.page_content)
        page = chunk.metadata["page"]
        text = f"Page {page + 1}: " + text
        clean_chunks.append(text)

    # We have one chunk = one page of original PDF,
    # join the chunks to have them together as close to pdf_chunk_size as possible
    enlarged_chunks = []
    accumulated_len = 0
    accumulated_chunk = ""
    for chunk in clean_chunks:
        chunk_len = len(chunk)
        if accumulated_len + chunk_len > Settings.pdf_chunk_size:
            enlarged_chunks.append(accumulated_chunk)
            accumulated_chunk = ""
            accumulated_len = 0

        accumulated_chunk = accumulated_chunk + "\n" + chunk
        accumulated_len += chunk_len

    if accumulated_len != 0:
        enlarged_chunks.append(accumulated_chunk)

    docs = []
    for chunk in enlarged_chunks:
        docs.append({"text": chunk})

    return docs


@st.cache_data
def _split_docs(_docs, cache_data_pdf_hash):
    # we split the docs, first and last one we keep in memory to always put in llm,
    # rest upload we to elstic for fruther retrival
    docs = _docs

    if docs[0] != docs[-1]:
        first_last_doc = [docs[0], docs[-1]]
        docs.pop(-1)
        docs.pop(0)
    else:
        first_last_doc = [docs[0]]
        docs.pop(0)

    logger.info(f"First and last docs: {first_last_doc}")

    return first_last_doc, docs


@st.cache_data(show_spinner=False)
def _upload_docs_to_elastic(_docs, cache_data_pdf_hash, _on_finish_fn):
    docs = _docs
    on_finish_fn = _on_finish_fn

    es = st.session_state.es
    logger.info(f"Information about elastic instance: {es.info()}")

    index_name = _index_name()
    logger.info(f"Uploading {len(docs)} docs to index {index_name}")

    # we delete existing index to overwrite docs
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    index_settings = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {"properties": {"text": {"type": "text"}}},
    }
    es.indices.create(index=index_name, body=index_settings)

    progress_text = "Upload documents"
    progress_bar = st.progress(0, text=progress_text)
    total_chunks = len(docs)

    for idx, doc in enumerate(docs):
        es.index(index=index_name, document=doc)
        progress_bar.progress(idx / total_chunks, text=progress_text)

    progress_bar.empty()
    on_finish_fn()


def _remove_chunks_from_elastic_on_exit():
    # from https://discuss.streamlit.io/t/detecting-user-exit-browser-tab-closed-session-end/62066
    thread = threading.Timer(interval=2, function=_remove_chunks_from_elastic_on_exit)

    # insert context to the current thread, needed for
    # getting session specific attributes like st.session_state

    add_script_run_ctx(thread)

    session_id = _session_id()

    runtime = get_instance()  # this is the main runtime, contains all the sessions

    if runtime.is_active_session(session_id=session_id):
        # Session is running
        thread.start()
    else:
        # Session is not running, Do what you want to do on user exit here
        logger.info("User closed the page. Removing chunks from elastic.")

        es = st.session_state.es
        index_name = _index_name()
        if es.indices.exists(index=index_name):
            es.indices.delete(index=index_name)

        return


@st.cache_data
def _translate_sample_questions(_first_last_docs, questions):
    first_last_docs = _first_last_docs
    template = PromptTemplate.from_template("""
    What is the language of the [CONTEXT]? Return the answer as one word, if you're not sure say English.
                                            
    [CONTEXT]
    {context}
    """)
    prompt = template.format(context=first_last_docs)
    model = ChatOpenAI(api_key=Settings.openai_api_key, model=LLM_MODEL)
    parser = StrOutputParser()
    chain = model | parser

    doc_language = chain.invoke(prompt)
    logger.info(f"Document language: {doc_language}")

    template = PromptTemplate.from_template("""
    Translate [CONTEXT] line by line into {language} language. 
    Return answer as a JSON list of strings. 
    For example, when translated lines are "a" and "b" format your output like ["a", "b"].

    [CONTEXT]
    {context}
    """)
    prompt = template.format(context=questions, language=doc_language)
    parser = StrOutputParser()
    chain = model | parser
    translated_questions = json.loads(chain.invoke(prompt))

    # roughly check output
    if len(translated_questions) != len(questions):
        translated_questions = questions

    return translated_questions


def _stream_rag(prompt, messages_history, first_last_doc, facts):
    def prompt_fn(prompt):
        docs = first_last_doc + _query_elastic(prompt)
        llm_prompt = _build_llm_prompt(prompt, messages_history, facts, docs)
        logger.info(f"Prompt for LLM: {llm_prompt}")
        return llm_prompt

    model = ChatOpenAI(api_key=Settings.openai_api_key, model=LLM_MODEL)
    parser = StrOutputParser()
    chain = prompt_fn | model | parser
    logger.info(f"User prompt: {prompt}")
    return chain.stream(prompt)


def _query_elastic(question):
    search_query = {
        "size": 5,
        "query": {"bool": {"must": {"multi_match": {"query": question, "fields": ["text"], "type": "best_fields"}}}},
    }

    es = st.session_state.es
    index_name = _index_name()
    response = es.search(index=index_name, body=search_query)

    chunks = []
    for hit in response["hits"]["hits"]:
        chunks.append(hit["_source"])

    return chunks


def _build_llm_prompt(user_prompt, messages_history, facts, docs):
    template = PromptTemplate.from_template(Settings.llm_prompt_template)

    content = []
    for doc in docs:
        content.append(doc["text"])

    content_text = "\n".join(content)

    history = []
    for message in messages_history:
        history.append(f"{message['role']}: {message['content']}")

    history_text = "\n".join(history)

    context = f"""
    # Facts about the document
    {facts}
    
    # Content
    {content_text}

    # Our previous conversation
    {history_text}
    """

    return template.format(context=context, question=user_prompt)


def _index_name():
    session_id = _session_id()
    return f"chunks_{session_id}"


def _session_id():
    # context is required to get session_id of the calling
    # thread (which would be the script thread)
    ctx = get_script_run_ctx()
    return ctx.session_id


def _absolute_path(directory_path, file_name):
    relative_path = os.path.join(directory_path, file_name)
    return os.path.abspath(relative_path)
