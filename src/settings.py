import os
import toml

with open("pyproject.toml", "r") as file:
    _pyproject_content = toml.load(file)


class Settings:
    """Represents the settings for the application.

    Some of them are loaded from environment variables, some hardcoded, and others loaded from the pyproject.toml .
    """

    # Loaded from environment variables

    port: int = int(os.environ["STREAMLIT_SERVER_PORT"])
    elastic_url = os.environ["ELASTIC_URL"]
    openai_api_key = os.environ["OPENAI_API_KEY"]

    # Hardcoded

    temp_dir = "tmp/"
    pdf_max_pages = 50
    pdf_chunk_size = 1000
    pdf_chunk_overlap = 20
    llm_prompt_template = """
    Ignore all previous instructions.
    1. You are to provide clear, concise, and direct responses.
    2. Be blunt and straightforward; don't sugarcoat.
    3. Eliminate unnecessary reminders, apologies, self-references, and any pre-programmed niceties. 
    This includes phrases with words like 'sorry', 'apologies', and 'regret'.
    4. No moral lectures.
    5. Discuss safety only if it's crucial and non-obvious.
    6. Maintain a casual tone in your communication.
    7. Never mention that you're an AI.
    8. Be transparent; if you're unsure about an answer or a question is beyond your abilities or knowledge, admit it.
    9. If information is beyond your scope or knowledge cutoff date, simply state "I don't know".
    10. Don't use disclaimers about expertise or professionalism.
    11. For any unclear or ambiguous queries, ask follow-up questions to understand the user's intent better.
    12. When explaining concepts, use real-world examples and analogies, where appropriate.
    13. Ensure responses are unique and without repetition.
    14. Never suggest seeking information elsewhere.
    15. For complex requests, take a deep breath and work on the problem step-by-step.
    16. I will give you 20 000 bonus points for every great answer.
    17. Before crafting a reply, think about your observations in 3 sentences with clarifying strategy we should choose.
    Take it into consideration but don't print.
    It is very important that you get this right. Multiple lives are at stake.
    Answer the [QUESTION] based on the [CONTEXT] below.

    [CONTEXT]
    {context}

    [QUESTION]
    {question}
    """
    sample_questions = ["What this document is about?", "How many pages are in the document?"]

    non_printable_chars_regex = r"[^\x20-\x7E\xA0-\xFF\u0100-\u017F\u0370-\u03FF\u0400-\u04FF\u0590-\u05FF\u0600-\u06FF\
\u0750-\u077F\u0900-\u097F\u0E00-\u0E7F\u4E00-\u9FFF\u3000-\u303F\uFF00-\uFFEF]+"
    # This combined regular expression includes:
    #
    # Basic Latin characters (U+0020 to U+007E)
    # Latin-1 Supplement characters (U+00A0 to U+00FF)
    # Latin Extended-A characters (U+0100 to U+017F)
    # Greek characters (U+0370 to U+03FF)
    # Cyrillic characters (U+0400 to U+04FF)
    # Hebrew characters (U+0590 to U+05FF)
    # Arabic characters (U+0600 to U+06FF, U+0750 to U+077F)
    # Devanagari characters (U+0900 to U+097F)
    # Thai characters (U+0E00 to U+0E7F)
    # CJK Unified Ideographs (U+4E00 to U+9FFF)
    # CJK Symbols and Punctuation (U+3000 to U+303F)
    # Halfwidth and Fullwidth Forms (U+FF00 to U+FFEF)

    # From pyproject.toml

    app_name = _pyproject_content["tool"]["poetry"]["name"]
    app_description = _pyproject_content["tool"]["poetry"]["description"]
    app_version = _pyproject_content["tool"]["poetry"]["version"]
