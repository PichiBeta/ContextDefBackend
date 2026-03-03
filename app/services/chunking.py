import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
enc = tiktoken.get_encoding("cl100k_base")


def token_len(text):
    return len(enc.encode(text))

splitter = text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=20,
        length_function=token_len)