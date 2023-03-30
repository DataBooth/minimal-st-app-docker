import platform as plf
from pathlib import Path

import httpx
from IPython.display import Markdown, display
from streamlit import cache_data, markdown


def read_markdown_file(markdown_file, png_url):
    try:
        text = Path(markdown_file).read_text()
        if png_url is not None:
            text = text.replace("eda_titanic_data_files", png_url)
        return text
    except Exception as IOError:
        raise(IOError)



@cache_data
def st_read_markdown_file(markdown_file, png_url):
    return read_markdown_file(markdown_file, png_url)


def render_markdown_file(markdown_file, png_url=None, output="streamlit"):
    if output == "jupyter":
        md_text = read_markdown_file(markdown_file)
        display(Markdown(md_text))
    else:
        md_text = st_read_markdown_file(markdown_file, png_url)
        markdown(md_text, unsafe_allow_html=True)
    return


def get_file_header(file_name, n_char=100, n_row=1):
    if file_name[:len("http")] == "http":
        with httpx.Client() as client:
            r = client.get(file_name)
        return r.text[:min(n_char, len(r.text))]
    else:
        with open(file_name, "r") as f:
            return f.readlines()[:n_row]


def get_system_info():
    plat = plf.uname()
    return [f"System: {plat.system}", f"Node Name: {plat.node}", 
            f"Release: {plat.release}", f"Version: {plat.version}", 
            f"Machine: {plat.machine}", f"Processor: {plat.processor}", 
            f"Architecture: {plf.architecture()}", f"Python version: {plf.python_version()}"]
