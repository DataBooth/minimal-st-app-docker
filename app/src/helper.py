import platform as plf
from pathlib import Path

import httpx
from IPython.display import Markdown, display
from streamlit import cache, markdown


def read_markdown_file(markdown_file):
    try:
        return Path(markdown_file).read_text()
    except Exception as IOError:
        sys_list.append(f"Error with markdown file: {markdown_file}")
        raise(IOError)


@cache
def st_read_markdown_file(markdown_file):
    return read_markdown_file(markdown_file)


def render_markdown_file(markdown_file, output="streamlit"):
    if output == "jupyter":
        md_text = read_markdown_file(markdown_file)
        display(Markdown(md_text))
    else:
        md_text = st_read_markdown_file(markdown_file)
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
