from pathlib import Path

from src.helper import render_markdown_file

render_markdown_file(Path.cwd()/"docs/titanic_blog.md")
