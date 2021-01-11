import re
import sys

from nbformat import NO_CONVERT, read

from tagger.model import load_model


def rmd(nb):
    with open(nb, "r") as file:
        filedata = file.read()
    filedata = re.sub('src="', 'src="/assets/images/', filedata)
    with open(nb, "w") as file:
        file.write(filedata)


def ipynb(nb):
    title = nb.split(".")[0]
    with open(f"{title}.ipynb") as f:
        notebook = read(f, NO_CONVERT)
    text = get_text(notebook)
    tags = predict_tags(text)
    yaml = build_yaml(tags)
    with open(nb) as file:
        filedata = file.read()
    filedata = re.sub(r"!\[svg\]\(", '<img src="/assets/images/', filedata)
    filedata = re.sub(".svg\)", '.svg">', filedata)
    filedata = re.sub(r"!\[png\]\(", '<img src="/assets/images/', filedata)
    filedata = re.sub(".png\)", '.png">', filedata)
    filedata = yaml + filedata
    with open(nb, "w") as file:
        file.write(filedata)


def get_text(notebook):
    markdown_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    ]
    text = " ".join(cell["source"] for cell in markdown_cells)
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    text = re.sub(r"\$.*?\$", "math variable", text)
    text = re.sub(r"\]\(.*?\)", r"]", text)
    text = re.sub(r"(#)+ \w*", "", text)
    text = text.replace("[", "").replace("]", "")
    text = re.sub(r"`.*?`", "code variable", text)
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text


def predict_tags(text):
    model = load_model()
    tags = model.predict(text)
    return tags


def build_yaml(tags):
    tag_header = ""
    for tag in tags:
        tag_header += f"  - {tag}\n"
    return (
        f"---\ntitle: TITLE\nmathjax: true\ntoc: true\n"
        f"categories:\n  - category\ntags:\n{tag_header}---\n\n"
    )


if __name__ == "__main__":
    opt = sys.argv[1]
    nb = sys.argv[2]
    if opt == "-r":
        rmd(nb)
    else:
        ipynb(nb)
