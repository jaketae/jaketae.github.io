import re
import sys


def read(file_path):
    with open(file_path) as file:
        text_list = []
        count = 0
        for line in file:
            if line == "---\n" and count < 2:
                count += 1
                continue
            line = line.rstrip("\n")
            if line and line[:4].strip() and count > 1:
                text_list.append(line.lower())
    return " ".join(text_list)


def clean(text):
    regexps = (
        r"```.*?```",
        r"\$\$.*?\$\$",
        r"`.*?`",
        r"\$.*?\$",
        r"\(.*?\)",
        r"[\[\]]",
        r"<.*?\>",
    )
    for regexp in regexps:
        text = re.sub(regexp, "", text)
    return re.sub(r"[\W]", " ", text)


def parse(file_path):
    text = read(file_path)
    return " ".join(clean(text).split())


def write(file_path):
    contents = parse(file_path)
    file_name = file_path.split(".")[0] + ".txt"
    with open(file_name, "w+") as file:
        file.write(contents)


if __name__ == "__main__":
    print("Start!")
    file_path = sys.argv[1]
    write(file_path)
    print("Complete!")
