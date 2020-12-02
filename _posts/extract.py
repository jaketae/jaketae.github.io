import argparse
import os
import re


def read(file_path, lower):
    tag_list = []
    body_list = []
    in_body = False
    in_category = False
    with open(file_path) as file:
        for i, line in enumerate(file):
            if i == 0 or "#" in line:
                continue
            line = line.rstrip("\n")
            if lower:
                line = line.lower()
            if line == "---":
                in_body = True
            if "tags" in line:
                in_category = True
            if in_body:
                body_list.append(line)
            elif in_category:
                tag_list.append(line.replace("-", "").strip())
    return " ".join(tag_list[1:]), " ".join(body_list[1:])


def clean(text, remove_punc):
    regexps = (
        r"```.*?```",
        r"\$\$.*?\$\$",
        r"`.*?`",
        r"\$.*?\$",
        r"\(.*?\)",
        r"[\[\]]",
        r"\<div\>.*?</div\>",
        r"\<.*?\>",
        r"\*",
        r">",
    )
    for regexp in regexps:
        text = re.sub(regexp, "", text)
    text = re.sub("-", " ", text)
    if remove_punc:
        return re.sub(r"[\W]", " ", text)
    return text


def write(file_path, lower, remove_punc):
    tags, body_text = read(file_path, lower)
    body_text = " ".join(clean(body_text, remove_punc).split())
    source_file, _ = os.path.splitext(file_path)
    file_name = f"{source_file}.txt"
    body_save_dir = os.path.join("data", file_name)
    label_save_dir = os.path.join("labels", file_name)
    with open(body_save_dir, "w+") as file:
        file.write(body_text)
    with open(label_save_dir, "w+") as file:
        file.write(tags)


def main(args):
    for file in os.listdir(os.getcwd()):
        if os.path.isdir(file):
            continue
        source_file, extension = os.path.splitext(file)
        if extension == ".md":
            write(file, args.lower, args.remove_punc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lower", type=bool, default=True, help="convert to lowercase"
    )
    parser.add_argument(
        "--remove_punc", type=bool, default=False, help="remove punctuation"
    )
    args = parser.parse_args()
    main(args)
