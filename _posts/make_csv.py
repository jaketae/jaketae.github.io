import argparse
import os
import re
from collections import defaultdict

import pandas as pd


def parse(file_path, lower):
    tag_list = []
    body_list = []
    in_body = False
    in_category = False
    with open(file_path) as file:
        for i, line in enumerate(file):
            if i == 0 or "#" in line or not line[0:5]:
                continue
            line = line.rstrip("\n")
            if lower:
                line = line.lower()
            if line == "---":
                in_body = True
            if "tags" in line:
                in_category = True
            if in_body:
                if (
                    " http" in line
                    or "sha256" in line
                    or "=====" in line
                    or "-----" in line
                    or "_____" in line
                    or "█████" in line
                ):
                    continue
                body_list.append(line)
            elif in_category:
                tag = line.replace("-", "").strip()
                if tag:
                    tag_list.append(tag)
    return set(tag_list[1:]), body_list[1:]


def clean(text, remove_punc):
    regexps = (
        r"```.*?```",
        r"\$\$.*?\$\$",
        r"`.*?`",
        r"\$.*?\$",
        r"\{.*?\}",
        r"\(.*?\)",
        r"\[.*?\]",
        r"[\[\]]",
        r"\<.*?\>.*?</.*?\>",
        r"\<.*?\>",
        r"\*",
        r">",
        '"',
    )
    for regexp in regexps:
        text = re.sub(regexp, "", text)
    text = re.sub("-", " ", text)
    if remove_punc:
        text = re.sub(r"[\W]", " ", text)
    return re.sub("\s+", " ", text).strip()


def organize_one(file_path, lower, remove_punc):
    tags, body_list = parse(file_path, lower)
    body = clean(" ".join(body_list), remove_punc)
    return {"body": body, "tags": tags}


def organize_all(lower, remove_punc):
    posts = []
    all_tags = set()
    tags_count = defaultdict(int)
    for file in os.listdir(os.getcwd()):
        if os.path.isdir(file):
            continue
        title, extension = os.path.splitext(file)
        if extension == ".md":
            post = organize_one(file, lower, remove_punc)
            post["title"] = title
            posts.append(post)
            tags = post["tags"]
            all_tags.update(tags)
            for tag in tags:
                tags_count[tag] += 1
    for tag, count in tags_count.items():
        if count == 1:
            all_tags.remove(tag)
    return posts, tuple(all_tags)


def build_df(posts, all_tags, save_path):
    final = []
    for post in posts:
        tags = post["tags"]
        count = 0
        for tag in all_tags:
            has_tag = int(tag in tags)
            post[tag] = has_tag
            count += has_tag
        if count:
            post.pop("tags")
            final.append(post)
        else:
            continue
    df = pd.DataFrame(final).set_index("title")
    df.to_csv(save_path)


def main(args):
    posts, all_tags = organize_all(args.lower, args.remove_punc)
    build_df(posts, all_tags, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lower", type=bool, default=True, help="convert to lowercase"
    )
    parser.add_argument(
        "--remove_punc", type=bool, default=False, help="remove punctuation"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="../../bert-blog-tagger/data.csv",
        help="final csv file save path",
    )
    args = parser.parse_args()
    main(args)
