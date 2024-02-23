# -*- coding: utf-8 -*-

import logging
import os
import re
import zipfile

import nltk
import wget
from nltk.tokenize import sent_tokenize
from numpy import short


# 定义nltk_prepare函数
def nltk_prepare():
    # 如果nltk_data文件夹不存在，则创建
    if not os.path.exists("/home/zsdfbb/ssd_2t/ai_model/nltk_data/tokenizers"):
        os.makedirs("/home/zsdfbb/ssd_2t/ai_model/nltk_data/tokenizers")

    # 如果punkt文件夹不存在，则下载并解压
    if not os.path.exists("/home/zsdfbb/ssd_2t/ai_model/nltk_data/tokenizers/punkt"):
        file_name = wget.download(
            "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip",
            out="/home/zsdfbb/ssd_2t/ai_model/nltk_data",
        )
        with zipfile.ZipFile(
            "/home/zsdfbb/ssd_2t/ai_model/nltk_data/punkt.zip", "r"
        ) as zin:
            zin.extractall("/home/zsdfbb/ssd_2t/ai_model/nltk_data/tokenizers/")

    # 将nltk_data文件夹路径添加到nltk.data.path中
    nltk.data.path.append(os.getcwd() + "//home/zsdfbb/ssd_2t/ai_model/nltk_data")

    logging.debug(nltk.data.path)


def split_sentence(sentence, sequence_length):
    substrings = []
    for i in range(0, len(sentence), sequence_length):
        substrings.append(sentence[i : i + sequence_length])
    return substrings


# 定义一个函数，用于将文档路径分割成文档内容
def nltk_split(doc_path, sequence_length=512):
    # 定义一个变量，用于存储文档内容
    doc_content = ""
    # 定义一个列表，用于存储句子
    sentences = []
    # 打开文档路径，以utf-8编码方式读取文档内容
    with open(doc_path, "r", encoding="utf-8") as f:
        # 读取文档中的所有行
        lines = f.readlines()
        # 将每一行中的换行符替换为空格
        lines = list(map(lambda l: l.replace("\n", " "), lines))
        # 将每一行中的多个空格替换为一个空格
        lines = list(map(lambda l: re.sub(r"\s\s+", " ", l), lines))
        # 打印文档内容
        # logging.debug(lines)
        doc_content = doc_content.join(lines)

        # 使用NLTK的sent_tokenize函数，将文本拆分成句子
        sentences = sent_tokenize(doc_content)
        # 打印拆分后的句子

    short_senteces = []
    for s in sentences:
        if len(s) < sequence_length:
            short_senteces.append(s)
        else:
            short_senteces.extend(split_sentence(s, sequence_length))

    logging.debug("\n")
    logging.debug("================ 文章拆分前 ================")
    logging.debug(doc_content)

    logging.debug("\n")
    logging.debug("================ 文章拆分后 ================")
    max_len = 0
    max_s = ""
    for s in short_senteces:
        logging.debug("SENTENCE HEAD: " + s)
        if max_len < len(s):
            max_len = len(s)
            max_s = s

    logging.debug("\n")
    logging.debug("================ 拆分统计信息 ================")
    logging.debug(f"longest_string_length: {max_len}")
    logging.debug(f"longest_string: {max_s}")
    logging.debug("short_senteces list length: {}".format(len(short_senteces)))

    return short_senteces


# 定义一个函数，用于将句子合并成段落，参数sentences为句子列表，sequence_length为每个段落的最大长度，truncate为是否截断
def merge_sentences(sentences, sequence_length=512, truncate=True):
    # 定义一个段落列表
    paragraphs = []
    # 定义一个临时字符串
    tmp_s = ""
    # 定义一个临时字符串的长度
    tmp_s_len = 0
    # 遍历句子列表
    for s in sentences:
        # 如果临时字符串的长度加上句子长度小于段落最大长度，则将句子拼接到临时字符串后面
        if len(s) + tmp_s_len < sequence_length:
            tmp_s = tmp_s + " " + s
            tmp_s_len = tmp_s_len + len(s)
        # 否则，将临时字符串添加到段落列表中，并将句子长度设置为句子长度
        else:
            paragraphs.append(tmp_s)
            tmp_s_len = len(s)
            tmp_s = s
    # 将最后一个临时字符串添加到段落列表中
    paragraphs.append(tmp_s)

    # 返回段落列表
    return paragraphs


def text_split(text, chunk_size=8000):
    # 使用正则表达式匹配单词
    words = re.findall(r"\b\w+\b", text)

    # 计算需要分割多少个单词
    num_chunks = len(words) // chunk_size + (1 if len(words) % chunk_size > 0 else 0)

    # 分割文本
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size if (i + 1) * chunk_size < len(words) else len(words)
        # 截取对应长度的单词列表，并将其连接成字符串
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)

    return chunks


# ========================================================


def try_split(doc_path):
    nltk_prepare()
    sentences = nltk_split(doc_path)
    paragraphs = merge_sentences(sentences)
    for p in paragraphs:
        print("\n ========= paragraph: ========= \n")
        print(p)


if __name__ == "__main__":
    logging.root.setLevel(logging.DEBUG)
    try_split("../test_data/test_mysql_longtext.txt")
