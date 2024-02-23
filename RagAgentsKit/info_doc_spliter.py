# -*- coding: utf-8 -*-

# 将 *.info 文件按章节进行分割

import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures._base import Future

INFO_NODE_PATTERN = r"^File: .*Node: .*"


# define struct section, which contains the name of the section, the start line number and the end line number
class Section:
    def __init__(self, name, start_line, end_line):
        self.name = name
        self.start_line = start_line
        self.end_line = end_line
        self.file_name = ""


# 定义一个函数，用于显示信息节
def get_info_sections(info_file):
    # 创建一个列表，用于存储节
    section_list = []

    # 打印节名
    logging.warning(f" ===== {info_file} =========== ")

    # 打开文件
    with open(info_file) as f:
        # 读取文件内容
        file_contents = f.read()
        # 将文件内容按行分割
        info_file_lines = file_contents.splitlines()
        max_len = len(info_file_lines)

        # 遍历每一行
        for line_num, line in enumerate(info_file_lines, start=0):
            # 如果匹配到节点模式
            if re.match(INFO_NODE_PATTERN, line):
                # 将后两行和行号存入元组
                section = Section(
                    name=info_file_lines[line_num + 2],
                    start_line=line_num + 1,
                    end_line=max_len - 1,
                )
                # 将元组存入列表
                section_list.append(section)

    for i, section in enumerate(section_list, start=0):
        # Avoid access list out of line
        if i < len(section_list) - 1:
            section.end_line = section_list[i + 1].start_line - 3

    # 返回列表
    logging.warning(f" ========== Info file section ========== ")
    for sec in section_list:
        logging.warning(f"SECTION NAME: {sec.name}")

    return (info_file_lines, section_list)


# 定义一个函数，用于将info_file_lines文件中的指定行写入到out_file_path文件中
# 参数：info_file_lines：文件行列表；start_line：起始行；end_line：结束行；out_file_path：输出文件路径
def write_lines_to_file(p):
    (info_file_lines, section, out_file_name) = p

    end_line = len(info_file_lines)
    # 打开输出文件
    with open(out_file_name, "w") as out_file:
        # 遍历文件行列表
        new_doc = ""
        i = section.start_line
        while i <= section.end_line and i < end_line:
            new_doc += info_file_lines[i] + "\n"
            i += 1
        out_file.write(new_doc)

    # 返回输出文件路径
    return out_file_name


def write_done(res: Future):
    str = "Done: " + res.result()
    logging.warning(str)


# 定义一个函数，将部分写入文件
def extract_section_to_file(info_file_lines, section_list, base_dir, thread_num=16):
    # max_workers表示工人数量,也就是线程池里面的线程数量
    thread_pool = ThreadPoolExecutor(max_workers=thread_num)
    thread_task_list = []

    # 遍历section_list中的每一个section
    for i, section in enumerate(section_list, start=0):
        # 获取section的名称
        section_name = section.name.strip("\n")
        section_name = section_name.replace("/", "_")
        section_name = section_name.replace(":", "_")
        section_name = section_name.replace(" ", "_")
        # 拼接section的文件路径
        section_file_path = base_dir + section_name + ".txt"
        section.file_name = section_name + ".txt"
        thread_task_list.append((info_file_lines, section, section_file_path))

    for task in thread_task_list:
        futrue = thread_pool.submit(write_lines_to_file, task)  # type: ignore
        futrue.add_done_callback(write_done)

    thread_pool.shutdown()


# Save the list of extracted chapters to a file
def save_chapters(sec_dir="../data/mysql-8.0.info_section_dir/", section_list=[]):
    name_all_str = ""
    file_all_str = ""
    for sec in section_list:
        name_all_str += sec.name + "\n"
        file_all_str += sec.file_name + "\n"

    os.makedirs(sec_dir + "chapters", exist_ok=True)
    with open(sec_dir + "chapters/" + "chapters.txt", "w") as f:
        f.write(name_all_str)
    with open(sec_dir + "chapters/" + "chapters_file_list.txt", "w") as f:
        f.write(file_all_str)


# NOTICE：该脚本将 *.info 文件分割为不同的section文件
# TEST: python3 info_doc_spliter.py ../data/mysql-8.0.info
# TEST: python3 info_doc_spliter.py ../test_data/test_mysql.info
#
def split():
    # 解析用户在命令行指定的pdf文件名
    if len(sys.argv) < 2:
        logging.warning("请指定 info 文件名")
        sys.exit(1)
    info_file_path = sys.argv[1]

    # 创建tmp目录
    tmp_sec_dir = info_file_path + "_section_dir/"
    if not os.path.exists(tmp_sec_dir):
        os.makedirs(tmp_sec_dir, exist_ok=True)

    (info_file_lines, section_list) = get_info_sections(info_file_path)
    # XXX Notice: 这里使用并发，不会更快，单线程 0.5秒 左右处理完
    extract_section_to_file(info_file_lines, section_list, tmp_sec_dir, thread_num=1)

    # XXX 存放标题到文件中
    save_chapters(tmp_sec_dir, section_list)

    logging.warning("\n ===================== \n")
    logging.warning(f"NOTICE: Section file is in {tmp_sec_dir}")
    logging.warning("\n ===================== \n")

    return tmp_sec_dir


if __name__ == "__main__":
    sec_dir = split()
