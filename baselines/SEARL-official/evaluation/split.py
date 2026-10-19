# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import os
import argparse

def split_log_into_chunks(log_file_path, output_dir=None, start_pattern='actor/kl_loss:'):
    '\n    \u6839\u636e\u8d77\u59cb\u6a21\u5f0f\u5c06\u4e00\u4e2a\u5927\u65e5\u5fd7\u6587\u4ef6\u5206\u5272\u6210\u591a\u4e2a\u5c0f\u6587\u4ef6\u3002\n\n    :param log_file_path: \u8f93\u5165\u7684\u65e5\u5fd7\u6587\u4ef6\u8def\u5f84\u3002\n    :param start_pattern: \u6807\u5fd7\u7740\u4e00\u4e2a\u65b0 chunk \u5f00\u59cb\u7684\u5b57\u7b26\u4e32\u3002\n    :param output_dir: \u5b58\u653e\u5207\u5206\u540e\u6587\u4ef6\u7684\u76ee\u5f55\u3002\n    '

    if output_dir is None:
        output_dir = os.path.splitext(log_file_path)[0]

    chunks_dir = os.path.join(output_dir, "chunks")

    if not os.path.exists(chunks_dir):
        os.makedirs(chunks_dir)
        print(f"\u521b\u5efa\u76ee\u5f55: {chunks_dir}")

    chunk_count = 0
    current_chunk_file = None

    try:
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if start_pattern in line.strip():

                    if current_chunk_file:
                        current_chunk_file.close()

                    chunk_count += 1

                    log_basename = os.path.splitext(os.path.basename(log_file_path))[0]
                    chunk_filename = os.path.join(chunks_dir, f'{log_basename}_chunk_{chunk_count}.log')
                    current_chunk_file = open(chunk_filename, 'w', encoding='utf-8')
                    print(f"\u68c0\u6d4b\u5230\u65b0 chunk\uff0c\u5f00\u59cb\u5199\u5165\u6587\u4ef6: {chunk_filename}")

                if current_chunk_file:
                    current_chunk_file.write(line)

        if current_chunk_file:
            current_chunk_file.close()

        print(f"\n\u5904\u7406\u5b8c\u6210\uff01\u603b\u5171\u5206\u5272\u51fa {chunk_count} \u4e2a\u6587\u4ef6\u3002")
        print(f"\u8f93\u51fa\u76ee\u5f55: {chunks_dir}")

    except FileNotFoundError:
        print(f"\u9519\u8bef: \u6587\u4ef6 '{log_file_path}' \u672a\u627e\u5230\u3002")
    except Exception as e:
        print(f"\u53d1\u751f\u9519\u8bef: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split log file into chunks based on a pattern.")
    parser.add_argument("--log_path", type=str,
                        default="logs/train.log",
                        help="Path to the input log file.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save the chunks. Defaults to the log file's directory.")
    parser.add_argument("--pattern", type=str, default="actor/kl_loss:",
                        help="The string pattern that signals the start of a new chunk.")

    args = parser.parse_args()

    split_log_into_chunks(args.log_path, args.output_dir, args.pattern)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
