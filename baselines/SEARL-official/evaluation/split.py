# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import os
import argparse

def split_log_into_chunks(log_file_path, output_dir=None, start_pattern='actor/kl_loss:'):
    'Split an input log into output files at start_pattern markers.'

    if output_dir is None:
        output_dir = os.path.splitext(log_file_path)[0]

    chunks_dir = os.path.join(output_dir, "chunks")

    if not os.path.exists(chunks_dir):
        os.makedirs(chunks_dir)
        print(f"Created directory: {chunks_dir}")

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
                    print(f"Writing a new chunk to: {chunk_filename}")

                if current_chunk_file:
                    current_chunk_file.write(line)

        if current_chunk_file:
            current_chunk_file.close()

        print(f"\nCompleted; split into {chunk_count} files.")
        print(f"Output directory: {chunks_dir}")

    except FileNotFoundError:
        print(f"ERROR: File \'{log_file_path}\' not found.")
    except Exception as e:
        print(f"Error: {e}")

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
