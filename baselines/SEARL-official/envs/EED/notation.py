# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

#This file is used to pre-process input latex expressions
#You only need a "master_convert()"
from latex2sympy2_extended import *
from sympy import simplify

def brackets_balanced(s: str) -> bool:
    '\n    Check if the brackets in a LaTeX string are balanced\n    Args:\n        s(str): the input string\n    Return:\n        bool: True if the brackets are balanced, False otherwise\n    '
    stack = []
    bracket_pairs = {')': '(', ']': '[', '}': '{'}

    for char in s:
        if char in bracket_pairs.values():
            stack.append(char)
        elif char in bracket_pairs:
            if not stack or stack[-1] != bracket_pairs[char]:
                return False
            stack.pop()
    return len(stack) == 0

def remove_non_ascii(text):
    return text.encode("ascii", errors="ignore").decode()

import re
def extract_bracket_content(s:str,bracket_position:int) -> str:
    start_idx=bracket_position

    stack = []
    content = []
    escaped = False
    brace_start=start_idx+1
    brace_depth = 0
    for i in range(brace_start, len(s)):
        char = s[i]
        if escaped:
            content.append(char)
            escaped = False
            continue
        if char == '\\':
            escaped = True
            content.append(char)
            continue
        if char == '{':
            brace_depth += 1
            content.append(char)
        elif char == '}':
            if brace_depth == 0:
                return ''.join(content),i
            brace_depth -= 1
            content.append(char)
        else:
            content.append(char)

    return None,-1
def find_first_unescaped_brace(s: str) -> int:
    escaped = False
    for i, c in enumerate(s):
        if c == '\\' and not escaped:
            escaped = True
            continue
        if c == '{' and not escaped:
            return i
        escaped = False
    return -1

def extract_command(s: str, brace_pos: int) -> str | None:
    """extract the command name from a bracket"""
    i = brace_pos - 1
    parameter_mode=False
    while i >= 0:
        if not parameter_mode and s[i] in ('^','_'):
            return s[i]
        if not parameter_mode and not s[i] in (' ','\t',']','['):
            break
        if s[i]==']':
            parameter_mode=True
        if s[i]=='[' and parameter_mode:
            parameter_mode=False
        i -= 1

    # Start point
    if i < 0 or s[i] == '\\':
        return None

    # Extract command name
    command_end = i
    i -= 1
    while i >= 0 and s[i].isalpha():
        i -= 1
    if i<-1 or s[i]!='\\':
        return None
    return s[i+1:command_end+1]

def remove_command(s,command,keep_inside=False):
    def remove_command(s, command, keep_inside=False):
        '\n        Removes all occurrences of a specified LaTeX-style command from a string.\n\n        This function searches for a given command in the input string `s` and removes it, \n        along with its associated content enclosed in curly braces `{}`. If `keep_inside` \n        is set to `True`, the content inside the braces is preserved, and only the command \n        itself is removed. The function handles nested braces correctly.\n\n        Args:\n            s (str): The input string from which the command should be removed.\n            command (str): The LaTeX-style command to be removed (e.g., "\\textbf").\n            keep_inside (bool, optional): If `True`, preserves the content inside the braces \n                while removing the command. Defaults to `False`.\n\n        Returns:\n            str: The modified string with the specified command removed.\n\n        Examples:\n            >>> remove_command("This is \\textbf{bold text}.", "\\textbf")\n            \'This is bold text.\'\n            \n            >>> remove_command("This is \\textbf{bold text}.", "\\textbf", keep_inside=True)\n            \'This is bold text.\'\n            \n            >>> remove_command("Nested \\textbf{bold \\textit{italic text}} example.", "\\textbf")\n            \'Nested bold \\textit{italic text} example.\'\n        '
    pos=s.find(command)
    if pos<0:
        return s
    end_index=pos+len(command)
    level=0
    escaped=False
    #print(end_index,s[end_index])
    if end_index < len(s) and s[end_index] == "{":
        while end_index<len(s):

            if s[end_index]=='{':
                level+=1
            elif s[end_index]=='}':
                level-=1
                if level==0:
                    break
            end_index+=1
    else:
        s1="".join([s[0:pos],s[end_index:]])
    if keep_inside:
        s1="".join([s[0:pos],s[pos+len(command)+1:end_index],s[end_index+1:]])
    else:
        s1="".join([s[0:pos],s[end_index+1:]])

    if command not in s1:
        return s1
    else:
        return remove_command(s1,command,keep_inside)

import re

def convert_latex_fractions(latex_str):
    '\n    Convert non-standard fraction like \x0crac\x07lpha2 to its standard-convertable \x0crac{\x07lpha}{2}\n    We suppoort single letter,number or standard form\n    '
    pattern = r'\\frac((?:\\[a-zA-Z]+|\d|[a-zA-Z]|{[^{}]*}))((?:\\[a-zA-Z]+|\d|[a-zA-Z]|{[^{}]*}))'

    def replacer(match):
        numerator, denominator = match.group(1), match.group(2)
        wrap_num = f'{{{numerator}}}' if not (numerator.startswith('{') and numerator.endswith('}')) else numerator
        wrap_den = f'{{{denominator}}}' if not (denominator.startswith('{') and denominator.endswith('}')) else denominator
        return fr'\frac{wrap_num}{wrap_den}'

    return re.sub(pattern, replacer, latex_str)

def get_first_brace_command(s: str) -> str | None:
    """ Find the first brace """
    brace_pos = find_first_unescaped_brace(s)
    if brace_pos == -1:
        return None
    return extract_command(s, brace_pos)
def remove_overall_brace(s:str) -> str:
    '\n    Remove the overall {xxx} brace\n    '
    pos=find_first_unescaped_brace(s)
    if pos==-1:
        return s,0
    command=get_first_brace_command(s)
    if not command:

        content,final=extract_bracket_content(s,pos)
        #print(s[final])
        if final==len(s) or not '}' in s[final+1:]:
            return content,1
    return s,0

def exp_frac(s):

    def exp_frac_single(s):
        position=s.find("^\\frac")+1
        if position == 0:
            return s
        level=0
        cnt=0
        idx=position
        while idx<len(s):
            if s[idx]=='{':
                cnt+=1
            elif s[idx]=='}':
                cnt-=1
                if cnt==0:
                    level+=1
                    if level==2:
                        break
            idx+=1
        s1="".join([s[0:position],'{',s[position:idx],'}',s[idx:]])
        return s1
    s1=exp_frac_single(s)
    cnt=0
    while s1 != s and cnt<100:
        cnt+=1
        s=s1
        s1=exp_frac_single(s)
    return s

def find_all(s, sub_str, allow_overlap=True):
    indexes = []
    start = 0
    step = 1 if allow_overlap else len(sub_str)
    cnt=0
    while True and cnt<100:
        pos = s.find(sub_str, start)
        if pos == -1:
            break
        indexes.append(pos)
        start = pos + step
        cnt+=1
    return indexes

def bar_inside_vec(s):
    indices=find_all(s,"\\vec{")
    if not indices:
        return s
    for i in range(len(indices)):
        position=find_all(s,"\\vec{")[i]
        idx=position+4
        #print(s[idx])
        idx2=idx
        level=0
        while idx2<len(s):
            if s[idx2]=='{':
                level+=1
            if s[idx2]=='}':
                level-=1
                if level==0:
                    break
            idx2+=1

        s1=s[idx+1:idx2]
        #print(s1)

        s1=remove_command(s1,"\\bar",keep_inside=True)
        s2= "".join([s[0:idx+1],s1,s[idx2:]])
        s=s2
    return s
def vec_lower_idx(input_str):
    'Normalize vector subscripts for latex2sympy.'
    pattern = r'\\vec\{([^{}]+)_{([^{}]+)}\}'
    replacement = r'\\vec{\1}_{\2}'
    return re.sub(pattern, replacement, input_str)
def convert_vec_syntax(text):
    '\n    Converts LaTeX vector syntax to a standardized form.\n\n    This function processes a given text string and ensures that LaTeX vector \n    notations are consistently formatted. Specifically, it transforms instances \n    of `\x0bec xxx` into `\x0bec{xxx}`. The function handles cases where the vector \n    notation is applied to single characters, Greek letters, or LaTeX commands.\n\n    Args:\n        text (str): The input string containing LaTeX code to be processed.\n\n    Returns:\n        str: The processed string with standardized vector syntax.\n\n    Examples:\n        >>> convert_vec_syntax(r"\x0bec x + \x0bec\x07lpha + \x0bec\\Gamma")\n        \'\\vec{x} + \\vec{\\alpha} + \\vec{\\Gamma}\'\n    '

    pattern = '\\\\vec(\\s*)(\\\\?[a-zA-Z\\u03b1-\\u03c9\\u0391-\\u03a9]+)'
    replacement = r'\\vec{\2}'
    return re.sub(pattern, replacement, text)

def remove_outer_braces(tex_str):
    'Rewrite {base}_{subscript} as base_{subscript}.'
    pattern = r'\{(\\(?:[a-zA-Z]+|.)|[^{}])+\}_\{([^}]+)\}'
    return re.sub(pattern, r'\1_{\2}', tex_str)

def extract_last_equal_content(s: str, strip_whitespace: bool = True) -> str:
    "\n    Extract the content after the last occurrence of specific mathematical comparison or assignment operators.\n\n    :param strip_whitespace: If True, removes leading and trailing whitespace from the extracted content. Defaults to True.\n    (e.g., '=', '\\approx', '\\ge', '\\le', etc.) within the input string `s`. It then extracts \n    and returns the content that follows the operator. If no operator is found, the entire string \n    is returned. Optionally, leading and trailing whitespace can be stripped from the extracted content.\n\n    Args:\n        s (str): The input string to process.\n        strip_whitespace (bool): Whether to strip leading and trailing whitespace from the extracted content. Defaults to True.\n\n    Returns:\n        str: The content after the last matching operator, or the entire string if no operator is found.\n    "
    comparison_operators=('=','\\approx','\\ge','\\le','\\geq','\\leq','<','>')

    content=s
    for sign in comparison_operators:
        if sign in s:
            rfind_index = s.rfind(sign)
            if rfind_index != -1:
                content = s[rfind_index + 1:]
    if strip_whitespace:
        return content.strip()
    return content

def first_pre_process(s,extrac_box=True):
    '\n    Perform the first stage of LaTeX string preprocessing.\n\n    if not brackets_balanced(s):\n        raise ValueError("The input string has unbalanced brackets. Please check the LaTeX expression.")\n    equality or comparison operator.\n\n    Args:\n        s (str): The input LaTeX string to preprocess.\n        extrac_box (bool): If True, extracts the content inside a \'\\boxed\' command. Defaults to True.\n\n    Returns:\n        str: The preprocessed LaTeX string.\n    '
    #s=remove_non_ascii(s)
    s=s.replace('\\{','(')
    s=s.replace('\\}',')')
    if not brackets_balanced(s):
        return s
    if extrac_box:
        boxed_content=remove_command(s,'\\boxed',keep_inside=True)
    else:
        boxed_content=s
    exist_overall_brace=True
    cnt=0
    while exist_overall_brace and cnt<10:
        boxed_content,exist_overall_brace=remove_overall_brace(boxed_content)
        cnt+=1

    if '\\quad' in boxed_content:
        boxed_content = boxed_content.split('\\quad')[0]

    last_equal_content=extract_last_equal_content(boxed_content)

    exist_overall_brace=True
    cnt=0
    while exist_overall_brace and cnt<10:
        last_equal_content,exist_overall_brace=remove_overall_brace(last_equal_content)
        cnt+=1
    return last_equal_content
def second_pre_process(s):
    "\n    Perform the second stage of LaTeX string preprocessing.\n\n    This function removes or modifies specific LaTeX commands and content to standardize\n    the input string for further processing. It handles commands like '\\text', '\\mathbf',\n    and '\\mathrm', removes unnecessary content, and applies transformations such as\n    converting fractions and vector syntax.\n\n    Args:\n        s (str): The input LaTeX string to preprocess.\n\n    Returns:\n        str: The preprocessed LaTeX string.\n    "

    kill_commands=[
        '\\begin',
        '\\end'
    ]
    remove_commands=[
        '\\text',
        '\\mathbf',
        '\\mathrm',
        '\\pmb',
        '\\hat',
        '\\overline',
        '\\boldsymbol',
    ]

    remove_content=[
        '\\,','$',',','`','latex','\\left','\\right','\\text','\\mathrm','\\Bigr','\\Bigl','\n','\\]','\\[',
        '\\Big','\\bigl','\\bigr','\\biggl','\\biggr','\\displaystyle','\\boldsymbol','\\infty'
    ]
    replace_content=[
        ('\\operatorname{asin}','\\asin'),
        ('\\operatorname{sech}','\\sech'),
        ('\\operatorname{acos}','\\acos'),
        ('\\operatorname{sinh}','\\sinh'),
        ('\\dfrac','\\frac'),
        ('\\tfrac','\\frac'),
        ('\\Exp','\\exp'),
        ('\\times','\\bar{times}'),
        ('\\partial','\\bar{partial}'),
        ('\\perp','\\bar{perp}'),
        ('\\epsilon','\\varepsilon'),
        ('\\varOmega','\\Omega'),
        ('I','\\bar{I}'),
        ('_e','_{e}'),
        ('e_','\\bar{e}_'),
        ('E_','\\bar{E}_'),
        ('\\pm','+'),
        ('\\mp','-'),
        ('{+}','{p}'),
        ("{-}",'{m}'),
        ("_+",'_p'),
        ('_-',"_m")
    ]
    for command in kill_commands:
        s=remove_command(s,command,keep_inside=False)
    for command in remove_commands:
        s=remove_command(s,command,keep_inside=True)
    for content in remove_content:
        s=s.replace(content,'')
    for content in replace_content:
        s=s.replace(content[0],content[1])
    s=convert_latex_fractions(s)
    #print(s)
    s=bar_inside_vec(s)
    s=vec_lower_idx(s)
    s=convert_vec_syntax(s)
    s=exp_frac(s)
    #s=remove_outer_braces(s)
    if s and s[-1] == '.':
        return s[:-1]
    return s

class MyConfig:

    interpret_as_mixed_fractions: bool = False
    interpret_simple_eq_as_assignment: bool = False
    interpret_contains_as_eq: bool = True
    lowercase_symbols: bool = False
    '\n    Args:\n        interpret_as_mixed_fractions (bool): Whether to interpert 2 \x0crac{1}{2} as 2/2 or 2 + 1/2\n        interpret_simple_eq_as_assignment (bool): Whether to interpret simple equations as assignments k=1 -> 1\n        interpret_contains_as_eq (bool): Whether to interpret contains as equality x \\in {1,2,3} -> x = {1,2,3}\n        lowercase_symbols (bool): Whether to lowercase all symbols\n    '
class MyNormalization:
    'Configuration for latex normalization.\n    \n    Each field controls a group of related normalizations:\n    - basic_latex: Basic latex command replacements (mathrm, displaystyle, etc.)\n    - units: Remove units and their variations\n    - malformed_operators: Fix malformed operators (sqrt, frac, etc.)\n    - nits: Small formatting fixes (spaces, dots, etc.)\n    - boxed: Extract content from boxed environments\n    - equations: Handle equation splitting and approximations (deprecated)\n    '
    basic_latex: bool = True
    units: bool = False
    malformed_operators: bool = True
    nits: bool = True
    boxed = "all"
    equations: bool = False

def master_convert(s):
    '\n    The only function needed to convert a LaTeX string into a SymPy expression.\n\n    Args:\n        s (str): The input LaTeX string. It should be a valid LaTeX mathematical expression, \n                 such as equations, fractions, or symbols, and must have balanced brackets.\n\n    Returns:\n        Sym (Sympy Expression): A SymPy expression representing the mathematical content of the input string.\n                                The returned object can be used for symbolic computation, simplification, \n                                or evaluation using SymPy\'s functionality.\n\n    Example:\n        >>> master_convert("\\frac{1}{2} + x")\n        1/2 + x\n    '
    preprocessed_stage1=first_pre_process(s)

    preprocessed_stage2=second_pre_process(preprocessed_stage1)

    Sym=latex2sympy(preprocessed_stage2,normalization_config=MyNormalization(),conversion_config=MyConfig())
    return Sym

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
