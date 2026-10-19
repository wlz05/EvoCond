# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

PROMPT_PREFIX = '\n\u4f60\u662f\u4e00\u4e2a\u667a\u80fd\u624b\u673a\u52a9\u624b\uff0c\u80fd\u591f\u6839\u636e\u5f53\u524d\u624b\u673a\u754c\u9762\u622a\u56fe\u3001\u4efb\u52a1\u6307\u4ee4\u4ee5\u53ca\u5386\u53f2\u5df2\u5b8c\u6210\u6b65\u9aa4\uff0c\u5224\u65ad\u5e76\u63a8\u7406\u5f53\u524d\u5e94\u6267\u884c\u7684\u52a8\u4f5c\u3002\n\n\u4f60\u9700\u8981\u4ece\u4e0b\u5217\u5019\u9009\u52a8\u4f5c\u4e2d\u9009\u62e9\u4e00\u4e2a\u6700\u5408\u9002\u7684\u6267\u884c\u52a8\u4f5c\uff1a\n\n\u5019\u9009\u52a8\u4f5c\u683c\u5f0f\u5982\u4e0b\uff1a\n1. {"action_type": "click", "action_position": [x, y]}              # \u70b9\u51fb\u5c4f\u5e55\u4e0a\u7684\u67d0\u4e2a\u4f4d\u7f6e\n2. {"action_type": "type", "action_info": "<text_input>", "action_position": [x, y]}  # \u5728\u8f93\u5165\u6846\u8f93\u5165\u6587\u672c\n3. {"action_type": "scroll", "action_info": "up/down/left/right", "action_position": [x, y]}  # \u6eda\u52a8\u5c4f\u5e55\n4. {"action_type": "navigate_back"}         # \u8fd4\u56de\u4e0a\u4e00\u4e2a\u9875\u9762\n5. {"action_type": "navigate_home"}         # \u8fd4\u56de\u684c\u9762\n6. {"action_type": "wait"}                  # \u6682\u505c\u4e00\u6bb5\u65f6\u95f4\n7. {"action_type": "enter"}                 # \u6309\u4e0b\u56de\u8f66\n8. {"action_type": "complete"}              # \u8868\u793a\u4efb\u52a1\u5df2\u7ecf\u5b8c\u6210\n9. {"action_type": "impossible"}            # \u5f53\u524d\u65e0\u6cd5\u5b8c\u6210\u8be5\u4efb\u52a1\n\n\u8bf7\u4e25\u683c\u6309\u7167\u4ee5\u4e0b\u683c\u5f0f\u8f93\u51fa\u4f60\u7684\u56de\u7b54\uff1a\n<think>\u4f60\u7684\u601d\u8003\u8fc7\u7a0b</think>\n<answer>{"action_type": ..., ...}</answer>\n'

# PROMPT_PREFIX = (
#     'You are an agent who can operate an Android phone on behalf of a user. '
#     "Given the user's goal/request, you need to decide and perform actions step by step.\n\n"
#     "At each step, you will:\n"
#     "1. Observe the current screenshot and review the history of actions performed.\n"
#     "2. Clearly state your reasoning for the next action within <think> tags.\n"
#     "3. Provide your chosen action in JSON format within <answer> tags.\n\n"
#     "Available actions (JSON format):\n"
#     "- Complete the task:\n"
#     "  {\"action_type\": \"status\", \"goal_status\": \"complete\"}\n"
#     "- Mark the task infeasible:\n"
#     "  {\"action_type\": \"status\", \"goal_status\": \"infeasible\"}\n"
#     "- Answer user's question:\n"
#     "  {\"action_type\": \"answer\", \"text\": \"<answer_text>\"}\n"
#     "- Click a screen location:\n"
#     "  {\"action_type\": \"click\", \"x\": <x>, \"y\": <y>}\n"
#     "- Long press a screen location:\n"
#     "  {\"action_type\": \"long_press\", \"x\": <x>, \"y\": <y>}\n"
#     "- Input text:\n"
#     "  {\"action_type\": \"input_text\", \"text\": \"<text>\", \"x\": <x>, \"y\": <y>}\n"
#     "- Scroll the screen:\n"
#     "  {\"action_type\": \"scroll\", \"direction\": \"<up|down|left|right>\"}\n"
#     "- Navigate back:\n"
#     "  {\"action_type\": \"navigate_back\"}\n"
#     "- Navigate home:\n"
#     "  {\"action_type\": \"navigate_home\"}\n"
#     "- Press Enter key:\n"
#     "  {\"action_type\": \"keyboard_enter\"}\n"
#     "- Open an app:\n"
#     "  {\"action_type\": \"open_app\", \"app_name\": \"<name>\"}\n"
#     "- Wait:\n"
#     "  {\"action_type\": \"wait\"}\n\n"
#     "Ensure your reasoning clearly explains why you're choosing the action."
#     "\n\nExamples (make sure to use the correct format):\n"
#     "Correct:\n"
#     "<answer>{\"action_type\": \"click\", \"x\": 240, \"y\": 1020}</answer>\n"
#     "Wrong (do NOT use this format):\n"
#     "<answer>{\"action_type\": \"click\", \"x\": [240, 1020]}</answer>\n"
#     "Explanation: Do not put x and y together in a list. Use separate fields \"x\" and \"y\"."
#     "Do NOT use x2/y2 coordinates in any action. All actions should use only x and y (for clicks, long_press, etc) or direction (for scroll). Actions like swipe are NOT supported."
# )

# FORMAT_CONFIGS = {
#     "free_think": {
#     "format": "<think>...</think><answer>...</answer>",
#     "description": "Provide your reasoning process followed by a JSON-formatted Android action as your answer. "
#                    "Use separate fields for coordinates: 'x' and 'y'. DO NOT use 'position': [x, y].",
#     "example": "<think>The user wants to open Gmail. Based on the current screenshot, I can see the Gmail app icon located at coordinates (240, 1020). I will click on that icon to launch the app.</think>\n"
#                "<answer>{\"action_type\": \"click\", \"x\": 240, \"y\": 1020}</answer>"
# },
# "default": {
#     "format": "<answer>...</answer>",
#     "description": "Default format fallback. Do not use this unless explicitly specified. "
#                    "Use separate 'x' and 'y' instead of 'position'.",
#     "example": "<answer>{\"action_type\": \"click\", \"x\": 100, \"y\": 200}</answer>"
# },
# "grounding": {
#     "description": "You should first give your thought process with your observation and reasoning, and finally your answer.\n"
#                    "The observation should be described in detail about what you see in the environment.\n"
#                    "Use 'x' and 'y' fields instead of 'position'.",
#     "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
#     "example": "<think><observation>The screen shows the Android home screen. I see the WeChat icon at the bottom row, third from the left. There are also icons for Phone, Camera, and Settings nearby.</observation><reasoning>The user probably wants to open WeChat. I will click on the WeChat icon to open the app. Once inside, I should click the search bar at the top to allow the user to search for something.</reasoning></think><answer>{\"action_type\": \"click\", \"x\": 600, \"y\": 1700}</answer>"
# }
# }

FORMAT_CONFIGS = {
    "free_think": {
        "format": "<think>...</think><answer>...</answer>",
        "description": '\u5148\u7ed9\u51fa\u4f60\u7684\u63a8\u7406\u8fc7\u7a0b\uff0c\u7136\u540e\u8f93\u51fa\u4e00\u4e2a\u7b26\u5408 JSON \u683c\u5f0f\u7684 Android \u52a8\u4f5c\u3002',
        "example": '<think>\u7528\u6237\u60f3\u6253\u5f00 Gmail\u3002\u6839\u636e\u5f53\u524d\u622a\u56fe\uff0c\u6211\u770b\u5230 Gmail \u5e94\u7528\u56fe\u6807\u4f4d\u4e8e\u5750\u6807 (240, 1020)\uff0c\u6211\u5c06\u70b9\u51fb\u5b83\u3002</think>\n'
                   "<answer>{\"action_type\": \"click\", \"action_position\": [240, 1020]}</answer>"
    },
    "default": {
        "format": "<answer>...</answer>",
        "description": '\u9ed8\u8ba4\u683c\u5f0f\uff08\u4ec5\u7528\u4e8e\u515c\u5e95\uff09\u3002',
        "example": "<answer>{\"action_type\": \"click\", \"action_position\": [100, 200]}</answer>"
    },
    "grounding": {
        "description": '\u4f60\u5e94\u9996\u5148\u63d0\u4f9b\u89c2\u5bdf\uff08observation\uff09\u548c\u63a8\u7406\u8fc7\u7a0b\uff08reasoning\uff09\uff0c\u7136\u540e\u518d\u7ed9\u51fa\u52a8\u4f5c\u7b54\u6848\u3002',
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
        "example": '<think><observation>\u5c4f\u5e55\u663e\u793a Android \u4e3b\u754c\u9762\u3002\u6211\u770b\u5230\u5de6\u4e0b\u89d2\u6709 Gmail \u56fe\u6807\u3002</observation><reasoning>\u7528\u6237\u53ef\u80fd\u60f3\u6253\u5f00 Gmail\u3002\u6211\u5c06\u70b9\u51fb\u8be5\u56fe\u6807\u3002</reasoning></think>'
                   "<answer>{\"action_type\": \"click\", \"action_position\": [100, 1800]}</answer>"
    }
}

GUIDANCE = (
    'Here are some useful guidelines you need to follow:\n'
    'General:\n'
    '- Usually there will be multiple ways to complete a task, pick the'
    ' easiest one. Also when something does not work as expected (due'
    ' to various reasons), sometimes a simple retry can solve the problem,'
    " but if it doesn't (you can see that from the history),"
    ' SWITCH to other solutions.\n'
    '- Sometimes you may need to navigate the phone to gather information'
    ' needed to complete the task, for example if user asks'
    ' "what is my schedule tomorrow", then you may want to open the calendar'
    ' app (using the `open_app` action), look up information there, answer'
    " user's question (using the `answer` action) and finish (using"
    ' the `status` action with complete as goal_status).\n'
    '- For requests that are questions (or chat messages), remember to use'
    ' the `answer` action to reply to user explicitly before finish!'
    ' Merely displaying the answer on the screen is NOT sufficient (unless'
    ' the goal is something like "show me ...").\n'
    '- If the desired state is already achieved (e.g., enabling Wi-Fi when'
    " it's already on), you can just complete the task.\n"
    'Action Related:\n'
    '- Use the `open_app` action whenever you want to open an app'
    ' (nothing will happen if the app is not installed), do not use the'
    ' app drawer to open an app unless all other ways have failed.\n'
    '- Use the `input_text` action whenever you want to type'
    ' something (including password) instead of clicking characters on the'
    ' keyboard one by one. Sometimes there is some default text in the text'
    ' field you want to type in, remember to delete them before typing.\n'
    '- Consider exploring the screen by using the `scroll`'
    ' action with different directions to reveal additional content.\n'
    '- The direction parameter for the `scroll` action can be confusing'
    " sometimes as it's opposite to swipe, for example, to view content at the"
    ' bottom, the `scroll` direction should be set to "down". It has been'
    ' observed that you have difficulties in choosing the correct direction, so'
    ' if one does not work, try the opposite as well.\n'
    'Text Related Operations:\n'
    '- Normally to select certain text on the screen: <i> Enter text selection'
    ' mode by long pressing the area where the text is, then some of the words'
    ' near the long press point will be selected (highlighted with two pointers'
    ' indicating the range) and usually a text selection bar will also appear'
    ' with options like `copy`, `paste`, `select all`, etc.'
    ' <ii> Select the exact text you need. Usually the text selected from the'
    ' previous step is NOT the one you want, you need to adjust the'
    ' range by dragging the two pointers. If you want to select all text in'
    ' the text field, simply click the `select all` button in the bar.\n'
    "- At this point, you don't have the ability to drag something around the"
    ' screen, so in general you can not select arbitrary text.\n'
    '- To delete some text: the most traditional way is to place the cursor'
    ' at the right place and use the backspace button in the keyboard to'
    ' delete the characters one by one (can long press the backspace to'
    ' accelerate if there are many to delete). Another approach is to first'
    ' select the text you want to delete, then click the backspace button'
    ' in the keyboard.\n'
    '- To copy some text: first select the exact text you want to copy, which'
    ' usually also brings up the text selection bar, then click the `copy`'
    ' button in bar.\n'
    '- To paste text into a text box, first long press the'
    ' text box, then usually the text selection bar will appear with a'
    ' `paste` button in it.\n'
    '- When typing into a text field, sometimes an auto-complete dropdown'
    ' list will appear. This usually indicating this is a enum field and you'
    ' should try to select the best match by clicking the corresponding one'
    ' in the list.\n'
)

# ACTION_SELECTION_PROMPT_TEMPLATE = (
#     PROMPT_PREFIX
#     + '\nThe current user goal/request is: {goal}\n\n'
#     'Here is a history of what you have done so far:\n{history}\n\n'
#     'The current screenshot (and optionally an annotated screenshot) is also given to you.\n\n'
#     + GUIDANCE
#     + '{additional_guidelines}'
#     + '\nNow output an action from the above list in the correct JSON format,'
#     ' following the reason why you do that. Your answer should look like:\n'
#     'Reason: ...\nAction: {{"action_type": ...}}\n\n'
#     'Your Answer:\n'
# )
ACTION_SELECTION_PROMPT_TEMPLATE = (
        PROMPT_PREFIX
        + '\nThe current user goal/request is: {goal}\n\n'
          'Here is a history of what you have done so far:\n{history}\n\n'
          'The current screenshot (and optionally an annotated screenshot) is also given to you.\n\n'
        + '{additional_guidelines}'
        + '\nNow output an action from the above list in the correct JSON format,'
          ' following the reason why you do that. Your answer should look like:\n'
          'Reason: ...\nAction: {{"action_type": ...}}\n\n'
          'Your Answer:\n'
)

SUMMARY_PROMPT_TEMPLATE = (
        PROMPT_PREFIX
        + '\nThe overall user goal/request is: {goal}\n'
          'Now summarize the latest step.\n'
          'You will be given:\n'
          '- A screenshot **before** the action (with a "before" label on bottom right)\n'
          '- A screenshot **after** the action (with an "after" label)\n'
          '- The action you selected, and your reason\n\n'
          'Screenshot before the action: {before_screenshot}\n'
          'Screenshot after the action: {after_screenshot}\n'
          'Chosen action: {action}\n'
          'Reason: {reason}\n\n'
          'Now, compare the screenshots and the action taken. Write a brief summary of what happened. This will be saved into action history and used for future decisions.\n'
          'Try to capture:\n'
          '- What you intended to do and why\n'
          '- Whether it worked or not\n'
          '- If it failed, what might be the cause\n'
          '- Any helpful memory for future steps\n\n'
          '**Rules**:\n'
          '- Keep it concise (\u2264 50 words), single-line\n'
          "- Some actions like `answer`, `wait` may not cause visible changes \u2014 that's okay\n"
          '- Include only useful details that help with future decisions\n\n'
          'Summary of this step: '
)

def _action_selection_prompt(
        goal: str,
        history: list[str],
        additional_guidelines: list[str] | None = None,
) -> str:
    'Generate the prompt for the action selection.\n\n    Args:\n      goal: The current goal.\n      history: Summaries for previous steps.\n      additional_guidelines: Task specific guidelines.\u989d\u5916\u63d0\u793a\n\n    Returns:\n      The text prompt for action selection that will be sent to gpt4v.\n    '
    if history:
        history = '\n'.join(history)
    else:
        history = 'You just started, no action has been performed yet.'

    extra_guidelines = ''
    if additional_guidelines:
        extra_guidelines = 'For The Current Task:\n'
        for guideline in additional_guidelines:
            extra_guidelines += f'- {guideline}\n'

    return ACTION_SELECTION_PROMPT_TEMPLATE.format(
        goal=goal,
        history=history,
        additional_guidelines=extra_guidelines,
    )

def format_prompt_generator(format_type):
    def prompt_function(**kwargs):
        max_actions_per_step = kwargs.get("max_actions_per_step", 5)
        action_sep = kwargs.get("action_sep", ",")
        add_example = kwargs.get("add_example", True)

        config = FORMAT_CONFIGS.get(format_type, FORMAT_CONFIGS["default"])

        base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.\n{config["description"]}"""

        if format_type == "grounding":
            base_prompt += "\nYou will receive a screenshot of the current Android screen as <image>, which you should observe carefully."

        if "additional_info" in config:
            base_prompt += f"\n{config['additional_info']}"

        base_prompt += f"""\nYour response should be in the format of:\n{config["format"]}"""

        if add_example:
            example = config["example"]
            if "{action_sep}" in example:
                example = example.replace("{action_sep}", action_sep)
            return base_prompt + '\n' + f"e.g. {example}"

        return base_prompt

    return prompt_function

def init_observation_template(**kwargs):
    instruction = kwargs.get("instruction", "No instruction provided.")
    observation = kwargs.get("observation", "No observation provided.")
    return f"""[Initial Observation]:\n{observation}\nHuman Instruction: {instruction}\nDecide your next action(s)."""

def system_prompt(**kwargs):
    format_type = kwargs.get("format", "free_think")

    base_prompt = PROMPT_PREFIX

    example = FORMAT_CONFIGS.get(format_type, {}).get("example", "")

    if example:
        return f"{base_prompt}\n\nExample:\n{example}"

    return base_prompt

format_prompt = {format_type: format_prompt_generator(format_type)
                 for format_type in FORMAT_CONFIGS}

def action_template(**kwargs):
    observation = kwargs.get("observation", "No observation provided.")
    instruction = kwargs.get("instruction", "No instruction provided.")
    valid_action = kwargs.get("valid_action", "No valid action provided.")
    env_feedback = kwargs.get("env_feedback", "No environment feedback provided.")
    reward = kwargs.get("reward", "No reward provided.")
    done = kwargs.get("done", "No done status provided.")
    return f"""After your answer, the extracted valid action is {valid_action}.\nThe environment feedback is: {env_feedback}\nreward: {reward}\ndone: {done}\nAfter that, the observation is:\n{observation}\nHuman Instruction: {instruction}\nDecide your next action(s)."""

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
