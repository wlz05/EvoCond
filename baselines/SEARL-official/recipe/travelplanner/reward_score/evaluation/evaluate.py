# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from .commonsense import evaluation as commonsense_eval
from .hard import evaluation as hard_eval
# def convert_json_plan_to_string_format(json_plan):
#     """

#     Args:

#     Returns:

#     """
#     string_plan = []

#     for day in json_plan:
#         day_dict = {
#             "days": day["days"]
#         }

#         if isinstance(day.get("current_city"), dict):
#             if "from" in day["current_city"] and "to" in day["current_city"]:
#                 day_dict["current_city"] = f"from {day['current_city']['from']} to {day['current_city']['to']}"
#             else:

#                 day_dict["current_city"] = str(day["current_city"])
#         else:
#             day_dict["current_city"] = str(day.get("current_city", ""))

#         trans = day.get("transportation", "-")
#         if isinstance(trans, dict):
#             if trans.get("type") == "flight":
#                 day_dict["transportation"] = (
#                     f"Flight Number: {trans.get('flight_number', '')}, "
#                     f"from {trans.get('from', '')} to {trans.get('to', '')}, "
#                     f"Departure Time: {trans.get('departure_time', '')}, "
#                     f"Arrival Time: {trans.get('arrival_time', '')}"
#                 )
#             elif trans.get("type") == "taxi":
#                 day_dict["transportation"] = f"Taxi; Cost: {trans.get('cost', 0)}"
#             elif trans.get("type") == "self-driving":
#                 day_dict["transportation"] = f"Self-driving; Cost: {trans.get('cost', 0)}"
#             else:
#                 day_dict["transportation"] = str(trans)
#         else:
#             day_dict["transportation"] = str(trans)

#         for meal in ["breakfast", "lunch", "dinner"]:
#             meal_info = day.get(meal, "-")
#             if isinstance(meal_info, dict):
#                 cost_str = f"; Cost: {meal_info.get('cost', '')}" if "cost" in meal_info else ""
#                 day_dict[meal] = f"{meal_info.get('name', '')}, {meal_info.get('city', '')}{cost_str}"
#             else:
#                 day_dict[meal] = str(meal_info)

#         attractions = day.get("attraction", [])
#         if isinstance(attractions, list) and len(attractions) > 0:
#             attraction_str = ""
#             for attr in attractions:
#                 if isinstance(attr, dict):
#                     attraction_str += f"{attr.get('name', '')}, {attr.get('city', '')};"
#                 else:
#                     attraction_str += f"{str(attr)};"
#             day_dict["attraction"] = attraction_str
#         else:
#             day_dict["attraction"] = str(attractions)

#         accom = day.get("accommodation", "-")
#         if isinstance(accom, dict):
#             nights_str = f" for {accom.get('nights', '')} nights" if "nights" in accom else ""
#             cost_str = f"; Cost: {accom.get('cost', '')}" if "cost" in accom else ""
#             day_dict["accommodation"] = f"{accom.get('name', '')}, {accom.get('city', '')}{cost_str}{nights_str}"
#         else:
#             day_dict["accommodation"] = str(accom)

#         string_plan.append(day_dict)

#     return string_plan

def evaluate_plan(query_data: dict, plan: list, weight: list=[1,1,1,1,1,1]) -> dict:
    '\n    \u8bc4\u4f30\u5355\u4e2a\u65c5\u884c\u8ba1\u5212\u662f\u5426\u6ee1\u8db3\u7ed9\u5b9a\u7684\u7ea6\u675f\u6761\u4ef6\n\n    Args:\n        query_data (dict): \u7ea6\u675f\u6761\u4ef6\n            {\n                "level": "easy/medium/hard",\n                "days": 3/5/7,\n                "org": "\u51fa\u53d1\u57ce\u5e02",\n                "dest": "\u76ee\u7684\u5730\u5dde",\n                "budget": \u9884\u7b97\u91d1\u989d,\n                "local_constraint": {\n                    "house rule": "\u623f\u95f4\u89c4\u5219\u7ea6\u675f" \u6216 None,\n                    "cuisine": "\u9910\u5385\u7c7b\u578b\u7ea6\u675f" \u6216 None,\n                    "room type": "\u623f\u95f4\u7c7b\u578b\u7ea6\u675f" \u6216 None,\n                    "transportation": "\u4ea4\u901a\u65b9\u5f0f\u7ea6\u675f" \u6216 None\n                }\n            }\n        \n        plan (list): \u5f85\u8bc4\u4f30\u7684\u65c5\u884c\u8ba1\u5212\uff0c\u6bcf\u4e2a\u5143\u7d20\u662f\u4e00\u5929\u7684\u884c\u7a0b\n            [\n                {\n                    "days": \u5929\u6570,\n                    "current_city": "\u5f53\u524d\u57ce\u5e02",\n                    "transportation": "\u4ea4\u901a\u4fe1\u606f",\n                    "breakfast": "\u65e9\u9910\u4fe1\u606f",\n                    "attraction": "\u666f\u70b9\u4fe1\u606f",\n                    "lunch": "\u5348\u9910\u4fe1\u606f",\n                    "dinner": "\u665a\u9910\u4fe1\u606f",\n                    "accommodation": "\u4f4f\u5bbf\u4fe1\u606f"\n                },\n                ...\n            ]\n\n    Returns:\n        dict: \u8bc4\u4f30\u7ed3\u679c\uff0c\u53ea\u5305\u542b\u4e94\u4e2a\u5173\u952e\u6307\u6807\n    '

    commonsense_info_box = commonsense_eval(query_data, plan)

    hard_info_box = None
    if (commonsense_info_box and
        commonsense_info_box['is_not_absent'][0] and
        commonsense_info_box['is_valid_information_in_sandbox'][0]):
        hard_info_box = hard_eval(query_data, plan)

    result = {
        'Delivery Rate': 1 if plan else 0
    }

    if commonsense_info_box:
        commonsense_pass = True
        commonsense_count = 0
        commonsense_pass_count = 0

        for key, value in commonsense_info_box.items():
            if value[0] is not None:
                commonsense_count += 1
                if value[0]:
                    commonsense_pass_count += 1
                else:
                    commonsense_pass = False

        result['Commonsense Constraint Micro Pass Rate'] = (
            commonsense_pass_count / commonsense_count if commonsense_count > 0 else 0
        )
        result['Commonsense Constraint Macro Pass Rate'] = 1 if commonsense_pass else 0
    else:
        result['Commonsense Constraint Micro Pass Rate'] = 0
        result['Commonsense Constraint Macro Pass Rate'] = 0

    if hard_info_box:
        hard_pass = True
        hard_count = 0
        hard_pass_count = 0

        for key, value in hard_info_box.items():
            if value[0] is not None:
                hard_count += 1
                if value[0]:
                    hard_pass_count += 1
                else:
                    hard_pass = False

        result['Hard Constraint Micro Pass Rate'] = (
            hard_pass_count / hard_count if hard_count > 0 else 0
        )
        result['Hard Constraint Macro Pass Rate'] = 1 if hard_pass else 0
    else:
        result['Hard Constraint Micro Pass Rate'] = 0
        result['Hard Constraint Macro Pass Rate'] = 0

    commonsense_pass = result['Commonsense Constraint Macro Pass Rate'] == 1
    hard_pass = result['Hard Constraint Macro Pass Rate'] == 1
    result['Final Pass Rate'] = 1 if (commonsense_pass and hard_pass) else 0
    reward = 0
    for i, (key, value) in enumerate(result.items()):
        reward += weight[i] * value
    return reward

def evaluate_plan_infer(query_data: dict, plan: list, weight: list=[1,1,1,1,1,1]) -> dict:
    '\n    \u8bc4\u4f30\u5355\u4e2a\u65c5\u884c\u8ba1\u5212\u662f\u5426\u6ee1\u8db3\u7ed9\u5b9a\u7684\u7ea6\u675f\u6761\u4ef6\n\n    Args:\n        query_data (dict): \u7ea6\u675f\u6761\u4ef6\n            {\n                "level": "easy/medium/hard",\n                "days": 3/5/7,\n                "org": "\u51fa\u53d1\u57ce\u5e02",\n                "dest": "\u76ee\u7684\u5730\u5dde",\n                "budget": \u9884\u7b97\u91d1\u989d,\n                "local_constraint": {\n                    "house rule": "\u623f\u95f4\u89c4\u5219\u7ea6\u675f" \u6216 None,\n                    "cuisine": "\u9910\u5385\u7c7b\u578b\u7ea6\u675f" \u6216 None,\n                    "room type": "\u623f\u95f4\u7c7b\u578b\u7ea6\u675f" \u6216 None,\n                    "transportation": "\u4ea4\u901a\u65b9\u5f0f\u7ea6\u675f" \u6216 None\n                }\n            }\n        \n        plan (list): \u5f85\u8bc4\u4f30\u7684\u65c5\u884c\u8ba1\u5212\uff0c\u6bcf\u4e2a\u5143\u7d20\u662f\u4e00\u5929\u7684\u884c\u7a0b\n            [\n                {\n                    "days": \u5929\u6570,\n                    "current_city": "\u5f53\u524d\u57ce\u5e02",\n                    "transportation": "\u4ea4\u901a\u4fe1\u606f",\n                    "breakfast": "\u65e9\u9910\u4fe1\u606f",\n                    "attraction": "\u666f\u70b9\u4fe1\u606f",\n                    "lunch": "\u5348\u9910\u4fe1\u606f",\n                    "dinner": "\u665a\u9910\u4fe1\u606f",\n                    "accommodation": "\u4f4f\u5bbf\u4fe1\u606f"\n                },\n                ...\n            ]\n\n    Returns:\n        dict: \u8bc4\u4f30\u7ed3\u679c\uff0c\u53ea\u5305\u542b\u4e94\u4e2a\u5173\u952e\u6307\u6807\n    '

    commonsense_info_box = commonsense_eval(query_data, plan)

    hard_info_box = None
    if (commonsense_info_box and
        commonsense_info_box['is_not_absent'][0] and
        commonsense_info_box['is_valid_information_in_sandbox'][0]):
        hard_info_box = hard_eval(query_data, plan)

    result = {
        'Delivery Rate': 1 if plan else 0
    }

    if commonsense_info_box:
        commonsense_pass = True
        commonsense_count = 0
        commonsense_pass_count = 0

        for key, value in commonsense_info_box.items():
            if value[0] is not None:
                commonsense_count += 1
                if value[0]:
                    commonsense_pass_count += 1
                else:
                    commonsense_pass = False

        result['Commonsense Constraint Micro Pass Rate'] = (
            commonsense_pass_count / commonsense_count if commonsense_count > 0 else 0
        )
        result['Commonsense Constraint Macro Pass Rate'] = 1 if commonsense_pass else 0
    else:
        result['Commonsense Constraint Micro Pass Rate'] = 0
        result['Commonsense Constraint Macro Pass Rate'] = 0

    if hard_info_box:
        hard_pass = True
        hard_count = 0
        hard_pass_count = 0

        for key, value in hard_info_box.items():
            if value[0] is not None:
                hard_count += 1
                if value[0]:
                    hard_pass_count += 1
                else:
                    hard_pass = False

        result['Hard Constraint Micro Pass Rate'] = (
            hard_pass_count / hard_count if hard_count > 0 else 0
        )
        result['Hard Constraint Macro Pass Rate'] = 1 if hard_pass else 0
    else:
        result['Hard Constraint Micro Pass Rate'] = 0
        result['Hard Constraint Macro Pass Rate'] = 0

    commonsense_pass = result['Commonsense Constraint Macro Pass Rate'] == 1
    hard_pass = result['Hard Constraint Macro Pass Rate'] == 1
    result['Final Pass Rate'] = 1 if (commonsense_pass and hard_pass) else 0
    reward = 0
    for i, (key, value) in enumerate(result.items()):
        reward += weight[i] * value
    return reward, result

if __name__ == "__main__":

    query_data_example = {
        "level": "medium",
        "days": 7,
        "org": "Myrtle Beach",
        "dest": "Michigan",
        "budget": 8300,
        "local_constraint": {
            "house rule": None,
            "cuisine": ["French", "American"],
            "room type": None,
            "transportation": None
        }
    }

#     plan_example = [
#   {
#     "day": 1,
#     "current_city": "from Ithaca to Charlotte",
#     "transportation": "Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:38, Arrival Time: 07:46",
#     "breakfast": "Nagaland's Kitchen, Charlotte",
#     "attraction": "The Charlotte Museum of History, Charlotte",
#     "lunch": "Cafe Maple Street, Charlotte",
#     "dinner": "Bombay Vada Pav, Charlotte",
#     "accommodation": "Affordable Spacious Refurbished Room in Bushwick!, Charlotte"
#   },
#   {
#     "day": 2,
#     "current_city": "Charlotte",
#     "transportation": "-",
#     "breakfast": "Olive Tree Cafe, Charlotte",
#     "attraction": "The Mint Museum, Charlotte;Romare Bearden Park, Charlotte",
#     "lunch": "Birbal Ji Dhaba, Charlotte",
#     "dinner": "Pind Balluchi, Charlotte",
#     "accommodation": "Affordable Spacious Refurbished Room in Bushwick!, Charlotte"
#   },
#   {
#     "day": 3,
#     "current_city": "from Charlotte to Ithaca",
#     "transportation": "Flight Number: F3786167, from Charlotte to Ithaca, Departure Time: 21:42, Arrival Time: 23:26",
#     "breakfast": "Subway, Charlotte",
#     "attraction": "Books Monument, Charlotte",
#     "lunch": "Olive Tree Cafe, Charlotte",
#     "dinner": "Kylin Skybar, Charlotte",
#     "accommodation": "-"
#   }
# ]
    test_file = 'assets/data/final_annotation_easy_test.jsonl'
    with open(test_file, 'r') as f:
        query_ls = json.load(f)

    output_path = 'assets/data/output1.json'
    with open(output_path, 'r') as f:
        plan_ls = json.load(f)

    for query, plan in zip(query_ls, plan_ls):
        final_answer_reward = evaluate_plan(query, plan['answer'])
        print(final_answer_reward)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
