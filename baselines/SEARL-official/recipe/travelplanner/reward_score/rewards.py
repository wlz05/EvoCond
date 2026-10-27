# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import sys
import os

# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
import re
import math
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from search_r1.tool.tools.flights.apis import Flights
from search_r1.tool.tools.accommodations.apis import Accommodations
from search_r1.tool.tools.restaurants.apis import Restaurants
from search_r1.tool.tools.googleDistanceMatrix.apis import GoogleDistanceMatrix
from search_r1.tool.tools.attractions.apis import Attractions

flight = Flights()
accommodation = Accommodations()
restaurants = Restaurants()
googleDistanceMatrix = GoogleDistanceMatrix()
attractions = Attractions()

def extract_before_parenthesis(s):
    match = re.search(r'^(.*?)\([^)]*\)', s)
    return match.group(1) if match else s

def get_valid_name_city(info):
    # Modified the pattern to preserve spaces at the end of the name
    pattern = r'(.*?),\s*([^,]+)(\(\w[\w\s]*\))?$'
    match = re.search(pattern, info)
    if match:
        return match.group(1).strip(), extract_before_parenthesis(match.group(2).strip()).strip()
    else:
        print(f"{info} can not be parsed, '-' will be used instead.")
        return "-","-"

def extract_from_to(text: str):
    '\n    Extracts \'A\' and \'B\' from the format "from A to B" in the given text, with B ending at a comma or the end of the string.\n    \n    Args:\n    - text (str): The input string.\n    \n    Returns:\n    - tuple: A tuple containing \'A\' and \'B\'. If no match is found, returns (None, None).\n    '
    pattern = r"from\s+(.+?)\s+to\s+([^,]+)(?=[,\s]|$)"
    matches = re.search(pattern, text)
    return matches.groups() if matches else (None, None)

def get_total_cost(question, tested_data):
    total_cost = 0
    for i in range(min(question['days'],len(tested_data))):
        unit = tested_data[i]
        # transporation
        if unit['transportation'] and  unit['transportation'] != '-':
            value = unit['transportation']
            org_city, dest_city = extract_from_to(value)
            if org_city == None or dest_city == None:
                org_city, dest_city = extract_from_to(unit['current_city'])

            if org_city == None or dest_city == None:
                pass
            else:
                if 'flight number' in value.lower():
                    res = flight.data[flight.data['Flight Number'] == value.split('Flight Number: ')[1].split(',')[0]]
                    if len(res) > 0:
                        total_cost += res['Price'].values[0] * question['people_number']

                elif 'self-driving' in value.lower() or 'taxi' in value.lower():
                    if 'self-driving' in value.lower():
                        cost = googleDistanceMatrix.run_for_evaluation(org_city,dest_city,'self-driving')['cost']
                        total_cost += cost * math.ceil(question['people_number'] * 1.0 / 5)
                    else:
                        cost = googleDistanceMatrix.run_for_evaluation(org_city,dest_city,'taxi')['cost']
                        total_cost += cost * math.ceil(question['people_number'] * 1.0 / 4)

        # breakfast
        if unit['breakfast'] and unit['breakfast'] != '-':
            name, city = get_valid_name_city(unit['breakfast'])
            res = restaurants.data[(restaurants.data['Name'].astype(str).str.contains(re.escape(name))) & (restaurants.data['City'] == city)]
            if len(res) > 0:
                total_cost += res['Average Cost'].values[0] * question['people_number']

        # lunch
        if unit['lunch'] and unit['lunch'] != '-':
            name, city = get_valid_name_city(unit['lunch'])
            res = restaurants.data[(restaurants.data['Name'].astype(str).str.contains(re.escape(name))) & (restaurants.data['City'] == city)]
            if len(res) > 0:
                total_cost += res['Average Cost'].values[0] * question['people_number']

        # dinner
        if unit['dinner'] and unit['dinner'] != '-':
            name, city = get_valid_name_city(unit['dinner'])
            res = restaurants.data[(restaurants.data['Name'].astype(str).str.contains(re.escape(name))) & (restaurants.data['City'] == city)]
            if len(res) > 0:
                total_cost += res['Average Cost'].values[0] * question['people_number']

        # accommodation
        if unit['accommodation'] and unit['accommodation'] != '-':
            name, city = get_valid_name_city(unit['accommodation'])
            res = accommodation.data[(accommodation.data['NAME'].astype(str).str.contains(re.escape(name))) & (accommodation.data['city'] == city)]
            if len(res) > 0:
                total_cost += res['price'].values[0] * math.ceil(question['people_number'] * 1.0 / res['maximum occupancy'].values[0])
    return total_cost

def is_valid_room_rule(question, tested_data):
    if question['local_constraint']['house rule'] is None:
        return None,None

    for i in range(min(question['days'],len(tested_data))):
        unit = tested_data[i]
        if unit['accommodation'] and unit['accommodation'] != '-':
            name, city = get_valid_name_city(unit['accommodation'])
            res = accommodation.data[(accommodation.data['NAME'].astype(str).str.contains(re.escape(name))) & (accommodation.data['city'] == city)]
            if len(res) > 0:
                if question['local_constraint']['house rule'] == 'smoking' and 'No smoking' in str(res['house_rules'].values[0]):
                    return False, f"The house rule should be {question['local_constraint']['house rule']}."
                if question['local_constraint']['house rule'] == 'parties' and 'No parties' in str(res['house_rules'].values[0]):
                    return False, f"The house rule should be {question['local_constraint']['house rule']}."
                if question['local_constraint']['house rule'] == 'children under 10' and 'No children under 10' in str(res['house_rules'].values[0]):
                    return False, f"The house rule should be {question['local_constraint']['house rule']}."
                if question['local_constraint']['house rule'] == 'visitors' and 'No visitors' in str(res['house_rules'].values[0]):
                    return False, f"The house rule should be {question['local_constraint']['house rule']}."
                if question['local_constraint']['house rule'] == 'pets' and 'No pets' in str(res['house_rules'].values[0]):
                    return False, f"The house rule should be {question['local_constraint']['house rule']}."

    return True, None

def is_valid_cuisine(question, tested_data):
    cuisine_set = set()
    if question['local_constraint']['cuisine']:
        for i in range(min(question['days'],len(tested_data))):
            unit = tested_data[i]

            if unit['breakfast'] and unit['breakfast'] != '-':
                name, city = get_valid_name_city(unit['breakfast'])
                if city == question['org']:
                    continue
                res = restaurants.data[(restaurants.data['Name'].astype(str).str.contains(re.escape(name))) & (restaurants.data['City'] == city)]
                if len(res) > 0:
                    for cuisine in question['local_constraint']['cuisine']:
                        if cuisine in res.iloc[0]['Cuisines']:
                            cuisine_set.add(cuisine)

            if unit['lunch'] and unit['lunch'] != '-':
                name, city = get_valid_name_city(unit['lunch'])
                if city == question['org']:
                    continue
                res = restaurants.data[(restaurants.data['Name'].astype(str).str.contains(re.escape(name))) & (restaurants.data['City'] == city)]
                if len(res) > 0:
                    for cuisine in question['local_constraint']['cuisine']:
                        if cuisine in res.iloc[0]['Cuisines']:
                            cuisine_set.add(cuisine)

            if unit['dinner'] and unit['dinner'] != '-':
                name, city = get_valid_name_city(unit['dinner'])
                if city == question['org']:
                    continue
                res = restaurants.data[(restaurants.data['Name'].astype(str).str.contains(re.escape(name))) & (restaurants.data['City'] == city)]
                if len(res) > 0:
                    for cuisine in question['local_constraint']['cuisine']:
                        if cuisine in res.iloc[0]['Cuisines']:
                            cuisine_set.add(cuisine)

        if len(cuisine_set) == len(question['local_constraint']['cuisine']):
            return True, None
        else:
            # judge which cuisine is not satisfied
            for cuisine in question['local_constraint']['cuisine']:
                if cuisine not in cuisine_set:
                    return False, f"The cuisine {cuisine} is not satisfied."
    else:
        return None,None

def is_valid_transportation(question, tested_data):
    if question['local_constraint']['transportation'] is None:
        return None,None
    for i in range(min(question['days'],len(tested_data))):
        unit = tested_data[i]
        if unit['transportation'] and unit['transportation'] != '-':
            value = unit['transportation']
            if question['local_constraint']['transportation'] == 'no flight' and 'Flight' in value:
                return False, f"The transportation should not be {question['local_constraint']['transportation']}."
            elif question['local_constraint']['transportation'] == 'no self-driving' and 'Self-driving'  in value:
                return False, f"The transportation should not be {question['local_constraint']['transportation']}."

    return True, None

def is_valid_room_type(question, tested_data):
    if question['local_constraint']['room type'] is None:
        return None,None
    for i in range(min(question['days'],len(tested_data))):
        unit = tested_data[i]
        if unit['accommodation'] and unit['accommodation'] != '-':
            name, city = get_valid_name_city(unit['accommodation'])
            res = accommodation.data[(accommodation.data['NAME'].astype(str).str.contains(re.escape(name))) & (accommodation.data['city'] == city)]
            if len(res) > 0:
                if question['local_constraint']['room type'] == 'not shared room' and res['room type'].values[0] == 'Shared room':
                    return False, f"The room type should be {question['local_constraint']['room type']}."
                # "shared room", "not shared room", "private room", "entire room"
                elif question['local_constraint']['room type'] == 'shared room' and res['room type'].values[0] != 'Shared room':
                    return False, f"The room type should be {question['local_constraint']['room type']}."

                elif question['local_constraint']['room type'] == 'private room' and res['room type'].values[0] != 'Private room':
                    return False, f"The room type should be {question['local_constraint']['room type']}."

                elif question['local_constraint']['room type'] == 'entire room' and res['room type'].values[0] != 'Entire home/apt':
                    return False, f"The room type should be {question['local_constraint']['room type']}."

    return True, None

def evaluation(query_data, tested_data):
    return_info = {}
    return_info['valid_cuisine'] = is_valid_cuisine(query_data, tested_data)
    return_info['valid_room_rule'] = is_valid_room_rule(query_data, tested_data)
    return_info['valid_transportation'] = is_valid_transportation(query_data, tested_data)
    return_info['valid_room_type'] = is_valid_room_type(query_data, tested_data)
    return_info['valid_cost'] = (bool(get_total_cost(query_data, tested_data) <= query_data['budget']), None)
    return return_info

def boolean_evaluation(query_data, tested_data):
    return_info = {}
    return_info['valid_cuisine'] = is_valid_cuisine(query_data, tested_data)
    return_info['valid_room_rule'] = is_valid_room_rule(query_data, tested_data)
    return_info['valid_transportation'] = is_valid_transportation(query_data, tested_data)
    return_info['valid_room_type'] = is_valid_room_type(query_data, tested_data)
    return_info['valid_cost'] = (bool(get_total_cost(query_data, tested_data) <= query_data['budget']), None)
    for key in return_info:
        if return_info[key][0] == False:
            print(key)
            return False
    return True

def compute_hard_constraint(final_answer: dict) -> float:
    'Score hard travel constraints from tested_data and optional hard_constraint_data: 1.0 for all, 0.1 for some and 0.0 for none.'
    try:

        if 'tested_data' not in final_answer:
            print('DEBUG: final_answer has no tested_data field')
            return 0.0

        tested_data = final_answer['tested_data']

        if 'hard_constraint_data' not in final_answer:
            print('DEBUG: final_answer has no hard_constraint_data field')
            return 0.0

        hard_constraint_data = final_answer['hard_constraint_data']

        constraint_results = evaluation(hard_constraint_data, tested_data)

        satisfied_constraints = 0
        total_constraints = 0

        for key, (is_valid, _) in constraint_results.items():

            if is_valid is not None:
                total_constraints += 1
                if is_valid:
                    satisfied_constraints += 1

        if total_constraints == 0:
            return 0.0
        elif satisfied_constraints == 0:
            return 0.0
        elif satisfied_constraints == total_constraints:
            return 1.0
        else:
            return 0.1

    except Exception as e:
        print(f"DEBUG: Hard-constraint reward failed: {e}")
        return 0.0

basic_test_data = [
    {
        "current_city": "New York",
        "transportation": "Self-driving from New York to Boston",
        "breakfast": "Cafe Central",
        "lunch": "Pizza Place",
        "dinner": "Italian Restaurant",
        "accommodation": "Hotel ABC"
    }
]

print('Starting tests...\n')

print('Test 1: no constraints')
final_answer = {
    "tested_data": basic_test_data,
    "hard_constraint_data": {
        "house rule": None,
        "cuisine": None,
        "transportation": None,
        "room type": None
    }
}
print('Result:', compute_hard_constraint(final_answer))

print('\nTest 2: all constraints satisfied')
final_answer = {
    "tested_data": basic_test_data,
    "hard_constraint_data": {
        "house rule": "smoking",
        "cuisine": ["Italian"],
        "transportation": "no flight",
        "room type": "private room"
    }
}
print('Result:', compute_hard_constraint(final_answer))

print('\nTest 3: some constraints satisfied')
final_answer = {
    "tested_data": basic_test_data,
    "hard_constraint_data": {
        "house rule": "smoking",
        "cuisine": ["Italian"],
        "transportation": None,
        "room type": None
    }
}
print('Result:', compute_hard_constraint(final_answer))

print('\nTest 4: no constraints satisfied')
final_answer = {
    "tested_data": basic_test_data,
    "hard_constraint_data": {
        "house rule": "no smoking",
        "cuisine": ["Japanese"],
        "transportation": "no flight",
        "room type": "shared room"
    }
}
print('Result:', compute_hard_constraint(final_answer))

print('\nTests completed.')

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
