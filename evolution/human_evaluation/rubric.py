from typing import NamedTuple

class Rubric(NamedTuple):
    overall_enjoyment : int
    
CATEGORY_LOWER_BOUND = 1
CATEGORY_UPPER_BOUND = 20
CATEGORY_NIL_VALUE = -999

RUBRIC_INPUT_PROMPT = f"Rate your overall enjoyment on a scale of " + \
                      f"{CATEGORY_LOWER_BOUND} to {CATEGORY_UPPER_BOUND} " + \
                      f"(or enter {CATEGORY_NIL_VALUE} to skip): "

INVALID_NUMBER_OF_VALUES_MESSAGE = "Invalid number of values given, " + \
                                  f"should have {len(Rubric._fields)} value(s)"
                      
INVALID_CATEGORY_VALUE_MESSAGE = f"All values must be between " + \
                                   f"{CATEGORY_LOWER_BOUND} and " + \
                                   f"{CATEGORY_UPPER_BOUND} inclusive, " + \
                                   f"or exactly {CATEGORY_NIL_VALUE}"

def is_valid_category_value(v):
    return CATEGORY_LOWER_BOUND <= v <= CATEGORY_UPPER_BOUND or \
           v == CATEGORY_NIL_VALUE
           
def is_not_nil(v):
    return v != CATEGORY_NIL_VALUE
    
def input_rubric():
    while True:
        input_line = input(RUBRIC_INPUT_PROMPT)
        values = list(map(int, input_line.split()))
        if len(values) != len(Rubric._fields):
            print(INVALID_NUMBER_OF_VALUES_MESSAGE)
        elif not all(map(is_valid_category_value, values)):
            print(INVALID_CATEGORY_VALUE_MESSAGE)
        else:
            if all(map(is_not_nil, values)):
                return Rubric(*values)
            else:
                return None

def rubric_score(r):
    return sum(r)