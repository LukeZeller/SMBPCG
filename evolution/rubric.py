from typing import NamedTuple

class Rubric(NamedTuple):
    difficulty : int
    novelty : int
    overall_enjoyment : int
    
def input_rubric():
    input_line = input("Enter rubric score as: (difficulty, novelty, overall enjoyment): ")
    try:
        values = list(map(int, input_line.split()))
        assert len(values) == 3
    except:
        print("Invalid input")
        return input_rubric()
    return Rubric(*values)

def rubric_score(r):
    return sum(r)