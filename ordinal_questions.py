import math
import copy
import json
import os
def d2(x,y):
    r1 = math.hypot(x[0], x[1])
    r2 = math.hypot(y[0], y[1])
    if r1 > r2:
        r1, r2 = r2, r1
        x, y = y , x
    phi1 = math.atan2(x[1], x[0])
    phi2 = math.atan2(y[1], y[0])

    diff_angle = abs(phi1 - phi2)
    diff_angle %= (2*math.pi)
    diff_angle = min(diff_angle, 2*math.pi - diff_angle)
    return (r2 - r1) + r1 * diff_angle

def correct_ordinal_questions(result, distance_matrix, same_ring=False):
    # d2 = distance in the world
    total_questions = 0
    correct_answers = 0
    n = len(result)
    matrix = [[{} for i in range(len(result)-1)] for j in range(n)]
    A_array = [{} for i in range(len(result))]
    by_positions = [{"id": i, "questions": 0, "answers": 0} for i in range(n-1)]
    position_matrix = [[{"id_x": i, "id_y": j, "questions": 0, "answers": 0} for j in range(n-1)] for i in range(n-1)]
    for x in range(n):
        in_world = list(range(n))
        in_world.remove(x)
        in_world.sort(key=lambda y: d2(result[x], result[y]))

        A_questions = 0
        A_answers = 0
        for i in range(n-1):
            B_questions = 0
            B_answers = 0
            for j in range(n-1):
                if i==j:
                    continue
                if d2(result[x], result[in_world[i]]) == d2(result[x], result[in_world[j]]):
                    continue
                if same_ring and abs(math.hypot(result[in_world[i]][0], result[in_world[i]][1]) - math.hypot(result[in_world[j]][0], result[in_world[j]][1])) > 1e-6:
                    continue
                total_questions += 1
                position_matrix[i][j]["questions"] += 1
                A_questions += 1
                B_questions += 1
                # d2(j) > d2(i)
                if (j-i)*(distance_matrix[x][in_world[j]]-distance_matrix[x][in_world[i]])>=0:
                    correct_answers += 1
                    A_answers += 1
                    B_answers += 1
                    position_matrix[i][j]["answers"] += 1
            
            by_positions[i]["questions"] += B_questions
            by_positions[i]["answers"] += B_answers
            matrix[x][i] = {"A": x, "B": i, "questions": B_questions, "answers": B_answers, "P": B_answers / max(1, B_questions)}
        
        A_array[x] = {"A": x, "questions": A_questions, "answers": A_answers, "P": A_answers / max(1, A_questions)}

    for i in range(n-1):
        by_positions[i]["P"] = by_positions[i]["answers"] / max(1, by_positions[i]["questions"])
    
    for i in range(len(position_matrix)):
        for j in range(i+1, len(position_matrix[i])):
            position_matrix[i][j]["P"] = position_matrix[i][j]["answers"] / max(1, position_matrix[i][j]["questions"])
    
    return ({"questions": total_questions, "answers": correct_answers, "P": correct_answers / max(1, total_questions)}, A_array, matrix, by_positions, position_matrix)

def question_answers(everything, d1, file_name, matrix_time, running_time):
    result = {}
    result["Running time"] = running_time
    result["Generating distance matrix time"] = matrix_time
    total, A_array, res_mat,res_pos, res_pos_mat = correct_ordinal_questions(everything, d1)
    result["All ordinal questions"] = total
    result["All ordinal questions for each A"] = A_array
    result["Ordinal questions for each A, B"] = res_mat
    result["Ordinal questions for each position"] = res_pos
    result["Ordinal questions for each pair of positions"] = res_pos_mat

    total, A_array, res_mat,res_pos, res_pos_mat = correct_ordinal_questions(everything, d1, same_ring=True)

    result["All ordinal questions (same ring)"] = total
    result["All ordinal questions for each A (same ring)"] = A_array
    result["Ordinal questions for each A, B (same ring)"] = res_mat
    result["Ordinal questions for each position (same ring)"] = res_pos
    result["Ordinal questions for each pair of positions (same ring)"] = res_pos_mat

    with open(os.path.join(file_name, "info.txt"),"w") as f:
        json.dump(result, f, indent=2)