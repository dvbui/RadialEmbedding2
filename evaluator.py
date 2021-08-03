from onering import *
total_folder_name = "onering (100 points random, TFIDF, NYT, min_dist=0.06)"
from plotting import *
import os
from ordinal_questions import *
from distance import *
from timeit import default_timer as timer

chosen_articles = [2982, 3902, 2867, 2103, 4385]#, 591, 264, 3675, 760, 5093, 1491, 1218, 3629, 3632, 3553, 4571, 752, 3154, 936, 3375, 3653, 3587, 3233, 1425, 5112, 1959, 5270, 223, 3329, 4442, 2278, 5237, 180, 4107, 2271, 138, 5483, 5414, 213, 1732, 659, 4964, 4464, 4719, 1462, 341, 5484, 5035, 2372, 3075, 3016, 1342, 4237, 2672, 4995, 1060, 3529, 2025, 3530, 424, 389, 4665, 873, 1313, 5465, 5134, 64, 2668, 1447, 4071, 1545, 1445, 1641, 3209, 5191, 4622, 5403, 2203, 1439, 4120, 5025, 3160, 2976, 4202, 2597, 4327, 2210, 61, 2414, 1364, 5436, 411, 3654, 5041, 4002, 2095, 1343, 4965, 2556, 93]
def test_solve():
    global total_folder_name
    if not os.path.isdir(total_folder_name):
        os.mkdir(total_folder_name)
    os.chdir(total_folder_name)
    results = []
    for i in range(0,len(chosen_articles)):
        id = chosen_articles[i]
        start_matrix = timer()
        distance_matrix = create_distance_matrix(id, False)
        end_matrix = timer()
        start = timer()
        result = fit_transform(distance_matrix, min_dist=0.06)
        end = timer()
        
        print("Elapsed time: {}".format(end-start))
        
        results.append(result)

        prefix = str(id)
        if not os.path.isdir(prefix):
            os.mkdir(prefix)
        
        distance_matrix = create_distance_matrix(id, False)
        question_answers(result, distance_matrix, prefix, end_matrix - start_matrix, end-start)
        
        print("End of test")
    
    os.chdir("..")
    return results
