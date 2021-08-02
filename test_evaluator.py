import os
import json
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import math


algorithm_names = ["tree"]#, "onering", "onering new", "parallel", "parallel new", "solarviewsa", "solarviewgreedy"]
category = "(100 points random, TFIDF, default, min_dist=0.06)"
def error(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def solve(total_folder_name):
    total_result = {}
    def update_result():
        data = json.load(open("info.txt", "r"))
        total_result["Running time"].append(data["Running time"])
        total_result["Generating distance matrix time"].append(data["Generating distance matrix time"])
        total_result["All ordinal questions"].append(data["All ordinal questions"]["P"])
        total_result["All ordinal questions (same ring)"].append(data["All ordinal questions (same ring)"]["P"])

        focus = data["Ordinal questions for each pair of positions"]
        total_questions = 0
        total_answers = 0
        for i in range(20):
            for j in range(20):
                total_questions += focus[i][j]["questions"]
                total_answers += focus[i][j]["answers"]
        
        total_result["Ordinal questions (B, C in top 20 of A)"].append(total_answers / total_questions)

        total_questions = 0
        total_answers = 0
        for i in range(3):
            for j in range(5, len(focus[i])):
                total_questions += focus[i][j]["questions"]
                total_answers += focus[i][j]["answers"]
        total_result["Ordinal questions (B in top 3 of A, C is outside of top 5)"].append(total_answers / total_questions)


        focus = data["Ordinal questions for each pair of positions (same ring)"]

        total_questions = 0
        total_answers = 0
        for i in range(20):
            for j in range(20):
                total_questions += focus[i][j]["questions"]
                total_answers += focus[i][j]["answers"]
        
        total_result["Ordinal questions (same ring, B, C in top 20 of A)"].append(total_answers / total_questions)

        total_questions = 0
        total_answers = 0
        for i in range(3):
            for j in range(5, len(focus[i])):
                total_questions += focus[i][j]["questions"]
                total_answers += focus[i][j]["answers"]
        total_result["Ordinal questions (same ring, B in top 3 of A, C is outside of top 5)"].append(total_answers / total_questions)




    os.chdir(total_folder_name)
    total_result["Running time"] = []
    total_result["All ordinal questions"] = []
    total_result["All ordinal questions (same ring)"] = []
    total_result["Ordinal questions (B, C in top 20 of A)"] = []
    total_result["Ordinal questions (same ring, B, C in top 20 of A)"] = []
    total_result["Ordinal questions (B in top 3 of A, C is outside of top 5)"] = []
    total_result["Ordinal questions (same ring, B in top 3 of A, C is outside of top 5)"] = []
    total_result["Generating distance matrix time"] = []

    tests = os.listdir(".")
    for i in tests:
        os.chdir(i)
        #os.chdir("with_optimizer")
        update_result()
        #os.chdir("..")
        os.chdir("..")
    
    os.chdir("..")

    return total_result

def plotting(name, overall):
    fig, ax = plt.subplots()

    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
    
    cnt = 0
    for x in overall:
        for y in overall[x][name]:
            ax.scatter(cnt, y, c='#1f77b41a', s=4)

        m, e = error(overall[x][name])
        a = np.arange(m-e, m+e, 0.001)
        ax.plot(np.array([cnt+0.5] * len(a)), a, c="black", linewidth = 2)
        ax.scatter(cnt, m, c="red", s=2)

        cnt+=1
    
    plt.savefig("{}.png".format(name))
    plt.close('all')

def main():
    overall = {}
    for name in algorithm_names:
        folder_name = name + " " + category
        overall[name] = solve(folder_name)
    
    keys = {}
    keys["Running time"] = []
    keys["All ordinal questions"] = []
    keys["All ordinal questions (same ring)"] = []
    keys["Ordinal questions (B, C in top 20 of A)"] = []
    keys["Ordinal questions (same ring, B, C in top 20 of A)"] = []
    keys["Ordinal questions (B in top 3 of A, C is outside of top 5)"] = []
    keys["Ordinal questions (same ring, B in top 3 of A, C is outside of top 5)"] = []
    keys["Generating distance matrix time"] = []
    for k in keys:
        plotting(k, overall)