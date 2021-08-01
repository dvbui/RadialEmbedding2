import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import umap.umap_ as umap

matplotlib.use("Agg")
font = {'size'   : 5}

matplotlib.rc('font', **font)

def create_photo(data, focus, file_name, with_label=True):
    fig, ax = plt.subplots()

    plt.title(file_name)
    # Move the left and bottom spines to x = 0 and y = 0, respectively.
    #ax.spines[["left", "bottom"]].set_position(("data", 0))
    # Hide the top and right spines.
    #ax.spines[["top", "right"]].set_visible(False)

    list_of_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] * 100
    # Draw arrows (as black triangles: ">k"/"^k") at the end of the axes.  In each
    # case, one of the coordinates (0) is a data coordinate (i.e., y = 0 or x = 0,
    # respectively) and the other one (1) is an axes coordinate (i.e., at the very
    # right/top of the axes).  Also, disable clipping (clip_on=False) as the marker
    # actually spills out of the axes.
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

    ax.scatter(0, 0, c=list_of_colors[focus["group"]])
    if with_label:
        ax.annotate(str(focus["object"])[:10], (0,0))
    for i in range(len(data)):
        ax.scatter(data[i][1][0], data[i][1][1], c=list_of_colors[data[i][0]["group"]])
        r = math.hypot(data[i][1][0], data[i][1][1])
        arc = np.arange(0, math.pi*2, 0.001)
        ax.plot(r*np.cos(arc), r*np.sin(arc), c="gray", linewidth=0.1)
        if with_label:
            ax.annotate(str(data[i][0]["object"])[:10], (data[i][1][0], data[i][1][1]))
    
    if with_label:
        plt.savefig(os.path.join(file_name, "embedding.png"), dpi=1000)
    else:
        plt.savefig(os.path.join(file_name, "embedding_no_labels.png"))
    
    plt.close('all')


def create_debug_photo(positions, i, j, events, loop_iteration, test_name, after=False):
    fig, ax = plt.subplots()
    ax.spines[["left", "bottom"]].set_position(("data", 0))
    ax.spines[["top", "right"]].set_visible(False)

    list_of_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] * 100
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
    
    for x in range(len(positions)):
        for y in range(len(positions[x])):
            tmp = positions[x][y]
            ax.scatter(tmp[1][0], tmp[1][1], c=list_of_colors[tmp[0]["group"]])
            r = math.hypot(tmp[1][0], tmp[1][1])
            arc = np.arange(0, math.pi*2, 0.001)
            ax.plot(r*np.cos(arc), r*np.sin(arc), c="gray", linewidth=0.1)
            if (x,y) == (i,j):
                ax.annotate("chosen", (tmp[1][0], tmp[1][1]))
    
    start_blue = 0
    stop_blue = 2*math.pi
    start_red = {}
    stop_red = {}
    for k in sorted(events.keys()):
        for event in events[k]:
            if event["event"] == "start allowing placement":
                start_blue = k
            if event["event"] == "stop allowing placement":
                stop_blue = k
            if event["event"] == "start forbidden":
                start_red[event["point"]] = k
            if event["event"] == "stop forbidden":
                stop_red[event["point"]] = k
    
    if start_blue > stop_blue:
        stop_blue += math.pi*2
    for k in start_red:
        if start_red[k] > stop_red[k]:
            stop_red[k] += math.pi*2
    
    r = math.hypot(positions[i][j][1][0], positions[i][j][1][1])
    arc = np.arange(start_blue, stop_blue, 0.001)
    ax.plot(r*np.cos(arc), r*np.sin(arc), c="blue", linewidth = 1)

    for k in start_red:
        arc = np.arange(start_red[k], stop_red[k], 0.001)
        ax.plot(r*np.cos(arc), r*np.sin(arc), c="red", linewidth = 1)
    
    plt.savefig(os.path.join(".", "debug", test_name, str(loop_iteration), "embedding_{}_{}_{}.png".format(i,j,"B" if after else "A")))

    plt.close('all')

def create_line_photo(data, focus, file_name, with_label=True):
    fig, ax = plt.subplots()

    plt.title(file_name)
    # Move the left and bottom spines to x = 0 and y = 0, respectively.
    #ax.spines[["left", "bottom"]].set_position(("data", 0))
    # Hide the top and right spines.
    #ax.spines[["top", "right"]].set_visible(False)

    list_of_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] * 100
    # Draw arrows (as black triangles: ">k"/"^k") at the end of the axes.  In each
    # case, one of the coordinates (0) is a data coordinate (i.e., y = 0 or x = 0,
    # respectively) and the other one (1) is an axes coordinate (i.e., at the very
    # right/top of the axes).  Also, disable clipping (clip_on=False) as the marker
    # actually spills out of the axes.
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

    ax.scatter(0, 0, c=list_of_colors[focus["group"]])
    if with_label:
        ax.annotate(str(focus["object"])[:10], (0,0))
    
    for i in range(len(data)):
        r = math.hypot(data[i][1][0], data[i][1][1])
        angle = math.atan2(data[i][1][1], data[i][1][0])

        ax.scatter(r*angle, r, c=list_of_colors[data[i][0]["group"]])
        r = math.hypot(data[i][1][0], data[i][1][1])
        if with_label:
            ax.annotate(str(data[i][0]["object"])[:10], (r*angle, r))
    
    print(os.path.join(file_name, "line_embedding.png"))
    if with_label:
        plt.savefig(os.path.join(file_name, "line_embedding.png"))
    else:
        plt.savefig(os.path.join(file_name, "line_embedding_no_labels.png"))
    
    plt.close('all')

def create_umap(result, distance_matrix, file_name, with_label=True):
    n_neighborer = [5, 15, 25, 100]
    list_of_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] * 100
    for n_neighbors in n_neighborer:
        fit = umap.UMAP(n_neighbors=n_neighbors,metric="precomputed")
        X = [[d1(x,y) for x in data] for y in data]
        u = fit.fit_transform(X)
        fig, ax = plt.subplots()
        color = []
        for i in range(len(data)):
            ax.annotate(str(data[i]["object"]), (u[i][0], u[i][1]))
            color.append(list_of_colors[data[i]["group"]])
        ax.scatter(u[:,0], u[:,1], c=color)
        plt.savefig(os.path.join(file_name, "umap_{}.png".format(n_neighbors)), dpi=1000)
        plt.close('all')

def create_radius_bar_chart(focus, data, d1, file_name):
    x = np.array([d1(focus, d) for d in data])
    fig, ax = plt.subplots()
    ax.hist(x, bins=10)
    plt.savefig(os.path.join(file_name, "radius_distribution.png"))
    plt.close('all')



def create_bar_chart(data, file_name):
    fig, ax = plt.subplots()

    ax.bar(list(data.keys()), list(data.values()), width=0.4)

    plt.savefig(file_name)

    plt.close('all')
