import numpy as np
import math
import sys
import matplotlib.pyplot as plt

def average_list_of_lists(li):
    li.sort(key = len)

    li_avg = np.zeros(len(li[-1]))
    li_avg_mat = np.zeros((len(li), len(li[-1])))
    length_i_minus_1 = 0
    for i in range(len(li)):
        length_i = len(li[i])
        li_avg_mat[i, 0 : length_i] = np.array(li[i])

    for i in range(len(li) - 1):
        length_i = len(li[i])
        if length_i > length_i_minus_1:
            li_avg[length_i_minus_1: length_i] = np.average(np.array(li_avg_mat[i:, length_i_minus_1: length_i]), axis=0)
            length_i_minus_1 = length_i

    if length_i_minus_1 < len(li[-1]):
        li_avg[length_i_minus_1: len(li[-1])] = np.array(li_avg_mat[len(li) - 1, length_i_minus_1: len(li[-1])])
    return li_avg


def protectedDiv(left, right):
    try:
        return 1 if right == 0 else left / right
    except ZeroDivisionError:
        return 1
    

def protectedExp(arg):
    try:
        return math.exp(min(round(arg, 4), 100))
    except:
        return 1.

def protectedSqrt(arg):
    return math.sqrt(abs(arg))

def cos(arg):
    try:
        return math.cos(math.radians(arg))
    except:
        return 0

def sin(arg):
    try:
        return math.sin(math.radians(arg))
    except:
        return 0

def tan(arg):
    try:
        return math.min(math.tan(math.radians(arg)), 100)
    except:
        return 0


def plot_superposed(x_axis, y_s, x_label = 'x', y_label = 'y', plot_labels = None, sub_title = None, sup_title = None, show = True, store = False, store_path = "symbolic_regression_estimate"):
    fig, axs = plt.subplots(1, 1, constrained_layout=True)
    for i, y_i in enumerate(y_s):
        if plot_labels and i <= len(plot_labels) - 1:
            if type(y_i) is tuple:
                axs.plot(x_axis, y_i[0], label = plot_labels[i], color = y_i[1])
            else:
                axs.plot(x_axis, y_i, label = plot_labels[i])
        else:
            if type(y_i) is tuple:
                axs.plot(x_axis, y_i[0], color = y_i[1])
            else:
                axs.plot(x_axis, y_i)
    if sub_title:
        axs.set_title(sub_title)
    axs.set_xlabel(x_label)
    axs.set_ylabel(y_label)
    fig.suptitle(sup_title, fontsize=16) 
    plt.legend()
    if store:
        plt.savefig(store_path, bbox_inches='tight')
        if show:
            plt.show()
    else:
        if show:
            plt.show()

def plot_superposed_multiple_xs(x_axis_s, y_s, x_label = 'x', y_label = 'y', plot_labels = None, sub_title = None, sup_title = None, show = True, store = False, store_path = "symbolic_regression_estimate"):
    fig, axs = plt.subplots(1, 1, constrained_layout=True)
    for i, y_i in enumerate(y_s):
        if plot_labels and i <= len(plot_labels) - 1:
            if type(y_i) is tuple:
                axs.plot(x_axis_s[i], y_i[0], label = plot_labels[i], color = y_i[1])
            else:
                axs.plot(x_axis_s[i], y_i, label = plot_labels[i])
        else:
            if type(y_i) is tuple:
                axs.plot(x_axis_s[i], y_i[0], color = y_i[1])
            else:
                axs.plot(x_axis_s[i], y_i)
    if sub_title:
        axs.set_title(sub_title)
    axs.set_xlabel(x_label)
    axs.set_ylabel(y_label)
    fig.suptitle(sup_title, fontsize=16) 
    plt.legend()
    if store:
        plt.savefig(store_path, bbox_inches='tight')
        if show:
            plt.show()
    else:
        if show:
            plt.show()

def plot_grouped_bar_graph(y_label, title, labels, groups, data, savepath, width, padding, bar_label = False):
    x = np.arange(len(labels))  # the label locations
    #width = 0.05  # the width of the bars
    fig, ax = plt.subplots()
    rects = []
    if len(groups)%2 == 0:
        r = len(groups)//2
        for i, j in enumerate(range(-r, r + 1)):
            if j < 0:
                rects += [ax.bar(x + j*width + width/2, data[i], width, label=groups[i])]
            elif j > 0:
                rects += [ax.bar(x + j*width - width/2, data[i - 1], width, label=groups[i - 1])]
    else:
        r = math.floor(len(groups)/2)
        for i, j in enumerate(range(-r, r + 1)):
            rects += [ax.bar(x + j*width, data[i], width, label=groups[i])]
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    if bar_label:
        for rect in rects:
            ax.bar_label(rect, padding=padding)
    fig.tight_layout()
    plt.savefig(savepath, bbox_inches='tight')