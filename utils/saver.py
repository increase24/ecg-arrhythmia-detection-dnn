import pickle
import torch
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(comfusion_matrix, classes, save_name):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = comfusion_matrix[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.3f" % (c,), color='white', fontsize=15, va='center', ha='center')
    
    plt.imshow(comfusion_matrix, interpolation='nearest', cmap=plt.cm.binary)
    plt.title('ResECG', fontdict = {'fontsize' : 12})
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label', fontdict = {'fontsize' : 12})
    plt.xlabel('Predict label', fontdict = {'fontsize' : 12})
    
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    # show confusion matrix
    plt.savefig(save_name, format='png')
    #plt.show()
        

def save_checkpoint(state, output_dir, output_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_path = os.path.join(output_dir, output_name)
    torch.save(state, model_path)