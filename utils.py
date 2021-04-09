from datetime import datetime
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import pandas as pd

# Save checkpoint
def save_checkpoint(model, optimizer, criterion, experiment_name='test', epoch=None,
                    time_str=None):
    if not time_str:
        time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = '{}_{}'.format(time_str, experiment_name)
    if epoch is not None:
        fname += '_e{:03d}'.format(epoch)
    fname += '.pth.tar'

    checkpoints_dir = '_checkpoints'
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    fname_path = os.path.join(checkpoints_dir, fname)

    checkpoint_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict()
    }

    if criterion.learn_beta:
        checkpoint_dict.update({'criterion_state_dict': criterion.state_dict()})

    torch.save(checkpoint_dict, fname_path)

    return fname_path

def draw_labels(label1, label2): # label1 and label2 are 1-D lists

    fig = plt.figure()
    ax = fig.gca(projection = '3d')


    y = range(len(x1))
    z = range(len(x1))

    ax.plot(label1,y,z)
    ax.legend()
    plt.show()



# loading labels for video 0.hvec

labels = pd.read_csv('labels.csv')
x1 = labels['label_1'].tolist()[:400]
x2 = labels['label_2'].tolist()[:400]
draw_labels(x1,x2)


