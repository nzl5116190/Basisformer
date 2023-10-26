import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

def plot_seq_feature(pred_, true_, history_,label = "train",error = False,input='',wv=''):
    assert(pred_.shape == true_.shape)

    index = -1
    if pred_.shape[2]>800:
        index = 840
    pred = pred_.detach().clone()[..., index].unsqueeze(2)
    true = true_.detach().clone()[..., index].unsqueeze(2)
    history = history_.detach().clone()[..., index].unsqueeze(2)

    if len(pred.shape) == 3:  #BLD
        if error == False:
            pred = pred[0]
            true = true[0]
            history = history[0]
        else:
            largest_loss = 0
            largest_index = 0
            criterion = nn.MSELoss()
            for i in range(pred.shape[0]):
                loss = criterion(pred[i],true[i])
                if  loss > largest_loss:
                    largest_loss = loss
                    largest_index = i
            pred = pred[largest_index]
            true = true[largest_index]
            history = history[largest_index]
            input_error = input[largest_index]
            # wv_error = wv[largest_index]
            # print('input mean',input_error.mean())
            # print('input std',input_error.std())
            # print('out mean',true.mean())
            # print('out std',true.std())
            # print('wv mean',wv_error.mean())
            # print('wv std',wv_error.std())
            # print('end')

    pred = pred.cpu().numpy()
    true = true.cpu().numpy()
    history = history.cpu().numpy()

    L, D = pred.shape
    L_h,D_h = history.shape
    # if D == 1:
    #     pic_row, pic_col = 1, 1
    # else:
    #     pic_col = 2
    #     pic_row = math.ceil(D/pic_col)
    pic_row, pic_col = D, 1


    fig = plt.figure(figsize=(8*pic_row,8*pic_col))
    for i in range(1):
        ax = plt.subplot(pic_row,pic_col,i+1)
        ax.plot(np.arange(L_h), history[:, i], label = "history")
        ax.plot(np.arange(L_h,L_h+L), pred[:, i], label = "pred")
        ax.plot(np.arange(L_h,L_h+L), true[:, i], label = "true")
        ax.set_title("dimension = {},  ".format(i) + label)
        ax.legend()

    return fig
