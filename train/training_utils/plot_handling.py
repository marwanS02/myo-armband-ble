import matplotlib.pyplot as plt
import numpy as np

def create_plot(metric):
    fig, ax = plt.subplots()
    ax.set_xlabel('Epochs')
    ax.set_ylabel(f'{metric}')
    line_train, = ax.plot([], [], label=f'Training {metric}')
    line_val, = ax.plot([], [], label=f'Validation {metric}')
    ax.legend()
    display(fig)
    return fig, ax, line_train, line_val

def update_plot(epoch, train_metric_list, val_metric_list,fig, ax, line_train, line_val):
    line_train.set_xdata(np.arange(1, epoch + 2))
    line_train.set_ydata(train_metric_list)
    line_val.set_xdata(np.arange(1, epoch+2))
    line_val.set_ydata(val_metric_list)
    ax.relim()
    ax.autoscale_view()
    display(fig)
