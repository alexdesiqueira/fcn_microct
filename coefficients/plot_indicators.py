import csv
import matplotlib.pyplot as plt
import numpy as np


# setting up the figures appearance.
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.size'] = 30
plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
plt.rcParams['axes.titlesize'] = 1.2*plt.rcParams['font.size']
plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']

# the files we are going to use.
FNAME_TIRAMISU = 'larson2019_tiramisu-67/output.train_tiramisu-67.py'
FNAME_UNET = 'larson2019_unet/output.train_unet.py'


def split_epochs(filename):
    """
    """
    content, idx_epochs, epochs = [], [], []

    with open(filename) as file:
        for row in csv.reader(file, delimiter='-'):
            content.append(row)

    for idx, row in enumerate(content):
        aux = ' '.join(row)
        if 'Epoch' in aux and len(aux) < 20:
            idx_epochs.append(idx)

    for idx, _ in enumerate(idx_epochs[:-1]): 
        epochs.append(content[idx_epochs[idx]:idx_epochs[idx+1]])

    return epochs


def epoch_indicators(epoch):
    """
    """
    all_accuracy, all_loss = [], []

    for row in epoch:
        aux = ' '.join(row)
        if 'accuracy:' in aux:
            #            / remove '\x08'     / separate num / remove space
            acc = row[-1].replace('\x08', '').split(':')[-1].strip()
            all_accuracy.append(float(acc))
            #             / separate num / remove space
            loss = row[-2].split(':')[-1].strip()
            all_loss.append(float(loss))

    return np.asarray(all_accuracy), np.asarray(all_loss)


# after the 1st epoch, things do not change that much. Maybe we could use only
# the first one...

epochs = split_epochs(filename=FNAME_TIRAMISU)
epoch_tiramisu = epochs[0]
accuracy_tiramisu, loss_tiramisu = epoch_indicators(epoch_tiramisu)
total_time_tiramisu = float(epoch_tiramisu[-1][1].split('s')[0])
time_tiramisu = np.linspace(start=0, stop=total_time_tiramisu, num=len(accuracy_tiramisu))

epochs = split_epochs(filename=FNAME_UNET)
epoch_unet = epochs[0]
accuracy_unet, loss_unet = epoch_indicators(epoch_unet)
total_time_unet = float(epoch_unet[-1][1].split('s')[0])
time_unet = np.linspace(start=0, stop=total_time_unet, num=len(accuracy_unet))


# plotting the indicators
COLOR_TIRAMISU = '#3e4989'
COLOR_UNET = '#6ece58'
fig, ax = plt.subplots(nrows=2)
ax[0].plot(time_tiramisu, accuracy_tiramisu, c=COLOR_TIRAMISU, linewidth=3)
ax[0].plot(time_unet, accuracy_unet, c=COLOR_UNET, linewidth=3)
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Accuracy')
ax[0].legend(['Tiramisu', 'U-net'], shadow=True)

ax[1].plot(time_tiramisu, loss_tiramisu, c=COLOR_TIRAMISU, linewidth=3)
ax[1].plot(time_unet, loss_unet, c=COLOR_UNET, linewidth=3)
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Loss')
ax[1].legend(['Tiramisu', 'U-net'], shadow=True)


# now, let's get how much each epoch took.
all_time_tiramisu, all_time_unet = [], []
epochs = split_epochs(filename=FNAME_TIRAMISU)
for epoch in epochs:
    all_time_tiramisu.append(float(epoch[-1][1].split('s')[0]))

epochs = split_epochs(filename=FNAME_UNET)
for epoch in epochs:
    all_time_unet.append(float(epoch[-1][1].split('s')[0]))

all_time_tiramisu = np.asarray(all_time_tiramisu)
all_time_unet = np.asarray(all_time_unet)

print(all_time_tiramisu.mean(), all_time_tiramisu.std())
print(all_time_unet.mean(), all_time_unet.std())
