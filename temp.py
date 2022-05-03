import pandas as pd
import matplotlib.pyplot as plt

file_path = '/localhome/studenter/renatask/Project/ML/saved_dataframes/DeepLabV3_all_epochs.csv'

file = pd.read_csv(file_path)

print(file)

val_loss_fig = file.loc[:, ['loss', 'val_loss']].plot(title='DeepLabV3+', xlabel='Epoch', ylabel='DICE Loss')
crossentropy_fig = file.loc[:, ['categorical_crossentropy', 'val_categorical_crossentropy']].plot(title='DeepLabV3+', xlabel='Epoch', ylabel='Categorical Cross-Entropy')
# dice_fig = file.loc[:, ['dice_loss', 'val_dice_loss']].plot()
val_acc_fig = file.loc[:, ['acc', 'val_acc']].plot(title='DeepLabV3+', xlabel='Epoch', ylabel='Accuracy')
loss_fig = file.loc[:, ['loss', 'val_loss', 'categorical_crossentropy', 'val_categorical_crossentropy']].plot(title='DeepLabV3+', xlabel='Epoch', ylabel='Loss')
plt.show()