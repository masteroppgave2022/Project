import pandas as pd
import matplotlib.pyplot as plt

file_path = '/localhome/studenter/mikaellv/Project/ML/saved_dataframes/first_set.csv'

file = pd.read_csv(file_path)

print(file)

val_loss_fig = file.loc[:, ['loss', 'val_loss']].plot()
crossentropy_fig = file.loc[:, ['categorical_crossentropy', 'val_categorical_crossentropy']].plot()
dice_fig = file.loc[:, ['dice_loss', 'val_dice_loss']].plot()
val_acc_fig = file.loc[:, ['acc', 'val_acc']].plot()
loss_fig = file.loc[:, ['dice_loss', 'val_dice_loss', 'categorical_crossentropy', 'val_categorical_crossentropy']].plot()
plt.show()