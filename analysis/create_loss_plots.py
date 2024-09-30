import numpy as np 
import matplotlib.pyplot as plt

train_loss_pth = "/media/selamg/DATA/beadsight/data/clip_vallosses_2024-06-01_13-08-36.npy"
val_loss_pth = "/media/selamg/DATA/beadsight/data/clip_vallosses_2024-06-03_21-17-21_frozen.npy"

train_losses = np.load(train_loss_pth, allow_pickle=True)
val_losses = np.load(val_loss_pth, allow_pickle=True)

print(train_losses.shape,val_losses.shape)

xs = np.arange(3500)

# vxs = np.arange(0,3050,10)
# val losses only calculated every 10th epoch

##fix the val losses:
# save_val_losses = []
# for i in range(val_losses.shape[0]):
#     if i % 10 == 0:
#         val_losses[i] = np.mean(val_losses[i])
#         save_val_losses.append(val_losses[i])
# np.save('fixed_val_losses.npy', save_val_losses)

plt.plot(xs,val_losses,label='frozen')
plt.plot(xs,train_losses,label='not_frozen')
plt.legend()
plt.show()