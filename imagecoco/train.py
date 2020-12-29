import time
import matplotlib.pyplot as plt
import numpy as np
import pickle
from torch.utils.data import DataLoader
from IPython import display

from coco_dataset import COCOImageCaptionsDataset
from generator import Generator
from rewarder import Rewarder
from utils import save

#########################################################################################
#  Hyper-parameters
#########################################################################################
# Constants
vocab_size = 4839

# General parameters
SEQ_LENGTH = 32

# Model parameters
g_hidden_state_size = 768
r_hidden_state_size = 256
action_embed_dim = 32


# Generator training step parameters
generator_batch_size = 64
roll_num = 4

# Rewarder training step parameters
real_batch_size = 64
generated_batch_size = 64

# Training parameters
G_LEARNING_RATE = 5e-5
R_LEARNING_RATE = 0.01
G_CLIP_MAX_NORM = 40
R_CLIP_MAX_NORM = 40
R_MOMENTUM = 0.9
NUM_ITERS = 51
G_ITERS = 1
R_ITERS = 5
PRETRAIN_ITERS = 20
restore = False


#########################################################################################
#  Initialization and Pretraining
#########################################################################################

str_map = pickle.load(open("save/str_map.pkl", "rb"))

# Load models
if restore:
    # Replace this with the path to the model you want to restore
    generator.restore_model("/content/gdrive/My Drive/rl-project/checkpoints/generator_30_-2297416384512.0.pt")
    rewarder.restore_model("/content/gdrive/My Drive/rl-project/checkpoints/rewarder_30_-33264479027.2.pt")
else:
    generator = Generator(SEQ_LENGTH, str_map, G_CLIP_MAX_NORM)
    rewarder = Rewarder(
        SEQ_LENGTH,
        vocab_size,
        g_hidden_state_size,
        action_embed_dim,
        r_hidden_state_size,
        R_LEARNING_RATE,
        R_CLIP_MAX_NORM,
        R_MOMENTUM
    )

# Load training data
train_data = COCOImageCaptionsDataset("save/train_data.pkl")
train_dataloader = DataLoader(train_data, batch_size=real_batch_size, shuffle=True)

# Pretrain generator
print("Pretraining generator")
pretrain_losses = []
for it in range(PRETRAIN_ITERS):
    batch_data = next(iter(train_dataloader))
    loss = generator.pretrain_step(batch_data).data.cpu().numpy()
    print(loss)
    pretrain_losses.append(loss)


plt.plot(np.arange(len(pretrain_losses)), np.array(pretrain_losses))
plt.show()

fig, ax = plt.subplots(1,2,figsize=(14,7))
g_losses = []
r_losses = []

#########################################################################################
#  Main Training Loop
#########################################################################################


for it in range(NUM_ITERS):

    # TRAIN GENERATOR
    start = time.time()
    loss_sum = 0
    for g_it in range(G_ITERS):
        g_loss = generator.rl_train_step(
            rewarder, generator_batch_size
        )
        loss_sum += g_loss
    speed = time.time() - start
    g_losses.append(loss_sum / G_ITERS)
    save(generator.model, "/checkpoints/model_checkpoints/generator_" + str(it) + ".pt")
    print(
        "MaxentPolicy Gradient {} iteration, Speed:{:.3f}, Loss:{:.3f}".format(
            it, speed, g_loss
        )
    )


    # TRAIN REWARDER
    start = time.time()
    loss_sum = 0
    for r_it in range(R_ITERS):
        real_trajectories = next(iter(train_dataloader))
        r_loss = rewarder.train_step(real_trajectories[0], generator, generated_batch_size)
        loss_sum += r_loss
    speed = time.time() - start
    r_losses.append(loss_sum / R_ITERS)
    save(rewarder.model, "/checkpoints/model_checkpoints/rewarder_" + str(it) + ".pt")
    print(
        "Reward training {} iteration, Speed:{:.3f}, Loss:{:.3f}".format(
            it, speed, r_loss
        )
    )


    # Logging
    if it % 5 == 0 or it == NUM_ITERS - 1 or it == 1:
        # Save models
        torch.save(generator.model.state_dict(), f"/content/gdrive/My Drive/rl-project/checkpoints/generator_{it}_{g_losses[-1]}.pt")
        torch.save(rewarder.model.state_dict(), f"/content/gdrive/My Drive/rl-project/checkpoints/rewarder_{it}_{r_losses[-1]}.pt")

        # Generate samples
        generated_samples = generator.generate(generator_batch_size, 1, None, False, False, True)
        output_file = f"/content/gdrive/My Drive/rl-project/checkpoints/generator_sample_{it}.txt"
        with open(output_file, 'w+') as fout:
            for sentence in generated_samples[0]:
                buffer = ' '.join(sentence) + "\n"
                fout.write(buffer)

        # Plot loss
        display.clear_output(wait=True)
        fig, ax = plt.subplots(1,2,figsize=(14,7))
        ax[0].cla(); ax[0].plot(g_losses)
        ax[1].cla(); ax[1].plot(r_losses)
        display.display(plt.gcf())
        print(it, g_losses, r_losses)
