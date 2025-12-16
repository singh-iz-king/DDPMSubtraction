import torch
import os
from Data.mnist_subtraction_dataset import MnistSubtractionDataset
from Model.u_net import UNet
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from Train.forward_process import forward_process
from Train.forward_process import get_ddpm_schedule
import torch.nn.functional as F
from Train.checkpoint import save_checkpoint
from Train.checkpoint import load_checkpoint
from Sampling.sample import sample


device = torch.device("cuda" if torch.cuda.is_available() else "mps")

transform = transforms.Compose([transforms.ToTensor()])

data = MnistSubtractionDataset(
    mnist_file_path="mnist_train.csv",
    subtraction_file_path="subtraction_mnist.csv",
    transform=transform,
)

data.subtraction = data.subtraction.iloc[1:100, :]

data_loader = DataLoader(data, batch_size=32, shuffle=True)

model = UNet()
model.to(device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

T = 1000
schedules = get_ddpm_schedule(T)

epochs = 500

CHECKPOINT_INTERVAL = 50
BEST_MODEL_PATH = "runs/run_001/checkpoints/best.pt"
LAST_CHECKPOINT_PATH = "runs/run_001/checkpoints/last.pt"
SAMPLE_PATH = "runs/run_001/samples"

fixed_batch = next(iter(data_loader))
fixed_im1, fixed_im2, _ = fixed_batch
fixed_im1, fixed_im2 = fixed_im1 / 255.0, fixed_im2 / 255.0
fixed_im1, fixed_im2 = fixed_im1 * 2.0 - 1.0, fixed_im2 * 2.0 - 1.0
fixed_conditions = torch.cat([fixed_im1, fixed_im2], dim=1).to(device=device)

best_loss = float("inf")

start_epoch = load_checkpoint(model, optimizer, LAST_CHECKPOINT_PATH)

for epoch in range(start_epoch, epochs):

    total_loss = 0

    print(f"Starting Epoch: {epoch}", flush=True)

    for im1, im2, res in data_loader:

        im1, im2, res = im1.to(device), im2.to(device), res.to(device)

        im1, im2, res = im1 / 255.0, im2 / 255.0, res / 255.0

        im1, im2, res = im1 * 2.0 - 1.0, im2 * 2.0 - 1.0, res * 2.0 - 1.0

        optimizer.zero_grad()

        t = torch.randint(1, T + 1, size=(res.shape[0],), device=device)

        noisy_res, epsilon = forward_process(res, t, schedules)

        conditioned_input = torch.cat([noisy_res, im1, im2], dim=1)

        predicted_epsilon = model(conditioned_input, t)

        loss = F.mse_loss(epsilon, predicted_epsilon)

        loss.backward()

        total_loss += loss.item()

        optimizer.step()

    avg_loss = total_loss / len(data_loader)

    if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
        save_checkpoint(
            model, optimizer, epoch, avg_loss, filename=LAST_CHECKPOINT_PATH
        )

        sample_filename = os.path.join(SAMPLE_PATH, f"epoch_{epoch+1:04d}.png")

        sample(conditions=fixed_conditions, model=model, filename=sample_filename)

    if avg_loss < best_loss:
        save_checkpoint(model, optimizer, epoch, avg_loss, filename=BEST_MODEL_PATH)

    print(f"Epoch: {epoch}, Average Loss : {avg_loss}", flush=True)
