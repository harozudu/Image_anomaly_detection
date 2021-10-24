import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import save_image
from gan.cuda import get_default_device, DeviceDataLoader


class DCGAN:
    """
    Class for our Deep Convolutional Generative Adversarial Network.
    It accepts 128x128 images.

    :param loss_fn: Loss function to use when training.
    :param ngpu: Number of GPU available. Set to 0 if no GPU.
    """
    def __init__(self, loss_fn=nn.BCELoss, ngpu=1):
        self.image_size = 128
        self.nc = 3  # Number of channels (3 for RGB)
        self.nz = 100  # Latent Vector size
        self.ngf = 128  # Generator feature map size
        self.ndf = 128  # Discriminator feature map size
        self.stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        self.device = get_default_device()
        self.ngpu = ngpu
        self.gen = Generator(self.ngpu, self.nz, self.ngf, self.nc)
        self.gen = self.gen.to(self.device)
        self.disc = Discriminator(self.ngpu, self.ngf, self.nc)
        self.disc = self.disc.to(self.device)

        # Enable multi-gpu if multiple GPU are available
        if (self.device.type == 'cuda') and (self.ngpu > 1):
            self.gen = nn.DataParallel(self.gen, list(range(self.ngpu)))
            self.disc = nn.DataParallel(self.disc, list(range(self.ngpu)))

        # Apply a random initialization with mean = 0 and stdev = 0.02.
        # As mentioned in the DCGAN paper https://arxiv.org/pdf/1511.06434.pdf.
        self.gen.apply(self._weights_init)
        self.disc.apply(self._weights_init)

        self.fixed_latent = torch.randn(64, self.nz, 1, 1, device=self.device)
        self.loss_fn = loss_fn()

    def _denorm(self, img_tensors):
        return img_tensors * self.stats[1][0] + self.stats[0][0]

    @staticmethod
    def _weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def _save_samples(
            self,
            index,
            latent_tensors,
            sample_dir="generated"):
        os.makedirs(sample_dir, exist_ok=True)
        fake_images = self.gen(latent_tensors)
        fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
        save_image(
            self._denorm(fake_images),
            os.path.join(sample_dir, fake_fname),
            nrow=8
        )
        print('Saving', fake_fname)

    def _train_disc(self, real_images, opt_d, batch_size):
        # Clear discriminator gradients
        opt_d.zero_grad()
        loss = self.loss_fn

        # Pass real images through discriminator
        real_preds = self.disc(real_images)
        real_targets = torch.ones(real_images.size(0), 1, device=self.device)
        real_loss = loss(real_preds, real_targets)
        real_score = torch.mean(real_preds).item()

        # Generate fake images
        latent = torch.randn(batch_size, self.nz, 1, 1, device=self.device)
        fake_images = self.gen(latent)

        # Pass fake images through discriminator
        fake_targets = torch.zeros(fake_images.size(0), 1, device=self.device)
        fake_preds = self.disc(fake_images)
        fake_loss = loss(fake_preds, fake_targets)
        fake_score = torch.mean(fake_preds).item()

        # Update discriminator weights
        train_loss = real_loss + fake_loss
        train_loss.backward()
        opt_d.step()
        return train_loss.item(), real_score, fake_score

    def _train_gen(self, opt_g, batch_size):
        # Clear generator gradients
        opt_g.zero_grad()
        loss = self.loss_fn

        # Generate fake images
        latent = torch.randn(batch_size, self.nz, 1, 1, device=self.device)
        fake_images = self.gen(latent)

        # Try to fool the discriminator
        preds = self.disc(fake_images)
        targets = torch.ones(batch_size, 1, device=self.device)
        gen_loss = loss(preds, targets)

        # Update generator weights
        gen_loss.backward()
        opt_g.step()

        return gen_loss.item()

    def fit(self,
            epochs,
            dataset_path: str,
            learn_rate=0.0002,
            batch_size=128,
            start_idx=1,
            save_samples=False):
        """
        Train the Generator and Discriminator.

        :param epochs: Number of epochs to train for.
        :param dataset_path: Path to the training dataset
        :param learn_rate: Learning rate of the model.
        :param batch_size: Size of the batch the Dataset will be
        subdivided in.
        :param start_idx: Index to start at when saving samples.
        :param save_samples: Save samples for visualizing progress.
        False by default.
        :return: Lists containing Generator Loss, Discriminator Loss, Scores on
        real images and Scores on fake images respectively.
        """
        torch.cuda.empty_cache()

        train_dataset = ImageFolder(
            dataset_path,
            transform=tt.Compose([
                tt.Resize(self.image_size),
                tt.CenterCrop(self.image_size),
                tt.ToTensor(),
                tt.Normalize(*self.stats)
            ])
        )
        train_dl = DataLoader(train_dataset, batch_size, shuffle=True,
                              num_workers=3, pin_memory=True)
        train_dl = DeviceDataLoader(train_dl, self.device)

        # Losses & scores
        losses_g = []
        losses_d = []
        real_scores = []
        fake_scores = []

        # Create optimizers
        opt_d = torch.optim.Adam(self.disc.parameters(), lr=learn_rate,
                                 betas=(0.5, 0.999))
        opt_g = torch.optim.Adam(self.gen.parameters(), lr=learn_rate,
                                 betas=(0.5, 0.999))

        for epoch in tqdm(range(epochs)):
            loss_g, loss_d, real_score, fake_score = (0, 0, 0, 0)
            for real_images, _ in tqdm(train_dl, leave=False):
                # Train discriminator
                loss_d, real_score, fake_score = self._train_disc(
                    real_images,
                    opt_d,
                    batch_size
                )
                # Train generator
                loss_g = self._train_gen(opt_g, batch_size)

            # Record losses & scores
            losses_g.append(loss_g)
            losses_d.append(loss_d)
            real_scores.append(real_score)
            fake_scores.append(fake_score)

            # Log losses & scores (last batch)
            print(
                "Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, "
                "real_score: {:.4f}, fake_score: {:.4f}".format(
                    epoch + 1, epochs, loss_g, loss_d, real_score, fake_score))

            # Save generated images
            if save_samples:
                self._save_samples(epoch + start_idx, self.fixed_latent)

        return losses_g, losses_d, real_scores, fake_scores


class Generator(nn.Module):
    """
    Generator for the DCGAN.

    :param ngpu: Number of GPU available.
    :parap nz: Number of Latent Vectors Z.
    :param ngf: Size of the feature map.
    :param nc: Number of channels in the images.
    """
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    """
    Class containing the Discriminator for the DCGAN.

    :param ngpu: Number of GPU available.
    :param ndf: Size of the feature map.
    :param nc: Number of channels in the images.
    """
    def __init__(self, ngpu, ndf, nc):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, stride=1, padding=0, bias=False),

            nn.Flatten(),
            nn.Sigmoid()
            # state size. 1
        )

    def forward(self, input):
        return self.main(input)


if __name__ == "__main__":
    gan = DCGAN()
    gan.fit(epochs=10, dataset_path='../data/dice/')
