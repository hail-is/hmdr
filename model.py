import gc
import shutil
import math
import random
import os
import sys

import hail as hl
from hail.utils import new_temp_file
from hailtop.utils import grouped

import numpy as np

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms

import torch
import torch.nn as nn

from tqdm.auto import tqdm

import matplotlib.pyplot as plt

tmp = '/tmp/hailtmp'
os.makedirs(tmp, exist_ok=True)
hl.init(spark_conf={'spark.driver.memory': '32g'}, tmp_dir=tmp, local_tmpdir=tmp)

N = 60000                     # number of samples
M = 10000                     # number of variants
K = 3                         # size of intermediate phenotypes

nz = 10                       # The input size of the GAN
ngf = 64                      # The size of feature maps in the GAN's generator
nc = 3                        # The number of channels in the images.
image_size = 64               # We'll use 64x64-pixel square images
BATCH_SIZE = min(N//2, 8192)  # autoencoder training batch size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def normalize_cols(X):
    centered = X - torch.mean(X, 0)
    return torch.div(centered, torch.std(centered, 0))


def normalize_rows(X):
    return normalize_cols(X.T).T


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        return self.main(x)


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # 3, 64, 64
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 8, 64, 32
            nn.Conv2d(8, 12, 3, stride=2, padding=1),
            nn.BatchNorm2d(12),
            nn.LeakyReLU(0.2, inplace=True),
            # 12, 16, 16
            nn.Flatten(),
            nn.Linear(12 * 16 * 16, 128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128
            nn.Linear(128, 16),
            nn.LeakyReLU(0.2, inplace=True),
            # 16
            nn.Linear(16, 3),
            # 3
        )

        self.decoder = nn.Sequential(
            # 3
            nn.Linear(3, 16),
            nn.LeakyReLU(0.2, inplace=True),
            # 16
            nn.Linear(16, 128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128
            nn.Linear(128, 12 * 16 * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # 3 * 1024
            nn.Unflatten(1, (12, 16, 16)),
            nn.ConvTranspose2d(12, 8, 3, stride=2, padding=1, output_padding=1), # N, 16, 14, 14 (N,16,13,13 without output_padding)
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1), # N, 1, 28, 28  (N,1,27,27)
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return (encoded, decoded)


def generate_genotypes_and_simulated_latent_phenotypes(scratch):
    mt = hl.balding_nichols_model(1,
                                  n_samples=N,
                                  n_variants=M * 2,
                                  af_dist=hl.rand_unif(0.05, 0.95))
    mt = mt.filter_rows((hl.agg.call_stats(mt.GT, mt.alleles).AF[0] > 0.0) &
                        (hl.agg.call_stats(mt.GT, mt.alleles).AF[0] < 1.0))
    mt = mt.head(n_rows=M)
    mt = mt.checkpoint(scratch + '/g1.mt')
    count = mt.count()
    assert count == (M, N), count
    mt = hl.experimental.ldscsim.simulate_phenotypes(mt, mt.GT, [0.5 for phenotype_index in range(K)])
    mt = mt.checkpoint(scratch + '/g2.mt')
    assert mt.count() == (M, N)
    return mt


def generate_observed_phenotypes(netG10, mt, scratch):
    observed_phenotypes_folder = scratch + '/fake_faces'
    os.makedirs(observed_phenotypes_folder + '/group1/', exist_ok=True)

    phenos = mt.y.collect()
    phenos = torch.tensor(phenos).reshape(N, K).to(device)

    # Recall: A sends 3-dimensional vectors to 10-dimensional vectors
    A = torch.randn(K, nz).to(device) / math.sqrt(K)

    phenos = phenos @ A
    phenos = phenos.reshape([N, nz, 1, 1])

    for batch_index, batch in enumerate(tqdm(grouped(BATCH_SIZE, phenos), desc='generate observed batch')):
        with torch.no_grad():
            images = netG10(batch).detach().cpu()
            images = images.reshape(batch.shape[0], nc, 64, 64)
            for i in range(images.shape[0]):
                image = images[i, :, :, :]
                image -= torch.min(image)
                image /= torch.max(image)
                image = torch.round(image * 255).type(torch.uint8)

                image_global_index = batch_index * BATCH_SIZE + i
                fname = observed_phenotypes_folder
                fname += '/group1/'
                fname += f'{image_global_index:05}.jpeg'

                torchvision.io.write_jpeg(image, fname)

    dataset = dset.ImageFolder(
        root=observed_phenotypes_folder,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=BATCH_SIZE,
                                       num_workers=2)


def mt_to_numpy(mt, scratch: str):
    bm = hl.linalg.BlockMatrix.from_entry_expr(hl.float64(mt.GT.n_alt_alleles()))
    bm.export_blocks(scratch + '/g.npy', binary=True)
    g = hl.linalg.BlockMatrix.rectangles_to_numpy(scratch + '/g.npy', binary=True)
    g = torch.tensor(g)
    g = normalize_rows(g)
    return g.type(torch.float32).to(device)


def trace_heritability(phenos, normalized_g, return_qt_q=False):
    N, K = phenos.shape
    M, _ = normalized_g.shape

    q, _ = torch.linalg.qr(phenos)
    q = normalize_cols(q)

    betahat = normalized_g @ q / (N - 1)
    tr_H = betahat.square().sum() - K * M / N
    if return_qt_q:
        return (q.T @ q / (N - 1), tr_H)
    return tr_H


def train_autoencoder(dataloader: torch.utils.data.DataLoader,

                      g: torch.Tensor,
                      num_epochs: int,
                      recon_loss_weight: float = 1.0):
    model = Autoencoder().to(device)
    pixel_difference = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.002, betas=(0.5, 0.999))

    for _ in tqdm(range(num_epochs), desc='training epoch'):
        for batch_index, (img, _) in enumerate(dataloader):
            img = img.to(device)
            latent, recon = model(img)

            sample_start = batch_index * BATCH_SIZE
            sample_end = sample_start + BATCH_SIZE

            tr_H_est = trace_heritability(latent, g[:, sample_start:sample_end])

            recon_loss = pixel_difference(recon, img)

            loss = recon_loss_weight * recon_loss - tr_H_est

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def variance_explained(A: torch.Tensor, B: torch.Tensor):
    A_q, _ = torch.linalg.qr(A)
    B_q, _ = torch.linalg.qr(B)
    return torch.square(torch.norm(A_q.T @ B_q, 'fro')) / K


def tr_H_variance_after_linear_transformation(A: torch.Tensor, g: torch.Tensor, draws=100):
    return np.std([
        trace_heritability(
            A @ torch.tensor(
                np.random.multivariate_normal(
                    np.array([0] * K * K),
                    np.identity(K * K)).reshape(K, K)).type(torch.float32).to(device),
            g).item()
        for _ in range(draws)])


def latent_phenotypes_from_model_and_observed(model: Autoencoder, dataloader: torch.utils.data.DataLoader):
    return torch.cat(tuple([
        latent.reshape(1, K)
        for (batch, _) in tqdm(dataloader, desc='generate latent phenos')
        for latent in model.encoder(batch.to(device))])).to(device)


def iteration(seed, log, num_epochs: int, recon_loss_weight: float = 0.0):
    random.seed(seed)
    torch.manual_seed(seed)
    hl.set_global_seed(seed)

    netG10 = Generator().to(device)
    netG10.load_state_dict(torch.load('netG10.pth', map_location=device))
    netG10.eval()

    scratch = new_temp_file()
    try:
        mt = generate_genotypes_and_simulated_latent_phenotypes(scratch)
        dataloader = generate_observed_phenotypes(netG10, mt, scratch)
        g = mt_to_numpy(mt, scratch)
        simulated_latent_phenos = torch.tensor(mt.y.collect()).type(torch.float32).to(device)
        tr_H_sim = trace_heritability(simulated_latent_phenos, g)

        model = train_autoencoder(dataloader, g, num_epochs, recon_loss_weight)
        estimated_latent_phenos = latent_phenotypes_from_model_and_observed(model, dataloader)

        tr_H_est = trace_heritability(estimated_latent_phenos, g)
        var_exp = variance_explained(simulated_latent_phenos, estimated_latent_phenos)
        variance_tr_H_sim = tr_H_variance_after_linear_transformation(simulated_latent_phenos, g)
        variance_tr_H_est = tr_H_variance_after_linear_transformation(estimated_latent_phenos, g)

        vals = (seed, tr_H_sim, tr_H_est, var_exp, variance_tr_H_sim, variance_tr_H_est)
        log.write(','.join([str(float(x)) for x in vals]) + '\n')
        return vals
    finally:
        del netG10
        del mt
        del g
        del simulated_latent_phenos
        del model
        del estimated_latent_phenos

        shutil.rmtree(scratch)
        shutil.rmtree(tmp)
        os.makedirs(tmp, exist_ok=True)
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()


for num_epochs in [1, 5]:
    for recon_loss_weight in [1.0, 0.0]:
        logname = f'log-linear_tranform_variance3-{num_epochs}-{recon_loss_weight}'
        if hl.hadoop_exists(logname):
            start = len(open(logname).readlines())
        else:
            start = 0
        with open(logname, 'a') as log:
            for i in range(start, start + 50):
                iteration(i, log, num_epochs, recon_loss_weight)
                log.flush()
