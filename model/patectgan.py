
import torch
import torch.nn as nn
import numpy as np
from model.discriminator import Discriminator
from model.generator import Generator



class PATECTGAN:
    def __init__(self, embedding_dim, gen_dim, dis_dim, l2scale, batch_size, epochs, pack, loss, device=None):
        self.embedding_dim = embedding_dim
        self.gen_dim = gen_dim
        self.dis_dim = dis_dim
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.pack = pack
        self.loss = loss
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = Generator(self.embedding_dim, self.gen_dim, self.embedding_dim).to(self.device)
        self.discriminator = Discriminator(self.embedding_dim, self.dis_dim, self.loss, self.pack).to(self.device)
    
    def train(self, data_loader):
        optimizer_g = torch.optim.AdamW(self.generator.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=self.l2scale)
        optimizer_d = torch.optim.AdamW(self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))
        criterion = torch.nn.BCELoss() if self.loss == "cross_entropy" else self.w_loss

        self.discriminator_losses = []
        self.generator_losses = []

        for epoch in range(self.epochs):
            for real_data in data_loader:
                real_data = real_data[0].to(self.device)
                batch_size = real_data.size(0)

                # Train Discriminator
                optimizer_d.zero_grad()
                fake_data = self.generator(torch.randn(batch_size, self.embedding_dim).to(self.device)).detach()
                
                real_label = torch.ones((batch_size, 1), device=self.device)
                fake_label = torch.zeros((batch_size, 1), device=self.device)
                
                output_real = self.discriminator(real_data)
                output_fake = self.discriminator(fake_data)
                
                loss_d_real = criterion(output_real, real_label)
                loss_d_fake = criterion(output_fake, fake_label)
                
                gradient_pen = self.discriminator.gradient_penalty(real_data, fake_data, self.device)
                loss_d = loss_d_real + loss_d_fake + gradient_pen
                loss_d.backward()
                optimizer_d.step()

                # Train Generator
                optimizer_g.zero_grad()
                fake_data = self.generator(torch.randn(batch_size, self.embedding_dim).to(self.device))
                output_fake = self.discriminator(fake_data)
                
                # Feature Matching Loss
                real_features = torch.mean(real_data, dim=0)
                fake_features = torch.mean(fake_data, dim=0)
                feature_match_loss = torch.mean((real_features - fake_features) ** 2)
                
                loss_g = criterion(output_fake, real_label) + feature_match_loss
                loss_g.backward()
                optimizer_g.step()
 
                # Store losses for monitoring
                self.discriminator_losses.append(loss_d.item())
                self.generator_losses.append(loss_g.item())

            # Print losses for each epoch
            print(f"Epoch {epoch+1}/{self.epochs} \t Discriminator Loss: {loss_d.item()} \t Generator Loss: {loss_g.item()}")
    
    def generate(self, n):
        self.generator.eval()
        generated_data = []
        for _ in range(n // self.batch_size + 1):
            fake_data = self.generator(torch.randn(self.batch_size, self.embedding_dim).to(self.device)).detach().cpu().numpy()
            generated_data.append(fake_data)
        return np.vstack(generated_data)[:n]
    
    def w_loss(self, output, labels):
        vals = torch.cat([labels[None, :], output[None, :]], axis=1)
        ordered = vals[vals[:, 0].sort()[1]]
        data_list = torch.split(ordered, labels.shape[0] - int(labels.sum().item()))
        fake_score = data_list[0][:, 1]
        true_score = torch.cat(data_list[1:], axis=0)[:, 1]
        w_loss = -(torch.mean(true_score) - torch.mean(fake_score))
        return w_loss
