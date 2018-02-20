import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from utils import *

class fc_model(nn.Module):
    def __init__(self, num_channels, woh):
        super(fc_model, self).__init__()
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(num_channels*woh*woh, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3_1 = nn.Linear(512,200)
        self.fc3_2 = nn.Linear(512,200)

        self.fc4 = nn.Linear(200, 512)
        self.fc5 = nn.Linear(512, 1024)
        self.fc6 = nn.Linear(1024, num_channels*woh*woh)

    def encode(self, x):

        h1 = self.relu(self.fc1(x.view(-1, num_flat_features(x))))
        h2 = self.relu(self.fc2(h1))
        h3_1 = self.fc3_1(h2) # mu
        h3_2 = self.fc3_2(h2) # log_var

        return h3_1, h3_2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):

        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        h6 = self.sigmoid(self.fc6(h5))

        return h6

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class basic_model(nn.Module):
    def __init__(self):
        super(basic_model, self).__init__()
        
        self.conv1 = nn.Conv2d(3,    64, kernel_size=3, stride=1, padding=1) # 64,32,32
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,  128, kernel_size=3, stride=2, padding=1) # 128,16,16
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # 256,8,8
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1) # 512,4,4
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1) # 512,2,2
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6_1 = nn.Conv2d(512, 512, kernel_size=2, stride=1           ) # 512,1,1
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=2, stride=1           ) # 512,1,1
        
        self.fsconv1 = nn.ConvTranspose2d(512,512, kernel_size = 3, stride=2, padding=1, output_padding=1) # 512,2,2
        self.bn1_d = nn.BatchNorm2d(512)
        self.fsconv2 = nn.ConvTranspose2d(512,512, kernel_size = 3, stride=2, padding=1, output_padding=1) # 512,4,4
        self.bn2_d = nn.BatchNorm2d(512)
        self.fsconv3 = nn.ConvTranspose2d(512,256, kernel_size = 3, stride=2, padding=1, output_padding=1) # 256,8,8
        self.bn3_d = nn.BatchNorm2d(256)
        self.fsconv4 = nn.ConvTranspose2d(256,128, kernel_size = 3, stride=2, padding=1, output_padding=1) # 128,16,16
        self.bn4_d = nn.BatchNorm2d(128)
        self.fsconv5 = nn.ConvTranspose2d(128,64,  kernel_size = 3, stride=2, padding=1, output_padding=1) # 64,32,32
        self.bn5_d = nn.BatchNorm2d(64)
        self.fsconv6 = nn.ConvTranspose2d(64,3,    kernel_size = 3, stride=1, padding=1) # 3,32,32
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.bn1(self.conv1(x)))
        h2 = self.relu(self.bn2(self.conv2(h1)))
        h3 = self.relu(self.bn3(self.conv3(h2)))
        h4 = self.relu(self.bn4(self.conv4(h3)))
        h5 = self.relu(self.bn5(self.conv5(h4)))
        h6_1 = self.conv6_1(h5) # mu
        h6_2 = self.conv6_2(h5) # logvar
        
        return h6_1.view(-1,num_flat_features(h6_1)), h6_2.view(-1,num_flat_features(h6_2))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z_t = z.view(-1, num_flat_features(z) ,1, 1)

        d1 = self.relu(self.bn1_d(self.fsconv1(z_t)))
        d2 = self.relu(self.bn2_d(self.fsconv2(d1 )))
        d3 = self.relu(self.bn3_d(self.fsconv3(d2 )))
        d4 = self.relu(self.bn4_d(self.fsconv4(d3 )))
        d5 = self.relu(self.bn5_d(self.fsconv5(d4 )))
        d6 = self.sigmoid(self.fsconv6(d5))
        
        return d6

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


        
