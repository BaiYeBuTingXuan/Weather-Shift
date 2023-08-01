import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F



class ConvCell(nn.Module):
    def __init__(self, in_channels=4, out_channels=512, kernel_size=1,  stride=1, leaky=0.2, dropout=0.4, activation='None'):
        super(ConvCell, self).__init__()
        self.leaky = leaky
        self.dropout = dropout
        self.out_channels = out_channels

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        # print(x.size())
        x = self.conv1(x)
        x = nn.BatchNorm1d(self.out_channels, affine=False)(x)
        x = nn.ReLU()(x)

        # print(x1.size())

        x1 = self.conv2(x)
        x1 = nn.BatchNorm1d(self.out_channels, affine=False)(x1)
        x1 = nn.Tanh()(x1)
        x = x+x1

        x = nn.Dropout(self.dropout)(x)
        return x


class SelfAttentionCell(nn.Module):
    def __init__(self, in_channels=4, n_dim=512):
        super(SelfAttentionCell, self).__init__()

        self.n_dim = n_dim

        self.k_cnn = ConvCell(in_channels=in_channels, out_channels=n_dim, kernel_size=1,  stride=1, leaky=0.2, dropout=0.4)
        self.q_cnn = ConvCell(in_channels=in_channels, out_channels=n_dim, kernel_size=1,  stride=1, leaky=0.2, dropout=0.4)
        self.v_cnn = ConvCell(in_channels=in_channels, out_channels=n_dim, kernel_size=1,  stride=1, leaky=0.2, dropout=0.4)
        self.output = ConvCell(in_channels=n_dim, out_channels=in_channels, kernel_size=1,  stride=1, leaky=0.2, dropout=0.4)


    def forward(self, x):
        k = self.k_cnn(x)
        q = self.q_cnn(x)
        v = self.v_cnn(x)

        a = torch.matmul(k,q.transpose(1,2))
        a = nn.Softmax(dim=2)(a)
        x = torch.matmul(a,v)
        x = self.output(x)
        return x



class Generator(nn.Module):
    def __init__(self, point_dim=4, n_points=512):
        super(Generator, self).__init__()
        self.attention1 = SelfAttentionCell(in_channels=point_dim, n_dim=512)
        self.attention2 = SelfAttentionCell(in_channels=point_dim+1, n_dim=512)
        self.output = ConvCell(in_channels=point_dim+1, out_channels=point_dim, kernel_size=1,  stride=1, leaky=0.2, dropout=0.4)

    def forward(self, x, z):
        x = x + self.attention1(x)

        batch, _, points = x.size()
        z = z.view(batch, 1, 1).expand(batch, 1, points)

        x = torch.concat((x,z),dim=1)

        x = x + self.attention2(x)

        x = self.output(x)
        return x
    

class Discriminator(nn.Module):
    def __init__(self, point_dim=4, n_points=512):
        super(Discriminator, self).__init__()
        self.attention1 = SelfAttentionCell(in_channels=point_dim, n_dim=512)
        self.attention2 = SelfAttentionCell(in_channels=point_dim+1, n_dim=512)
        self.output = ConvCell(in_channels=point_dim+1, out_channels=1, kernel_size=1,  stride=1, leaky=0.2, dropout=0.4)

    def forward(self, x, z):
        x = x + self.attention1(x)

        batch, _, points = x.size()
        z = z.view(batch, 1, 1).expand(batch, 1, points)

        x = torch.concat((x,z),dim=1)

        x = x + self.attention2(x)

        x = self.output(x)
        return x
    

if __name__ == '__main__':
    x = torch.rand(32,4,1024)
    z = torch.rand(32,1)
    conv = Generator(point_dim=4, n_points=512)
    x = conv(x,z)
    print(x.size())