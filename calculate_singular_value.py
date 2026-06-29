import torch.nn as nn
import torch
# from cl_utils.lya import calc_singularvalues

def seed_everything(seed):
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

all_seed = 55
seed_everything(all_seed)


class TinyConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=2,
            bias=False
        )

        self.fc1 = nn.Linear(
            in_features=4,
            out_features=2,
            bias=False
        )

        self.fc2 = nn.Linear(
            in_features=2,
            out_features=2,
            bias=False
        )

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.act(x)

        x = self.fc2(x)

        return x

net = TinyConvNet()
with torch.no_grad():
    net.conv1.weight[:] = torch.tensor([[[1,0], [0,1]]])
    net.fc1.weight[:] = torch.tensor([[0.3, 0.1, -0.1, 0], [1.1, 0.5, -0.6, -1]])
    net.fc2.weight[:] = torch.tensor([[1.0, 2.0], [3.0, 2.0]])

data = torch.tensor([[[[1,2,3],[4,5,6],[7,8,9]]]], dtype=torch.float32)

output = net(data)

print(output)

# print(net.fc2.weight)

# calculating jacobian
def calc_singularvalues(model, inputs):

    inputs = inputs.requires_grad_(True)
    torch.cuda.empty_cache()
    jacobian_batch = torch.autograd.functional.jacobian(model, inputs, create_graph=True)
    print(jacobian_batch.shape)

    flat_jacobian_batch = jacobian_batch.reshape(jacobian_batch.shape[0], jacobian_batch.shape[1], -1)
    print(flat_jacobian_batch.shape)
    svdvals = torch.linalg.svdvals(flat_jacobian_batch)

    print(svdvals)


    return svdvals.max()

ms = calc_singularvalues(net, data)
print(ms)

