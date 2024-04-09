import torch
import torch.nn as nn

def DeriviateSigmoid(x):
    x = nn.Sigmoid()(x)
    return x*(1-x)

class RoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        x = DeriviateSigmoid(input - torch.round(input) - 0.5)
        ctx.save_for_backward(x)
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        deriviation = ctx.saved_tensors[0]
        # print(deriviation)
        return grad_output*deriviation

# 将自定义激活函数封装为 Module
class RoundActivation(nn.Module):
    def forward(self, input):
        return RoundFunction.apply(input)
    
class StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        deriviation = DeriviateSigmoid(input)
        ctx.save_for_backward(deriviation)
        # output = torch.where(input < 0.0, torch.tensor(0.0), torch.tensor(1.0))
        return torch.where(input < 0.0, torch.tensor(0.0).to(input.device), torch.tensor(1.0).to(input.device))

    @staticmethod
    def backward(ctx, grad_output):
        deriviation = ctx.saved_tensors[0]
        return grad_output*deriviation

# 将自定义激活函数封装为 Module
class StepActivation(nn.Module):
    def forward(self, input):
        return StepFunction.apply(input)