import torch
import torch.nn as nn
import pdb 
import matplotlib.pyplot as plt

def bent_identity(input):
    return (torch.sqrt(torch.pow(input, 2) + 1.) - 1. )/2. + input

def softplus(input):
    return torch.log(1 + torch.exp(input))

def mish(input):
    return input * torch.tanh(softplus(input))

def swish(input): 
    return input * torch.sigmoid(input)

# Define the retanh function combinging ReLU and tanh
# But to avoid the non-differentiable point at 0, we use the softplus function
def retanh(input):
    '''
    Applies the rectified tanh function element-wise:

        ReTanh(x) = max(0, tanh(x))
    '''
    
    #eturn torch.maximum(torch.tensor(0.0), torch.tanh(input))
    return input.relu().tanh()



# create a class wrapper from PyTorch nn.Module, so
# the function now can be easily used in models
class ReTanh(nn.Module):
    '''
    Applies the rectified tanh function element-wise:

        ReTanh(x) = max(0, tanh(x))

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input


    Examples:
        >>> m = ReTanh()
        >>> input = torch.randn(2)
        >>> output = m(input)

    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return retanh(input) 


if __name__ == '__main__':
    m = ReTanh()
    input = torch.tensor([-1.2, 0, 5, -5])
    print("Input", input)
    output = m(input)
    print("Output with ReTanh",output)
    t = nn.Tanh()
    output = t(input)
    print("Output with Tanh", output)

    # Generate 100 points between -1 and 1
    num_points = 200
    tensor = torch.linspace(-2, 2, num_points)

    # Calculate the custom activation function
    output_tensor = m(tensor)
    ref_tensor = t(tensor)

    # Plot the tensor and the output of the custom activation function
    plt.plot(tensor.numpy(), output_tensor.numpy(), label='ReLU and then Tanh')
    plt.plot(tensor.numpy(), ref_tensor.numpy(), linestyle='--', label='nn.Tanh')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.title('Custom Activation Function ReTanh')
    plt.legend()
    plt.grid(True)
    plt.savefig("ReTanh.png")  