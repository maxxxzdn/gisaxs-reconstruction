from torch import Tensor, sigmoid, log, clamp, log, log1p
from torch.nn import Parameter
from torch.nn.functional import softplus

from nflows.transforms.base import InverseTransform, Transform
from nflows.utils.torchutils import sum_except_batch  


class Sigmoid(Transform):
    """
    Sigmoid transform (forwards and inverse).
    Input:
        temperature (float): temperature of the sigmoid
        eps (float): epsilon for numerical stability
        learn_temperature (bool): whether to learn temperature
    """
    def __init__(self, temperature=1, eps=1e-6, learn_temperature=False):
        super().__init__()
        self.eps = eps
        if learn_temperature:
            self.temperature = Parameter(Tensor([temperature]))
        else:
            self.temperature = Tensor([temperature])
            
        self.temperature = self.temperature.cuda()

    def forward(self, inputs, context=None):
        """
        Forward pass of the sigmoid transform.
        Input:
            inputs (torch.Tensor): input tensor
            context (torch.Tensor): context tensor
        Output:
            outputs (torch.Tensor): output tensor
            logabsdet (torch.Tensor): log absolute determinant of the Jacobian
        """
        inputs = self.temperature * inputs
        outputs = sigmoid(inputs)
        logabsdet = sum_except_batch(
            log(self.temperature) - softplus(-inputs) - softplus(inputs)
        )
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        """
        Inverse pass of the sigmoid transform.
        Input:
            inputs (torch.Tensor): input tensor
            context (torch.Tensor): context tensor
        Output:
            outputs (torch.Tensor): output tensor
            logabsdet (torch.Tensor): log absolute determinant of the Jacobian
        """
        inputs = clamp(inputs, self.eps, 1 - self.eps)

        outputs = (1 / self.temperature) * (log(inputs) - log1p(-inputs))
        logabsdet = -sum_except_batch(
            log(self.temperature)
            - softplus(-self.temperature * outputs)
            - softplus(self.temperature * outputs)
        )
        return outputs, logabsdet
    
class Logit(InverseTransform):
    """
    Logit transform (forwards and inverse).
    Input:
        temperature (float): temperature of the sigmoid
        eps (float): epsilon for numerical stability
        learn_temperature (bool): whether to learn temperature
    """
    def __init__(self, temperature=1, eps=1e-6):
        super().__init__(Sigmoid(temperature=temperature, eps=eps))