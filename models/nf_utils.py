from torch import Tensor, sigmoid, log, clamp, log, log1p
from torch.nn import Parameter
from torch.nn.functional import softplus

from nflows.transforms.base import InverseTransform, Transform
from nflows.utils.torchutils import sum_except_batch  


class Sigmoid(Transform):
    def __init__(self, temperature=1, eps=1e-6, learn_temperature=False):
        super().__init__()
        self.eps = eps
        if learn_temperature:
            self.temperature = Parameter(Tensor([temperature]))
        else:
            self.temperature = Tensor([temperature])
            
        self.temperature = self.temperature.cuda()

    def forward(self, inputs, context=None):
        inputs = self.temperature * inputs
        outputs = sigmoid(inputs)
        logabsdet = sum_except_batch(
            log(self.temperature) - softplus(-inputs) - softplus(inputs)
        )
        return outputs, logabsdet

    def inverse(self, inputs, context=None):

        inputs = clamp(inputs, self.eps, 1 - self.eps)

        outputs = (1 / self.temperature) * (log(inputs) - log1p(-inputs))
        logabsdet = -sum_except_batch(
            log(self.temperature)
            - softplus(-self.temperature * outputs)
            - softplus(self.temperature * outputs)
        )
        return outputs, logabsdet
    
class Logit(InverseTransform):
    def __init__(self, temperature=1, eps=1e-6):
        super().__init__(Sigmoid(temperature=temperature, eps=eps))