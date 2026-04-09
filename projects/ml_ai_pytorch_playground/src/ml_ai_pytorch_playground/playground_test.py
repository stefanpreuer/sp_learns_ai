import pytest
import torch

from ml_ai_pytorch_playground import playground


@pytest.mark.manual
def test_fashion_mnist_model():
    model = playground.FashionMNISTModel()
    model.eval()
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    print(output)
    print(torch.argmax(output, dim=1))
    assert output.shape == (1, 10)
