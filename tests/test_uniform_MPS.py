import pytest
from src.uniform_MPS import UniformMPS
from src.cost import transfer_blocks

@pytest.fixture
def random_A():
    return UniformMPS.random(5, 2)

@pytest.fixture
def random_B():
    return UniformMPS.random(3, 2)
