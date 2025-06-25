import numpy as np
import pytest
from src.utils.geometry import (
    preconditioning, IdentityProjector, GrassmanProjector, SteifelProjector,
    GradientDescent, Retraction
)
from src.uniform_MPS import UniformMPS
from src.cost import HilbertSchmidt

class DummyProjector(IdentityProjector):
    pass



def test_preconditioning_identity():
    G = np.eye(2)
    r = np.eye(2)
    out = preconditioning(G, r)
    assert out.shape == (2, 2)
    assert np.allclose(out, G @ np.linalg.inv(r + np.eye(2) * np.linalg.norm(G.reshape(-1, 2)) ** 2))

def test_identity_projector():
    A = np.ones((2, 2, 2))
    D = np.ones((2, 2, 2)) * 2
    proj = IdentityProjector()
    out = proj.project(A, D)
    assert np.allclose(out, D)

def test_grassman_projector_shape():
    A = np.random.rand(2, 2, 2)
    D = np.random.rand(2, 2, 2)
    proj = GrassmanProjector()
    out = proj.project(A, D)
    assert out.shape == D.shape

def test_grassman_projector():
    d, p = 5, 2
    A = UniformMPS.from_random(d, p)
    B = UniformMPS.from_random(d, p)

    P = GrassmanProjector()
    f = HilbertSchmidt(A, B, 5)

    D = f.derivative()
    G = P.project(B.tensor, D)

    assert np.allclose(np.conj(A.matrix.T) @ G.reshape(d*p, d), np.zeros((d*p, d*p)))

def test_steifel_projector_shape():
    A = np.random.rand(2, 2, 2)
    D = np.random.rand(2, 2, 2)
    proj = SteifelProjector()
    out = proj.project(A, D)
    assert out.shape == D.shape

def test_gradient_descent_update():
    A = np.ones((2, 2, 2))
    D = np.ones((2, 2, 2))
    r = np.eye(2)
    gd = GradientDescent(IdentityProjector(), preconditioning=False)
    out = gd.update(A, D, alpha=0.1, r=r)
    assert out.shape == A.shape

def test_retraction_update():
    A = np.ones((2, 2, 2))
    D = np.ones((2, 2, 2))
    r = np.eye(2)
    ret = Retraction(IdentityProjector(), preconditioning=False)
    out = ret.update(A, D, alpha=0.1, r=r)
    assert out.shape == (4, 2) or out.shape == A.shape  # Accepts both possible shapes
