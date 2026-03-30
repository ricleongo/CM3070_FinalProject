import pytest
import inspect

import tensorflow as tf
from tensorflow.keras import layers

from unittest.mock import patch

from back_end.src.app.ml_models.mdgcn.inductive.layer import InductiveLayer

@pytest.fixture
def layer():
    in_dim = 3
    out_dim = 2
    K = 1
    return InductiveLayer(in_dim, out_dim, K)

def test_InductiveMDGCNLayer_extend_keras_layer():
    # Assert
    assert issubclass(InductiveLayer, layers.Layer)

def test_init_has_expected_parameters():
    # Arrange
    function_signature = inspect.signature(InductiveLayer.__init__)
    function_params = function_signature.parameters

    # Assert
    assert 'self' in function_params

    assert 'in_dim' in function_params
    assert 'out_dim' in function_params
    assert 'K' in function_params

def test_init_params_values_valid():
    # Arrange
    in_dim = 1
    out_dim = 2
    K = 4

    # Act
    class_layer = InductiveLayer(in_dim, out_dim, K)

    # Assert
    assert class_layer.in_dim == in_dim
    assert class_layer.out_dim == out_dim
    assert class_layer.K == K

def test_init_call_internal_functions(layer):
    # Arrange
    in_dim = 3
    out_dim = 10
    K = 2

    with patch.object(layer, "_set_embeddings", wraps=layer._set_embeddings) as spy_embeddings, \
         patch.object(layer, "_set_alpha", wraps=layer._set_alpha) as spy_alpha:
    
        # Act
        layer.__init__(in_dim, out_dim, K)
                
        # Assert
        spy_embeddings.assert_called()
        spy_alpha.assert_called()

def test_set_kernels_has_expected_parameters():
    # Arrange
    function_signature = inspect.signature(InductiveLayer._set_kernels)
    function_params = function_signature.parameters

    # Assert
    assert 'self' in function_params

def test_kernels_is_valid(layer):
    # Act
    layer.K = 10
    layer._set_kernels()

    # Assert
    assert layer.kernels is not None
    assert len(layer.kernels)  == layer.K + 1

    for kernel in layer.kernels:
        assert isinstance(kernel, layers.Dense)

def test_set_embeddings_has_expected_parameters():
    # Arrange
    function_signature = inspect.signature(InductiveLayer._set_embeddings)
    function_params = function_signature.parameters

    # Assert
    assert 'self' in function_params

def test_embeddings_is_valid(layer):
    # Act
    layer._set_embeddings()

    # Assert
    assert layer.embedding_layer is not None
    assert isinstance(layer.embedding_layer, layers.Dense)

def test_set_alpha_has_expected_parameters():
    # Arrange
    function_signature = inspect.signature(InductiveLayer._set_alpha)
    function_params = function_signature.parameters

    # Assert
    assert 'self' in function_params

def test_alpha_is_valid(layer):
    # Act
    layer._set_alpha()

    # Assert
    assert layer.alpha is not None
    assert isinstance(layer.alpha, tf.Variable)

def test_build_has_expected_parameters():
    # Arrange
    function_signature = inspect.signature(InductiveLayer.build)
    function_params = function_signature.parameters

    # Assert
    assert 'self' in function_params
    assert 'input_shape' in function_params

def test_build_invoque_internal_functions(layer):
    # Arrange
    input_shape = 3

    with patch.object(layer, "_set_kernels", wraps=layer._set_kernels) as spy_kernels:
        # Act
        layer.build(input_shape)
                
        # Assert
        spy_kernels.assert_called()

def test_get_output_has_expected_parameters():
    # Arrange
    function_signature = inspect.signature(InductiveLayer._get_output)
    function_params = function_signature.parameters

    # Assert
    assert 'self' in function_params
    assert 'node_features' in function_params
    assert 'adjacent_list' in function_params

def test_call_has_expected_parameters():
    # Arrange
    function_signature = inspect.signature(InductiveLayer.call)
    function_params = function_signature.parameters

    # Assert
    assert 'self' in function_params
    assert 'X' in function_params
    assert 'adjacent_list' in function_params

def test_call_invoque_internal_functions(layer):
    # Arrange
    layer.K = 2
    N = 3
    F = layer.in_dim    
    X = tf.random.normal((N, F))

    adjacent_list = [
        tf.eye(N) for _ in range(layer.K + 1)
    ]

    layer._set_kernels()

    with patch.object(layer, "_get_output", wraps=layer._get_output) as spy_get_output:
        # Act
        output = layer.call(X, adjacent_list)
                
        # Assert
        assert output.shape == (N, layer.out_dim)
        assert tf.reduce_min(output).numpy() >= 0.0
        spy_get_output.assert_called()

