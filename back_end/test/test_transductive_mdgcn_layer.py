import pytest
import inspect

import tensorflow as tf
from tensorflow.keras import layers

from unittest.mock import patch, MagicMock
from back_end.src.app.ml_models.mdgcn.transductive.layer import TransductiveLayer

@pytest.fixture
def layer():
    in_dim = 1
    out_dim = 2
    K = 4

    return TransductiveLayer(in_dim, out_dim, K)

def test_init_has_expected_parameters():
    # Arrange
    function_signature = inspect.signature(TransductiveLayer.__init__)
    function_params = function_signature.parameters

    # Assert
    assert 'self' in function_params
    assert 'in_dim' in function_params
    assert 'out_dim' in function_params
    assert 'K' in function_params
    assert 'embed_dim' in function_params

def test_init_params_validate():
    # Arrange
    in_dim = 1
    out_dim = 2
    K = 4

    # Act
    class_layer = TransductiveLayer(in_dim, out_dim, K)

    # Assert
    assert class_layer.in_dim == in_dim
    assert class_layer.out_dim == out_dim
    assert class_layer.K == K

def test_init_defines_variables(layer):
    # Assert
    assert layer.embed_initializer is not None

def test_TransductiveMDGCNLayer_extend_keras_layer():
    # Assert
    assert issubclass(TransductiveLayer, layers.Layer)

def test_set_kernels_has_expected_parameters():
    # Arrange
    function_signature = inspect.signature(TransductiveLayer._set_kernels)
    function_params = function_signature.parameters

    # Assert
    assert 'self' in function_params

def test_kernels_has_value(layer):
    # Act
    layer._set_kernels()

    # Assert
    assert layer.kernels  is not None

    for kernel in layer.kernels:
        assert isinstance(kernel, layers.Dense)    

def test_set_embeddings_has_expected_parameters():
    # Arrange
    function_signature = inspect.signature(TransductiveLayer._set_embeddings)
    function_params = function_signature.parameters

    # Assert
    assert 'self' in function_params

def test_embeddings_has_value(layer):
    # Act
    layer._set_embeddings()

    # Assert
    assert layer.embed1 is not None
    assert layer.embed2 is not None

    # assert isinstance(layer.embed1, tf.Variable)

def test_set_alpha_has_expected_parameters():
    # Arrange
    function_signature = inspect.signature(TransductiveLayer._set_alpha)
    function_params = function_signature.parameters

    # Assert
    assert 'self' in function_params

def test_alpha_has_value(layer):
    # Act
    layer._set_alpha()

    # Assert
    assert layer.alpha is not None

def test_build_has_expected_parameters():
    # Arrange
    function_signature = inspect.signature(TransductiveLayer.build)
    function_params = function_signature.parameters

    # Assert
    assert 'self' in function_params

def test_build_call_internal_functions(layer):
    # Arrange
    with patch.object(layer, "_set_kernels", wraps=layer._set_kernels) as spy_kernels, \
         patch.object(layer, "_set_embeddings", wraps=layer._set_embeddings) as spy_embeddings, \
         patch.object(layer, "_set_alpha", wraps=layer._set_alpha) as spy_alpha:
        
        # Act
        layer.build()
        
        # Assert
        spy_kernels.assert_called()
        spy_embeddings.assert_called()
        spy_alpha.assert_called()

def test_sparse_propagation_has_expected_parameters():
    # Arrange
    function_signature = inspect.signature(TransductiveLayer._sparse_propagation)
    function_params = function_signature.parameters

    # Assert
    assert 'self' in function_params
    assert 'adjacent_sparse' in function_params
    assert 'node_features_transformed' in function_params


def test_sparse_propagation_valid_value(layer):
    # Arrange
    mock_adj = MagicMock(name="mock_adjacent_sparse")
    mock_features = MagicMock(name="mock_node_features")
    mock_result = MagicMock(name="mock_result")

    # Act
    with patch("src.app.transductive_mdgcn_layer.tf.sparse.sparse_dense_matmul") as mock_matmul:
        mock_matmul.return_value = mock_result

        result = layer._sparse_propagation(mock_adj, mock_features)

        # Assert
        # Verify TensorFlow function was called correctly
        mock_matmul.assert_called_once_with(mock_adj, mock_features)

        # Verify returned value
        assert result == mock_result

def test_learned_propagation_has_expected_parameters():
    # Arrange
    function_signature = inspect.signature(TransductiveLayer._learned_propagation)
    function_params = function_signature.parameters

    # Assert
    assert 'self' in function_params
    assert 'node_features_transformed' in function_params


