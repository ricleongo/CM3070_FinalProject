import pytest
import inspect

import tensorflow as tf

from unittest.mock import patch

from src.app.supervised_mdgcn_model import SupervisedMdgcnModel
from src.app.transductive_mdgcn_layer import TransductiveMDGCNLayer

@pytest.fixture
def layer():
    num_nodes = 5
    in_dim = 1
    hidden_dim = 2
    K = 4

    return SupervisedMdgcnModel(num_nodes, in_dim, hidden_dim, K)

def test_SupervisedMdgcnModel_extend_keras_model():
    # Assert
    assert issubclass(SupervisedMdgcnModel, tf.keras.Model)

def test_init_has_expected_parameters():
    # Arrange
    function_signature = inspect.signature(SupervisedMdgcnModel.__init__)
    function_params = function_signature.parameters

    # Assert
    assert 'self' in function_params

    assert 'num_nodes' in function_params
    assert 'in_dim' in function_params
    assert 'hidden_dim' in function_params
    assert 'K' in function_params

def test_initial_values():
    # Arrange
    num_nodes = 5
    in_dim = 1
    hidden_dim = 2
    K = 4

    # Act
    class_layer = SupervisedMdgcnModel(num_nodes, in_dim, hidden_dim, K)

    # Assert
    assert class_layer.num_nodes == num_nodes
    assert class_layer.in_dim == in_dim
    assert class_layer.hidden_dim == hidden_dim
    assert class_layer.K == K
    
    assert class_layer.classifier_layer is not None
    assert isinstance(class_layer.classifier_layer, tf.keras.layers.Dense)

def test_set_distance_layer_has_expected_parameters():
    # Arrange
    function_signature = inspect.signature(SupervisedMdgcnModel._set_distance_layer)
    function_params = function_signature.parameters

    # Assert
    assert 'self' in function_params

def test_distance_layer_is_valid(layer):
    # Act
    # layer.K = 10
    layer._set_distance_layer()

    # Assert
    assert layer.distance_layer is not None
    assert isinstance(layer.distance_layer, TransductiveMDGCNLayer)

def test_build_has_expected_parameters():
    # Arrange
    function_signature = inspect.signature(SupervisedMdgcnModel.build)
    function_params = function_signature.parameters

    # Assert
    assert 'self' in function_params

def test_build_invoque_internal_functions(layer):
    # Arrange
    with patch.object(layer, "_set_distance_layer", wraps=layer._set_distance_layer) as spy_set_distance_layer:
        # Act
        layer.build()
                
        # Assert
        spy_set_distance_layer.assert_called()


def test_call_has_expected_parameters():
    # Arrange
    function_signature = inspect.signature(SupervisedMdgcnModel.call)
    function_params = function_signature.parameters

    # Assert
    assert 'self' in function_params
    assert 'node_features' in function_params
    assert 'adjacent_dist_list' in function_params
