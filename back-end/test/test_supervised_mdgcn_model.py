import pytest
import inspect

import tensorflow as tf

from unittest.mock import patch

from src.app.mdgcn.transductive.model import SupervisedTransductiveModel
from src.app.mdgcn.transductive.layer import TransductiveLayer

@pytest.fixture
def model():
    num_nodes = 5
    in_dim = 1
    hidden_dim = 2
    K = 4

    return SupervisedTransductiveModel(num_nodes, in_dim, hidden_dim, K)

def test_SupervisedMdgcnModel_extend_keras_model():
    # Assert
    assert issubclass(SupervisedTransductiveModel, tf.keras.Model)

def test_init_has_expected_parameters():
    # Arrange
    function_signature = inspect.signature(SupervisedTransductiveModel.__init__)
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
    class_layer = SupervisedTransductiveModel(num_nodes, in_dim, hidden_dim, K)

    # Assert
    assert class_layer.num_nodes == num_nodes
    assert class_layer.in_dim == in_dim
    assert class_layer.hidden_dim == hidden_dim
    assert class_layer.K == K
    
    assert class_layer.classifier_layer is not None
    assert isinstance(class_layer.classifier_layer, tf.keras.layers.Dense)

def test_set_distance_layer_has_expected_parameters():
    # Arrange
    function_signature = inspect.signature(SupervisedTransductiveModel._set_distance_layer)
    function_params = function_signature.parameters

    # Assert
    assert 'self' in function_params

def test_distance_layer_is_valid(model):
    # Act
    model._set_distance_layer()

    # Assert
    assert model.distance_layer is not None
    assert isinstance(model.distance_layer, TransductiveLayer)

def test_build_has_expected_parameters():
    # Arrange
    function_signature = inspect.signature(SupervisedTransductiveModel.build)
    function_params = function_signature.parameters

    # Assert
    assert 'self' in function_params

def test_build_invoque_internal_functions(model):
    # Arrange
    with patch.object(model, "_set_distance_layer", wraps=model._set_distance_layer) as spy_set_distance_layer:
        # Act
        model.build()
                
        # Assert
        spy_set_distance_layer.assert_called()


def test_call_has_expected_parameters():
    # Arrange
    function_signature = inspect.signature(SupervisedTransductiveModel.call)
    function_params = function_signature.parameters

    # Assert
    assert 'self' in function_params
    assert 'node_features' in function_params
    assert 'adjacent_dist_list' in function_params

def test_compute_loss_has_expected_parameters():
    # Arrange
    function_signature = inspect.signature(SupervisedTransductiveModel._compute_loss)
    function_params = function_signature.parameters

    # Assert
    assert 'self' in function_params
    assert 'labels' in function_params
    assert 'predicted' in function_params
    assert 'mask' in function_params

def test_compute_loss_return_value_is_valid(model):
    # Arrange
    labels = tf.constant([[1.0], [0.0], [1.0]])
    predicted = tf.constant([[0.9], [0.2], [0.8]])
    mask = tf.constant([1, 0, 1])
    mask = tf.cast(mask, tf.float32)

    masked_labels = tf.constant([[1.0], [0.0], [1.0]])
    masked_predicted = tf.constant([[0.9], [0.0], [0.8]])

    expected = tf.keras.losses.binary_crossentropy(masked_labels, masked_predicted)

    # Act
    loss = model._compute_loss(labels, predicted, mask)

    # Assert
    assert loss is not None
    tf.debugging.assert_near(loss, expected)

def test_train_step_has_expected_parameters():
    # Arrange
    function_signature = inspect.signature(SupervisedTransductiveModel.train_step)
    function_params = function_signature.parameters

    # Assert
    assert 'self' in function_params
    assert 'data' in function_params


def test_val_step_step_has_expected_parameters():
    # Arrange
    function_signature = inspect.signature(SupervisedTransductiveModel.val_step)
    function_params = function_signature.parameters

    # Assert
    assert 'self' in function_params
    assert 'data' in function_params

