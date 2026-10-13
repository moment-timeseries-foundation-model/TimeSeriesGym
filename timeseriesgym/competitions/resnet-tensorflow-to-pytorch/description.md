# Overview

## Description
In this competition, your task is to implement ResNet for `Time Series Classification` tasks using PyTorch, based on a provided implementation written in TensorFlow.

**Important**: This challenge evaluates your ability to accurately translate a TensorFlow-based implementation into a PyTorch version, while adhering to the required specifications for the PyTorch code. Since both your understanding of the original TensorFlow code and your Python coding proficiency are critical to this task, please provide clear documentation for each step in your submission file.

## Code Requirements

Your PyTorch implementation **must** include a `ResNet` class and two functions: `train` and `predict`. Note that capitalization matters for evaluation—e.g., the class must be named `ResNet`; a class named `resnet` is not considered correct. Below are the specifications:

### ResNet

This class defines the ResNet model and **must** inherit from `torch.nn.Module`, as shown below:

```python
from torch.nn import Module

class ResNet(Module):
```

It **must** include two methods: `__init__` and `forward`, each with the following requirements. Note that `self` is implicitly included as the first parameter for all class methods and is therefore omitted in the parameter descriptions below.

- `__init__` - Constructor to initialize the ResNet model. It **must** include the following parameters (you may add more if needed):

| Parameter           | Type     | Description                                                     |
|---------------------|----------|-----------------------------------------------------------------|
| `output_directory`  | str      | Directory to save the model and load an existing model from.    |
| `input_shape`       | tuple or list | Shape of the input time series (n_channels, n_time_steps). |
| `n_classes`         | int      | Number of classes for the classification tasks.                 |
| `n_feature_maps`    | int      | Number of feature maps in the convolutional layers.             |
| `verbose`           | bool     | If True, print details.                                         |
| `random_seed`       | int      | Random seed for reproducibility.                                |

- `forward` — Defines the forward pass of the model. It **must** take exactly the following input parameter and return the specified output tensor:

| Parameter    | Type         | Description                                                        |
|--------------|--------------|--------------------------------------------------------------------|
| `x`          | torch.Tensor | Input tensor of shape (batch_size, n_channels, n_time_steps).      |

| Returns      | Type         | Description                                                        |
|--------------|--------------|--------------------------------------------------------------------|
| `y`          | torch.Tensor | Output tensor of shape (batch_size, n_classes).                    |


### train

This function defines the training loop for the ResNet model. It **must** include the following parameters (you may add more if needed), and return exactly the specified output:

| Parameter        | Type                        | Description                                                                             |
|------------------|-----------------------------|-----------------------------------------------------------------------------------------|
| `model`          | ResNet                      | The ResNet model to train.                                                              |
| `train_data`     | torch.utils.data.Dataset    | Training dataset; each sample is a tuple (input time series, ground-truth class label). |
| `val_data`       | torch.utils.data.Dataset    | Validation dataset (optional), in the same format as the training dataset.              |
| `batch_size`     | int                         | Batch size for training.                                                                |
| `n_epochs`       | int                         | Number of epochs to train the model.                                                    |
| `learning_rate`  | float                       | Learning rate for the optimizer.                                                        |
| `verbose`        | bool                        | If True, print training and evaluation details.                                         |

| Returns      | Type         | Description                                       |
|--------------|--------------|---------------------------------------------------|
| `model`      | ResNet       | The trained model.                                |


### predict

This function generates predictions on a given test dataset using a ResNet model. It **must** include the following parameters (you may add more if needed), and return exactly the specified outputs:

| Parameter        | Type                        | Description                                                                          |
|------------------|-----------------------------|--------------------------------------------------------------------------------------|
| `model`          | ResNet                      | The ResNet model used for making predictions.                                        |
| `test_data`      | torch.utils.data.Dataset    | Test dataset; each sample is a tuple (input time series, ground-truth class label).  |

| Returns      | Type            | Description                                                           |
|--------------|-----------------|-----------------------------------------------------------------------|
| `y_true`     | numpy.ndarray   | Ground-truth labels for the test dataset, with shape (n_samples,).    |
| `y_pred`     | numpy.ndarray   | Predicted scores for each class, with shape (n_samples, n_classes).   |


**Important**: You may define additional helper classes, methods, or functions as needed, but the required components above must strictly follow their specifications.


## Evaluation

Submissions will be evaluated based on the following three criteria:

- **Code Definition**: All required components must be implemented and conform to the specifications outlined in **Code Requirements**.
- **Code Execution**: The `ResNet` class (including `__init__` and `forward`), as well as the `train` and `predict` functions, must execute successfully and produce valid outputs when given inputs that meet the specifications.
- **Test Results**: A model trained using your implementation, with the grading arguments shown below, should achieve an accuracy greater than 80% on the test set, using the provided training dataset. The accuracy is that of `argmax(y_pred, axis=1)` against the ground-truth test labels. The model must be trained for `n_epochs` passes over `train_data`, and no more: the grader counts the samples that are back-propagated through your `ResNet` (any instance of it, from the moment `submission.py` is imported until `predict` returns), and the test fails if there are more than `n_epochs * len(train_data)`. Forward passes without back-propagation, such as validation or predictions, do not count.

The following sample code illustrates how test predictions are generated using your implemented `ResNet` model. The grader passes the arguments by keyword, with these values, so the parameter names must match the specifications above:

```python
from submission import ResNet, train, predict

model = ResNet(output_directory="./output/", input_shape=[1, 50], n_classes=2, n_feature_maps=64, verbose=True, random_seed=42)
model = train(model=model, train_data=train_data, val_data=val_data, batch_size=64, n_epochs=1, learning_rate=0.001, verbose=True)
y_true, y_pred = predict(model=model, test_data=test_data)
```

### Submission Format

You **MUST** submit your code in a single Python script named `submission.py` in your submission directory.


## File Descriptions

- **resnet.py**: The ResNet implementation in TensorFlow.
- **utils.py**: Utility functions used in **resnet.py**, also written in TensorFlow.
- **sample_submission.py**: A sample submission Python script containing the correct class, methods, and function definitions, but with empty implementations.
- **x.pt**: A `torch.Tensor` file containing a small set of input time series for testing the `forward` method of the `ResNet` class.
- **[train/val/test]_data.pt**: Training, validation, and test datasets in `torch.utils.data.Dataset` format. The training and validation datasets include input time series and their ground-truth labels. The test dataset includes input time series with dummy class labels (all set to 0). Ground-truth test labels are only available for evaluation.


## Acknowledgments

The TensorFlow implementation of ResNet is sourced from the [dl-4-tsc GitHub repository](https://github.com/hfawaz/dl-4-tsc) as referenced in [1].


## References

1. Fawaz, H. I. (2020). Deep learning for time series classification. arXiv preprint arXiv:2010.00567.
