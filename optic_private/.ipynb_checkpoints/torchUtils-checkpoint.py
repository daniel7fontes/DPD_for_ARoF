import torch as th
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging as logg
from tqdm import tqdm
import contextlib


def signal_power(x):
    """
    Calculate the total power of x.

    Parameters
    ----------
    x : torch.Tensor
        Signal.

    Returns
    -------
    scalar
        Total power of x: P = sum(abs(x)**2).

    """
    return th.sum(th.mean(x * th.conj(x), dim=0).real)


class slidingWindowDataSet(Dataset):
    """
    A custom dataset class for creating sliding window samples from a signal.

    Args:
        x (numpy.ndarray): Input signal.
        y (numpy.ndarray): Array of corresponding targets.
        Ntaps (int): Number of taps/window size.
        SpS (int, optional): Samples per symbol. Defaults to 1.

    Attributes:
        Ntaps (int): Number of taps/window size.
        SpS (int): Samples per symbol.
        x (numpy.ndarray): Input signal padded with zeros.
        y (numpy.ndarray): Array of corresponding targets.

    Methods:
        __getitem__(self, idx): Retrieves the item at the specified index.
        __len__(self): Returns the total number of items in the dataset.
    """

    def __init__(self, x, y, Ntaps, SpS=1, c=False, augment = False):
        """
        Initialize the slidingWindowDataSet.

        Args:
            x (numpy.ndarray): Input signal.
            y (numpy.ndarray): Array of corresponding targets.
            Ntaps (int): Number of taps/window size.
            SpS (int, optional): Samples per symbol. Defaults to 1.
        """
        super(slidingWindowDataSet, self).__init__()
        self.Ntaps = Ntaps
        self.SpS = SpS
        x_pad = th.nn.functional.pad(x, (Ntaps // 2, Ntaps // 2), "constant", 0)
        if augment:
            self.x = augmentFeatures(x_pad).to(th.float32)
        else:
            self.x = th.view_as_real(x_pad).to(th.float32)

        self.y = th.view_as_real(y).to(th.float32)
        self.augment = augment

    def __getitem__(self, idx):
        """
        Retrieves the item at the specified index.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: A tuple containing the input and target tensors.
        """
        center_idx = idx * self.SpS + self.Ntaps // 2
        start_idx = center_idx - self.Ntaps // 2
        end_idx = center_idx + self.Ntaps // 2

        if start_idx == end_idx:
            inputs = self.x[center_idx, :].flatten()
        else:
            inputs = self.x[start_idx:end_idx, :].flatten()

        target = self.y[idx, :].flatten()

        return inputs, target

    def __len__(self):
        """
        Returns the total number of items in the dataset.

        Returns:
            int: Total number of items in the dataset.
        """
        return (len(self.x) - self.Ntaps) // self.SpS


def augmentFeatures(x):
    """
    Augment the features of a complex-valued tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with complex values.

    Returns
    -------
    torch.Tensor
        Tensor with augmented features. Each column contains:
        - Real part of the input tensor.
        - Imaginary part of the input tensor.
        - Absolute value of the input tensor.
        - Absolute squared value of the input tensor.
        - Absolute value to the third power of the input tensor.
    """
    return th.stack([x.real, x.imag, th.abs(x), th.abs(x) ** 2, th.abs(x) ** 3], dim=1)


class autoEncoderDataSet(Dataset):
    """
    A custom dataset class for an autoencoder model.

    Args:
        symbols_idx (numpy.ndarray): Array of symbol indices.
        M (int): Number of symbols.

    Attributes:
        symbols_idx (numpy.ndarray): Array of symbol indices.
        M (int): Number of symbols.

    Methods:
        __getitem__(self, idx): Retrieves the item at the specified index.
        __len__(self): Returns the total number of items in the dataset.
    """

    def __init__(self, symbols_idx, M):
        """
        Initialize the autoEncoderDataSet.

        Args:
            symbols_idx (numpy.ndarray): Array of symbol indices.
            M (int): Number of symbols.
        """
        super(autoEncoderDataSet, self).__init__()
        self.symbols_idx = symbols_idx
        self.M = M

    def __getitem__(self, idx):
        """
        Retrieves the item at the specified index.

        Args:
            idx (int): Index of the item.

        Returns:
            the input (and target) tensor.
        """
        # One-hot encoded vector
        vec_one_hot = th.zeros(self.M)

        vec_one_hot[self.symbols_idx[idx]] = 1

        return vec_one_hot

    def __len__(self):
        """
        Returns the total number of items in the dataset.

        Returns:
            int: Total number of items in the dataset.
        """
        return len(self.symbols_idx)


def separate_into_batches(nSamples, batchSize, shuffle=True):
    """
    Separates the indices of samples into batches of a specified size.

    Args:
        nSamples (int): Total number of samples.
        batchSize (int): Size of each batch.
        shuffle (bool): Whether to shuffle the indices before splitting into batches. Defaults to True.

    Yields:
        np.ndarray: Indices of each batch as a numpy array.
    """
    allIndices = np.arange((nSamples // batchSize) * batchSize)

    if shuffle:
        np.random.shuffle(allIndices)

    for startIdx in range(0, len(allIndices) - 1, batchSize):
        yield allIndices[startIdx : startIdx + batchSize]


class MLP(nn.Module):
    def __init__(self, layer_sizes, activation=nn.ReLU()):
        """
        Initialize the Multi-Layer Perceptron (MLP) network.

        Args:
            layer_sizes (list): List containing the number of neurons in each layer.
            activation (torch.nn.Module, optional): Activation function to be used in the hidden layers. Default is ReLU.

        Example:
            mlp = MLP([10, 5, 2], activation=nn.Sigmoid())  # Create an MLP with 3 layers, using Sigmoid activation in the hidden layers
        """
        super(MLP, self).__init__()
        self.num_layers = len(layer_sizes)
        self.layers = nn.ModuleList()
        self.activation = activation

        for i in range(self.num_layers - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x):
        """
        Perform forward propagation through the MLP network.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output of the MLP network.

        Example:
            output = mlp.forward(input_data)  # Perform forward propagation
        """
        for ind, layer in enumerate(self.layers):
            x = self.activation(layer(x)) if ind < self.num_layers - 2 else layer(x)
        return x


def generate_one_hot_vectors(class_ids, M):
    """
    Generates one-hot encoded vectors for a given array of class IDs.

    Args:
        class_ids (numpy.ndarray or list): Array of class IDs.
        M (int): Number of classes.

    Returns:
        torch.Tensor: One-hot encoded vectors as a tensor.

    """
    n_class_ids = len(class_ids)

    # One-hot encoded vectors
    vec_one_hot = th.zeros((n_class_ids, M))

    for ii in range(n_class_ids):
        vec_one_hot[ii, class_ids[ii]] = 1

    return vec_one_hot


class Encoder(nn.Module):
    """Encoder module of the MLP autoencoder.

    The encoder takes an input tensor and performs the encoding process by passing it through
    a series of linear layers and activation functions.

    Args:
        input_dim (int): The dimensionality of the input tensor.
        hidden_dims (list): List of integers specifying the number of neurons in each hidden layer.
        activation_fn (torch.nn.Module): Activation function to be applied after each linear layer.

    Example:
        encoder = Encoder(input_dim=784, hidden_dims=[256, 128], activation_fn=torch.nn.ReLU())
        encoded_data = encoder(input_data)
    """

    def __init__(self, dims, normalize=True, activation=nn.ReLU()):
        super(Encoder, self).__init__()
        self.normalize = normalize

        prev_dim = dims[0]

        encoder_layers = []
        for hidden_dim in dims[1:-1]:
            encoder_layers.extend(
                (
                    nn.Linear(prev_dim, hidden_dim),
                    activation,
                )
            )
            prev_dim = hidden_dim

        encoder_layers.append(nn.Linear(prev_dim, dims[-1]))

        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        """Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input tensor to be encoded.

        Returns:
            torch.Tensor: Encoded representation of the input tensor.
        """
        x = self.encoder(x)

        if self.normalize:
            if x.shape[1] == 2:
                x = x / th.sqrt(th.mean(x[:, 0] ** 2 + x[:, 1] ** 2))
            else:
                x = x / th.sqrt(th.mean(x**2))

        return x


class Decoder(nn.Module):
    """Decoder module of the MLP autoencoder.

    The decoder takes an encoded tensor and performs the decoding process by passing it through
    a series of linear layers and activation functions. It reconstructs the original input tensor
    from the encoded representation.

    Args:
        hidden_dims (list): List of integers specifying the number of neurons in each hidden layer.
        output_dim (int): The dimensionality of the output tensor.
        activation_fn (torch.nn.Module): Activation function to be applied after each linear layer.

    Example:
        decoder = Decoder(hidden_dims=[128, 256], output_dim=784, activation_fn=torch.nn.ReLU())
        decoded_data = decoder(encoded_data)
    """

    def __init__(self, dims, activation=nn.ReLU()):
        super(Decoder, self).__init__()

        prev_dim = dims[-2]

        decoder_layers = [nn.Linear(dims[-1], prev_dim)]
        for hidden_dim in reversed(dims[:-2]):
            decoder_layers.extend((activation, nn.Linear(prev_dim, hidden_dim)))
            prev_dim = hidden_dim

        decoder_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """Forward pass of the decoder.

        Args:
            x (torch.Tensor): Encoded tensor to be decoded.

        Returns:
            torch.Tensor: Reconstructed tensor from the encoded representation.
        """
        return self.decoder(x)


class memoryLessDataSet(Dataset):
    """
    A custom complex2real memoryless dataset class

    Args:
        signal (numpy.ndarray): Input signal.
        symbols (numpy.ndarray): Array of corresponding symbols.

    Attributes:
        signal (numpy.ndarray): Input signal.
        symbols (numpy.ndarray): Array of corresponding symbols.

    Methods:
        __getitem__(self, idx): Retrieves the item at the specified index.
        __len__(self): Returns the total number of items in the dataset.
    """

    def __init__(self, signal, symbols, augment=False):
        """
        Initialize the memoryLessDataSet.

        Args:
            signal (numpy.ndarray): Input signal.
            symbols (numpy.ndarray): Array of corresponding symbols.
        """
        super(memoryLessDataSet, self).__init__()
        self.symbols = th.view_as_real(symbols)
        self.augment = augment

        if augment:
            self.signal = augmentFeatures(signal)
        else:
            self.signal = th.view_as_real(signal)

    def __getitem__(self, idx):
        """
        Retrieves the item at the specified index.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: A tuple containing the input and target tensors.
        """
        inputs = self.signal[idx, :].to(th.float32)

        target = self.symbols[idx, :].to(th.float32)

        return inputs, target

    def __len__(self):
        """
        Returns the total number of items in the dataset.

        Returns:
            int: Total number of items in the dataset.
        """
        return len(self.signal)


def train_model(dataloader, model, loss_fn, optimizer):
    """
    Train the given model using the provided dataloader, loss function, and optimizer.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        The dataloader providing the training data.
    model : torch.nn.Module
        The neural network model to be trained.
    loss_fn : torch.nn.Module
        The loss function used for optimization.
    optimizer : torch.optim.Optimizer
        The optimizer object responsible for updating the model's parameters.

    Returns
    -------
    None
    """
    size = len(dataloader.dataset)
    model.train()
    losses = np.zeros(len(dataloader))

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses[batch] = loss.item()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            logg.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return losses


def test_model(dataloader, model, loss_fn):
    """
    Evaluate the given model using the provided dataloader and loss function.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        The dataloader providing the test data.
    model : torch.nn.Module
        The neural network model to be evaluated.
    loss_fn : torch.nn.Module
        The loss function used for evaluation.

    Returns
    -------
    loss : array of loss values
    """
    losses = np.zeros(len(dataloader))
    model.eval()
    with th.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            loss = loss_fn(pred, y)
            losses[batch] = loss.item()

    logg.info(f"Avg test loss: {np.mean(losses):>8f} \n")

    return losses


def fitFilterNN(
    sig, model, Ntaps, SpS=1, batchSize=100, augment=False, predict=True, prgsBar=False
):
    sigPad = th.nn.functional.pad(sig, (Ntaps // 2, Ntaps // 2), "constant", 0)

    model.eval() if predict else model.train()
    numSymb = len(sig) // SpS
    numBatches = numSymb // batchSize

    indTaps = th.arange(0, Ntaps, dtype=th.int64)
    y = th.zeros(numSymb, dtype=th.complex64, device=sig.device)

    if augment:
        sigPad = augmentFeatures(sigPad)
    else:
        sigPad = th.view_as_real(sigPad).to(th.float32)

    with th.no_grad() if predict else contextlib.nullcontext():
        for k in tqdm(range(numBatches), disable=not (prgsBar)):
            start_idx = k * batchSize
            end_idx = (k + 1) * batchSize

            sampleInd = th.arange(start_idx, end_idx, dtype=th.int64)
            indIn = (
                indTaps + sampleInd[:, None] * SpS
            )  # Broadcasting to avoid nested loops

            x = sigPad[indIn.flatten(), :].reshape(
                batchSize, -1
            )  # Flattening and reshaping
            y[sampleInd] = th.view_as_complex(model(x.unsqueeze(1))).squeeze(1)

    return y
