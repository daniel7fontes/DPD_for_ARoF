import torch as th
import numpy as np
from torch.fft import fft, ifft, fftshift


def hermit(V):
    """
    Hermitian symmetry block.

    Parameters
    ----------
    V : complex-valued th.tensor
        input tensor

    Returns
    -------
    Vh : complex-valued th.tensor
        tensor with Hermitian symmetry
    """
    Vh = th.zeros(2 * len(V) + 2, dtype=th.complex64, device=V.device)
    Vh[1 : len(V) + 1] = V

    for j in range(len(V)):
        Vh[len(Vh) - j - 1] = th.conj(V[j])

    return Vh


def zeroPad(x, L):
    """
    Pad an input array with zeros on both sides using PyTorch.

    Parameters
    ----------
    x : tensor
        Input tensor to be padded.
    L : int
        Number of zeros to pad on each side of the tensor.

    Returns
    -------
    padded_tensor : tensor
        Padded tensor with zeros added at the beginning and end.

    Notes
    -----
    This function pads the input tensor `x` with `L` zeros on both sides, effectively increasing
    its length by `2*L`.

    """
    return th.nn.functional.pad(x, (L, L), "constant", 0)


def interp1d(y_old, x_old, x_new):
    """
    Perform 1D linear interpolation with extrapolation.

    Parameters
    ----------
    y_old : th.Tensor
        Old y-values.
    x_old : th.Tensor
        Old x-values.
    x_new : th.Tensor
        New x-values for interpolation.

    Returns
    -------
    th.Tensor
        Interpolated y-values corresponding to x_new.
    """
    # Check if x_old is sorted
    if (x_old[1:] - x_old[:-1] < 0).any():
        raise ValueError("x_old must be sorted in ascending order.")

    # Find the indices of the interval ends for each x_new
    indices = th.searchsorted(x_old, x_new)

    # Find the closest indices for x_new that are within bounds
    indices = th.clamp(indices, 1, len(x_old) - 1)

    # Calculate the fractional distance within each interval
    dx = x_old[indices] - x_old[indices - 1]
    dy = y_old[indices] - y_old[indices - 1]
    frac = (x_new - x_old[indices - 1]) / dx

    # Perform linear interpolation
    y_new = y_old[indices - 1] + frac * dy

    # Handle extrapolation for points outside the original interval
    if (x_new < x_old[0]).any():
        y_new[x_new < x_old[0]] = y_old[0] + (x_new[x_new < x_old[0]] - x_old[0]) * (
            y_old[1] - y_old[0]
        ) / (x_old[1] - x_old[0])
    if (x_new > x_old[-1]).any():
        y_new[x_new > x_old[-1]] = y_old[-1] + (
            x_new[x_new > x_old[-1]] - x_old[-1]
        ) * (y_old[-1] - y_old[-2]) / (x_old[-1] - x_old[-2])

    return y_new


def modulateOFDM(symb, param):
    """
    Modulate OFDM signal.

    Parameters
    ----------
    Nfft          : scalar
                    size of FFT
    G             : scalar
                    cyclic prefix length
    pilot         : complex-valued scalar
                    pilot symbol
    pilotCarriers : np.array
                    indexes of pilot subcarriers
    symb          : complex-valued array of modulation symbols
                    symbols sequency transmitted
    hermitSym     : boolean
                    True-> Real OFDM symbols / False: Complex OFDM symbols
    SpS           : int
                    oversampling factor

    Returns
    -------
    symb_OFDM   : complex-valued np.array
                    OFDM symbols sequency transmitted
    """
    device = symb.device
    # Check and set default values for input parameters
    Nfft = getattr(param, "Nfft", 512)
    G = getattr(param, "G", 4)
    hermitSymmetry = getattr(param, "hermitSymmetry", False)
    pilot = getattr(param, "pilot", 1 + 1j)
    pilotCarriers = getattr(param, "pilotCarriers", np.array([], dtype=np.int64))
    SpS = getattr(param, "SpS", 2)

    # Number of pilot subcarriers
    Np = len(pilotCarriers)

    # Number of subcarriers
    Ns = Nfft // 2 if hermitSymmetry else Nfft
    numSymb = len(symb)
    numOFDMframes = numSymb // (Ns - Np - 1)

    Carriers = np.arange(0, Ns)
    dataCarriers = np.array(list(set(Carriers) - set(pilotCarriers) - set([Nfft//2])))
 
    # Serial to parallel
    symb_par = th.reshape(symb, (numOFDMframes, Ns - Np - 1))
    sigOFDM_par = th.zeros(
        (numOFDMframes, SpS * (Nfft + G)), dtype=th.complex64, device=device
    )
    
    for indFrame in range(numOFDMframes):
        # Start OFDM frame with zeros
        frameOFDM = th.zeros(Ns, dtype=th.complex64, device=device)

        # Insert data and pilot subcarriers
        frameOFDM[dataCarriers] = symb_par[indFrame, :]
        frameOFDM[pilotCarriers] = pilot

        # Hermitian symmetry
        if hermitSymmetry:
            frameOFDM = hermit(frameOFDM)

        # IFFT operation
        sigOFDM_par[indFrame, SpS * G : SpS * (G + Nfft)] = ifft(
            fftshift(zeroPad(frameOFDM, (Nfft * (SpS - 1)) // 2))
        ) * np.sqrt(SpS * Nfft)

        # Cyclic prefix addition
        if G > 0:
            sigOFDM_par[indFrame, 0 : SpS * G] = sigOFDM_par[
                indFrame, Nfft * SpS : SpS * (Nfft + G)
            ].clone()

    return sigOFDM_par.view(-1)


def demodulateOFDM(sig, param):
    """
    Demodulate OFDM signal.

    Parameters
    ----------
    Nfft          : scalar
                    size of FFT
    N             : scalar
                    number of transmitted subcarriers
    G             : scalar
                    cyclic prefix length
    pilot         : complex-valued scalar
                    pilot symbol
    pilotCarriers : th.tensor
                    indexes of pilot subcarriers
    sig           : complex-valued tensor
                    OFDM signal sequency received at one sample per symbol
    returnChannel : bool
                    return estimated channel

    Returns
    -------
    symbRx        : complex th.tensor
                    demodulated symbols sequency received
    Notes
    -----
    The input signal must be sampled at one sample per symbol.
    """
    device = sig.device

    # Check and set default values for input parameters
    Nfft = getattr(param, "Nfft", 512)
    G = getattr(param, "G", 4)
    hermitSymmetry = getattr(param, "hermitSymmetry", False)
    pilot = getattr(param, "pilot", 1 + 1j).to(device)
    pilotCarriers = getattr(param, "pilotCarriers", np.array([], dtype=np.int64))
    returnChannel = getattr(param, "returnChannel", False)

    # Number of pilot subcarriers
    Np = len(pilotCarriers)

    # Number of subcarriers
    N = Nfft // 2 if hermitSymmetry else Nfft
    Carriers = np.arange(0, N)
    dataCarriers = np.array(list(set(Carriers) - set(pilotCarriers) - set([Nfft//2])))

    H_abs = 0
    H_pha = 0

    numSymb = sig.shape[0]
    numOFDMframes = numSymb // (Nfft + G)

    sig_par = th.reshape(sig, (numOFDMframes, Nfft + G))

    # Cyclic prefix removal
    sig_par = sig_par[:, G : G + Nfft]

    # FFT operation
    for indFrame in range(numOFDMframes):
        sig_par[indFrame, :] = th.fft.fftshift(
            th.fft.fft(sig_par[indFrame, :])
        ) / np.sqrt(Nfft)

    if hermitSymmetry:
        # Removal of hermitian symmetry
        sig_par = sig_par[:, 1 : 1 + N]

    # Channel estimation and single tap equalization
    if Np != 0:
        # Channel estimation
        for indFrame in range(numOFDMframes):
            x_in = th.tensor(pilotCarriers, device=device)
            x_out = th.tensor(Carriers, device=device)

            H_est = sig_par[indFrame, :][pilotCarriers] / pilot
            H_abs += interp1d(th.abs(H_est), x_in, x_out)
            H_pha += interp1d(unwrap(th.angle(H_est)), x_in, x_out)

            if indFrame == numOFDMframes - 1:
                H_abs = H_abs / numOFDMframes
                H_pha = H_pha / numOFDMframes

        for indFrame in range(numOFDMframes):
            sig_par[indFrame, :] = sig_par[indFrame, :] / (H_abs * th.exp(1j * H_pha))

        # Pilot extraction
        sig_par = sig_par[:, dataCarriers]

    if returnChannel:
        return sig_par.view(-1), H_abs * th.exp(1j * H_pha)
    else:
        return sig_par.view(-1)


def unwrap(phase):
    """
    Unwrap phase values.

    Parameters
    ----------
    phase : torch.Tensor
        Input phase tensor.

    Returns
    -------
    torch.Tensor
        Unwrapped phase tensor.

    Notes
    -----
    This function computes the unwrapped phase values of a given input tensor.
    It calculates the differences between consecutive phase values, adjusts them
    to ensure smooth unwrapping, and accumulates these differences to obtain
    the unwrapped phase.
    """
    # Compute differences between consecutive phase values
    diff = th.diff(phase, prepend=th.tensor([0], device=phase.device))

    # Adjust differences to ensure smooth unwrapping
    diff = th.where(
        diff.abs() > 3.141592653589793, -th.sign(diff) * 6.283185307179586 + diff, diff
    )

    # Accumulate differences to obtain unwrapped phase
    unwrapped = th.cumsum(diff, dim=0)

    return unwrapped
