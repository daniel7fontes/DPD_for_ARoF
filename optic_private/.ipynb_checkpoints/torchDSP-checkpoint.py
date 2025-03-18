import torch as th
from torch.fft import fft, ifft, fftfreq
from torchaudio.functional import resample
import numpy as np
import scipy.constants as const


def firFilter(h, x):
    """
    Perform FIR filtering and compensate for filter delay using FFT-based convolution.

    Parameters
    ----------
    h : torch.Tensor
        Coefficients of the FIR filter (impulse response, symmetric).
    x : torch.Tensor
        Input signal.

    Returns
    -------
    y : torch.Tensor
        Output (filtered) signal.
    """
    if not th.is_tensor(h):
        h = th.tensor(h, dtype=x.dtype, device=x.device)
    else:
        h = h.to(x.device)

    if len(x.shape) == 1:
        x = x.view(len(x), 1)

    nModes = x.shape[1]

    if x.shape[0] >= len(h):
        Lpad = x.shape[0] - len(h)
        padTensor = th.zeros(Lpad // 2, dtype=x.dtype, device=x.device)
        h = th.cat([padTensor, h, padTensor])
    else:
        Lpad = len(h) - x.shape[0]
        padTensor = th.zeros((Lpad // 2, x.shape[1]), dtype=x.dtype, device=x.device)
        x = th.cat([padTensor, x, padTensor])

    # Perform FFT on the filter coefficients and input signal
    h_fft = fft(h, norm="ortho")
    x_fft = fft(x, dim=0, norm="ortho")

    y = th.zeros(x.shape, dtype=th.complex64, device=x.device)

    for n in range(nModes):
        y_fft = x_fft[:, n] * h_fft
        y[:, n] = ifft(y_fft)

    if y.shape[1] == 1:
        y = y.flatten()

    y = th.roll(y, len(h) // 2, 0)

    if th.is_complex(x) or th.is_complex(h):
        pass
    else:
        y = y.real

    return y


def upsample(x, factor):
    """
    Upsample a PyTorch tensor by inserting zeros between the samples.

    Parameters
    ----------
    x: torch.Tensor
        Input tensor to upsample.
    factor : int
        Upsampling factor.

    Returns
    -------
    y : torch.Tensor
        Upsampled tensor.
    """
    if factor <= 0:
        raise ValueError("Upsampling factor must be a positive integer.")

    num_samples, num_channels = x.shape
    upsampled_length = num_samples * factor

    y = th.zeros(upsampled_length, num_channels, dtype=x.dtype, device=x.device)
    y[::factor, :] = x
    y = th.roll(y, factor // 2, 0)

    return y


def pnorm(x):
    """
    Normalize the average power of each componennt of x.

    Parameters
    ----------
    x : np.array
        Signal.

    Returns
    -------
    np.array
        Signal x with each component normalized in power.

    """
    return x / th.sqrt(th.mean(th.abs(x) ** 2))


def anorm(x):
    """
    Normalize the amplitude of each componennt of x.

    Parameters
    ----------
    x : np.array
        Signal.

    Returns
    -------
    np.array
        Signal x with each component normalized in amplitude.

    """
    return x / th.max(th.abs(x))


def hilbert(x):
    """
    Calculate the Hilbert transform of a 1D PyTorch tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input signal.

    Returns
    -------
    torch.Tensor
        Hilbert transform of the input signal.
    """
    if x.dim() != 1:
        x = x.reshape(
            -1,
        )

    # Perform FFT
    x_fft = fft(x)

    # Create the Hilbert kernel
    N = x.size(0)
    if N % 2 == 0:
        kernel = th.zeros(N, dtype=x.dtype, device=x.device)
        kernel[0] = kernel[N // 2] = 1
        kernel[1 : N // 2] = 2
    else:
        kernel = th.zeros(N, dtype=x.dtype, device=x.device)
        kernel[0] = 1
        kernel[1 : (N + 1) // 2] = 2

    # Apply the Hilbert transform in the frequency domain
    # and perform IFFT to get the time domain signal
    return ifft(x_fft * kernel)


def rrcFilterTaps(t, alpha, Ts):
    """
    Generate Root-Raised Cosine (RRC) filter coefficients.

    Parameters
    ----------
    t : array-like
        Time values.
    alpha : float
        RRC roll-off factor.
    Ts : float
        Symbol period.

    Returns
    -------
    coeffs : th.Tensor
        RRC filter coefficients.

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. (2008). Digital Communications (5th Edition). McGraw-Hill Education.
    """
    coeffs = th.zeros(len(t), dtype=th.float64)

    for i, t_i in enumerate(t):
        t_abs = abs(t_i)
        if t_i == 0:
            coeffs[i] = (1 / Ts) * (1 + alpha * (4 / th.pi - 1))
        elif t_abs == Ts / (4 * alpha):
            term1 = (1 + 2 / th.pi) * th.sin(th.pi / (4 * alpha))
            term2 = (1 - 2 / th.pi) * th.cos(th.pi / (4 * alpha))
            coeffs[i] = (alpha / (Ts * th.sqrt(2))) * (term1 + term2)
        else:
            t1 = th.pi * t_i / Ts
            t2 = 4 * alpha * t_i / Ts
            coeffs[i] = (
                (1 / Ts)
                * (
                    th.sin(t1 * (1 - alpha))
                    + 4 * alpha * t_i / Ts * th.cos(t1 * (1 + alpha))
                )
                / (th.pi * t_i * (1 - t2**2))
            )

    return coeffs


def rcFilterTaps(t, alpha, Ts):
    """
    Generate Raised Cosine (RC) filter coefficients.

    Parameters
    ----------
    t : array-like
        Time values.
    alpha : float
        RC roll-off factor.
    Ts : float
        Symbol period.

    Returns
    -------
    coeffs : th.Tensor
        RC filter coefficients.

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. (2008). Digital Communications (5th Edition). McGraw-Hill Education.
    """
    coeffs = th.zeros(len(t), dtype=th.float64)
    π = th.pi

    for i, t_i in enumerate(t):
        t_abs = abs(t_i)
        if t_abs == Ts / (2 * alpha):
            coeffs[i] = π / (4 * Ts) * th.sinc(1 / (2 * alpha))
        else:
            coeffs[i] = (
                (1 / Ts)
                * th.sinc(t_i / Ts)
                * th.cos(π * alpha * t_i / Ts)
                / (1 - 4 * alpha**2 * t_i**2 / Ts**2)
            )

    return coeffs


def pulseShape(pulseType, SpS=2, N=1024, alpha=0.1, Ts=1):
    """
    Generate a pulse shaping filter.

    Parameters
    ----------
    pulseType : string ('rect','nrz','rrc')
        Type of pulse shaping filter.
    SpS : int, optional
        Number of samples per symbol of input signal. The default is 2.
    N : int, optional
        Number of filter coefficients. The default is 1024.
    alpha : float, optional
        Rolloff of RRC filter. The default is 0.1.
    Ts : float, optional
        Symbol period in seconds. The default is 1.

    Returns
    -------
    filterCoeffs : th.Tensor
        Array of filter coefficients (normalized).

    """
    fa = (1 / Ts) * SpS

    if pulseType == "rect":
        filterCoeffs = th.cat(
            (th.zeros(int(SpS / 2)), th.ones(SpS), th.zeros(int(SpS / 2)))
        )
    elif pulseType == "nrz":
        t = th.linspace(-2, 2, SpS)
        Te = 1
        filterCoeffs = th.convolve(
            th.ones(SpS),
            2 / (th.sqrt(th.pi) * Te) * th.exp(-(t**2) / Te),
            mode="full",
        )
    elif pulseType == "rrc":
        t = th.linspace(-N // 2, N // 2, N) * (1 / fa)
        filterCoeffs = rrcFilterTaps(t, alpha, Ts)

    elif pulseType == "rc":
        t = th.linspace(-N // 2, N // 2, N) * (1 / fa)
        filterCoeffs = rcFilterTaps(t, alpha, Ts)

    filterCoeffs = filterCoeffs / th.sqrt(th.sum(filterCoeffs**2))

    return filterCoeffs


def lowPassFIR(fc, fa, N, typeF="rect"):
    """
    Calculate FIR coefficients of a lowpass filter.

    Parameters
    ----------
    fc : float
        Cutoff frequency.
    fa : float
        Sampling frequency.
    N : int
        Number of filter coefficients.
    typeF : string, optional
        Type of response ('rect', 'gauss'). The default is "rect".

    Returns
    -------
    h : th.Tensor
        Filter coefficients.

    """
    fu = fc / fa
    d = (N - 1) / 2
    n = th.arange(0, N, dtype=th.float64)

    # calculate filter coefficients
    if typeF == "rect":
        h = (2 * fu) * th.sinc(2 * fu * (n - d))
    elif typeF == "gauss":
        h = (
            th.sqrt(2 * np.pi / np.log(2))
            * fu
            * th.exp(-(2 / np.log(2)) * (np.pi * fu * (n - d)) ** 2)
        )
    return h


def clockSamplingInterp(x, Fs_in=1, Fs_out=1, jitter_rms=1e-9):
    """
    Interpolate signal to a given sampling rate.

    Parameters
    ----------
    x : th.Tensor
        Input signal.
    Fs_in : float, optional
        Sampling frequency of the input signal. Default is 1.
    Fs_out : float, optional
        Sampling frequency of the output signal. Default is 1.
    jitter_rms : float, optional
        Standard deviation of the time jitter. Default is 1e-9.

    Returns
    -------
    y : th.Tensor
        Resampled signal.

    """
    nModes = x.shape[1]

    inTs = 1 / Fs_in
    outTs = 1 / Fs_out

    tin = th.arange(0, x.shape[0]) * inTs
    tout = th.arange(0, x.shape[0] * inTs, outTs)

    jitter = th.normal(0, jitter_rms, tout.shape)
    tout += jitter

    y = th.zeros((len(tout), x.shape[1]), dtype=x.dtype, device=x.device)

    for k in range(nModes):
        y[:, k] = th.interp(tout, tin, x[:, k])

    return y


def quantizer(x, nBits=16, maxV=1, minV=-1):
    """
    Quantize the input signal using a uniform quantizer with the specified precision.

    Parameters
    ----------
    x : th.Tensor
        The input signal to be quantized.
    nBits : int
        Number of bits used for quantization. The quantizer will have 2^nBits levels.
    maxV : float, optional
        Maximum value for the quantizer's full-scale range (default is 1).
    minV : float, optional
        Minimum value for the quantizer's full-scale range (default is -1).

    Returns
    -------
    th.Tensor
        The quantized output signal with the same shape as 'x', quantized using 'nBits' levels.

    """
    Δ = (maxV - minV) / (2**nBits - 1)

    d = th.arange(minV, maxV + Δ, Δ)

    y = th.zeros(x.shape, dtype=x.dtype, device=x.device)

    for indMode in range(x.shape[1]):
        for idx in range(len(x)):
            y[idx, indMode] = d[int(th.argmin(th.abs(x[idx, indMode] - d)))]

    return y


def decimate(Ei, param):
    """
    Decimate signal.

    Parameters
    ----------
    Ei : th.Tensor
        Input signal.
    param : core.parameter
        Decimation parameters:

        - param.SpS_in  : samples per symbol of the input signal.

        - param.SpS_out : samples per symbol of the output signal.

    Returns
    -------
    Eo : th.Tensor
        Decimated signal.

    """
    if len(Ei.shape) == 1:
        Ei = Ei.reshape(len(Ei), 1)

    decFactor = int(param.SpS_in / param.SpS_out)

    # simple timing recovery
    sampDelay = th.zeros(Ei.shape[1])

    # finds best sampling instant
    # (maximum variance sampling time)
    for k in range(Ei.shape[1]):
        a = Ei[:, k].reshape(Ei.shape[0], 1)
        varVector = th.var(a.reshape(-1, param.SpS_in), axis=0)
        sampDelay[k] = th.argmax(varVector).item()

    # downsampling
    Eo = Ei[::decFactor, :].clone()

    for k in range(Ei.shape[1]):
        Ei[:, k] = th.roll(Ei[:, k], -int(sampDelay[k]))
        Eo[:, k] = Ei[0::decFactor, k]

    return Eo


def gaussianComplexNoise(shapeOut, σ2=1.0, device="cpu"):
    """
    Generate complex circular Gaussian noise.

    Parameters
    ----------
    shapeOut : tuple of int
        Shape of ndarray to be generated.
    σ2 : float, optional
        Variance of the noise (default is 1).

    Returns
    -------
    noise : th.Tensor
        Generated complex circular Gaussian noise.
    """
    return th.normal(0, th.sqrt(σ2 / 2), shapeOut, device=device) + 1j * th.normal(
        0, th.sqrt(σ2 / 2), shapeOut, device=device
    )


def gaussianNoise(shapeOut, σ2=1.0, device="cpu"):
    """
    Generate Gaussian noise.

    Parameters
    ----------
    shapeOut : tuple of int
        Shape of ndarray to be generated.
    σ2 : float, optional
        Variance of the noise (default is 1).

    Returns
    -------
    noise : th.Tensor
        Generated Gaussian noise.
    """
    return th.normal(0, th.sqrt(σ2), shapeOut, device=device)


def phaseNoise(lw, Nsamples, Ts, device="cpu"):
    """
    Generate realization of a random-walk phase-noise process.

    Parameters
    ----------
    lw : float
        Laser linewidth.
    Nsamples : int
        Number of samples to be drawn.
    Ts : float
        Sampling period.

    Returns
    -------
    phi : th.Tensor
        Realization of the phase noise process.

    """
    σ2 = 2 * np.pi * lw * Ts
    phi = th.zeros(Nsamples, dtype=th.float64, device=device)

    for ind in range(Nsamples - 1):
        phi[ind + 1] = phi[ind] + th.normal(0, th.sqrt(th.tensor(σ2.to(device))))

    return phi


def finddelay(x, y):
    """
    Find delay between x and y using FFT-based cross-correlation.

    Parameters
    ----------
    x : torch.Tensor
        Signal 1.
    y : torch.Tensor
        Signal 2.

    Returns
    -------
    d : int
        Delay between x and y, in samples.

    """
    # Compute FFT of x and y
    fft_x = fft(th.abs(x), n=x.shape[0] * 2)
    fft_y = fft(th.abs(y), n=y.shape[0] * 2)

    # Compute complex conjugate of fft_y
    fft_y_conj = th.conj(fft_y)

    # Compute element-wise multiplication of fft_x and complex conjugate of fft_y
    cross_corr = ifft(fft_x * fft_y_conj)

    # Find index of maximum absolute value in cross-correlation
    max_idx = th.argmax(th.abs(cross_corr))

    # Compute delay
    delay = max_idx.item() - x.shape[0]

    return delay
