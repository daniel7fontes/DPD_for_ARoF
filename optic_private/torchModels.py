import logging as logg
import torch as th
from torch.fft import fft, ifft, fftfreq, fftshift
from optic_private.torchMetrics import signal_power
from optic_private.torchDSP import (
    firFilter,
    clockSamplingInterp,
    quantizer,
    gaussianComplexNoise,
    gaussianNoise,
    lowPassFIR,
    pnorm,
)
from optic_private.utils import parameters, lin2dB
import numpy as np
import scipy.constants as const
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


def pm(Ai, u, Vpi):
    """
    Optical Phase Modulator (PM).

    Parameters
    ----------
    Ai : th.Tensor
        Amplitude of the optical field at the input of the PM.
    u : th.Tensor
        Electrical driving signal.
    Vπ : float
        PM's Vπ voltage.

    Returns
    -------
    Ao : th.Tensor
        Modulated optical field at the output of the PM.

    """
    # if not isinstance(Vpi, th.tensor):
    #     Vpi = th.tensor([Vpi], device=u.device)
    if isinstance(Ai, (float, int)):
        Ai = Ai * th.ones(u.shape, dtype=u.dtype, device=u.device)
    elif isinstance(Ai, th.Tensor):
        if Ai.shape == th.Size([]) and u.shape != th.Size([]):
            Ai = Ai * th.ones(u.shape, dtype=u.dtype, device=u.device)
        else:
            assert Ai.shape == u.shape, "Ai and u need to have the same dimensions"
    else:
        raise ValueError("Unsupported data type for Ai")

    π = th.acos(th.zeros(1, device=u.device)).item() * 2

    return Ai * th.exp(1j * (u / Vpi) * π)


def mzm(Ai, u, param=None):
    """
    Optical Mach-Zehnder Modulator (MZM).

    Parameters
    ----------
    Ai : float or np.array
        Amplitude of the optical field at the input of the IQM.
    u : th.Tensor
        Modulator's driving signal (complex-valued baseband).
    param : parameter object  (struct)
        Object with physical/simulation parameters of the mzm.

        - param.Vpi: MZM's Vpi voltage [V][default: 2 V]

        - param.Vb: MZM's bias voltage [V][default: -1 V]

    Returns
    -------
    Ao : torch.Tensor
        Modulated optical field at the output of the MZM.
    """
    if param is None:
        param = []

    # check input parameters
    Vpi = getattr(param, "Vpi", 2)
    Vb = getattr(param, "Vb", -1)

    if isinstance(Ai, (float, int)):
        Ai = Ai * th.ones(u.shape, dtype=u.dtype, device=u.device)
    elif isinstance(Ai, th.Tensor):
        if Ai.shape == th.Size([]) and u.shape != th.Size([]):
            Ai = Ai * th.ones(u.shape, dtype=u.dtype, device=u.device)
        else:
            assert Ai.shape == u.shape, "Ai and u need to have the same dimensions"
    else:
        raise ValueError("Unsupported data type for Ai")

    π = th.acos(th.zeros(1, device=u.device)).item() * 2

    return Ai * th.cos(0.5 / Vpi * (u + Vb) * π)


def iqm(Ai, u, param=None):
    """
    Optical In-Phase/Quadrature Modulator (IQM).

    Parameters
    ----------
    Ai : float or np.array
        Amplitude of the optical field at the input of the IQM.
    u : th.Tensor
        Modulator's driving signal (complex-valued baseband).
    param : parameter object  (struct)
        Object with physical/simulation parameters of the mzm.

        - param.Vpi: MZM's Vpi voltage [V][default: 2 V]

        - param.VbI: I-MZM's bias voltage [V][default: -2 V]

        - param.VbQ: Q-MZM's bias voltage [V][default: -2 V]

        - param.Vphi: PM bias voltage [V][default: 1 V]

    Returns
    -------
    Ao : th.Tensor
        Modulated optical field at the output of the IQM.

    """
    if param is None:
        param = parameters()

    # check input parameters
    Vpi = getattr(param, "Vpi", 2)
    VbI = getattr(param, "VbI", -2)
    VbQ = getattr(param, "VbQ", -2)
    Vphi = getattr(param, "Vphi", 1)

    if isinstance(Ai, (float, int)):
        Ai = Ai * th.ones(u.shape, dtype=u.dtype, device=u.device)
    elif isinstance(Ai, th.Tensor):
        if Ai.shape == th.Size([]) and u.shape != th.Size([]):
            Ai = Ai * th.ones(u.shape, dtype=u.dtype, device=u.device)
        else:
            assert Ai.shape == u.shape, "Ai and u need to have the same dimensions"
    else:
        raise ValueError("Unsupported data type for Ai")

    # define parameters for the I-MZM:
    paramI = parameters()
    paramI.Vpi = Vpi
    paramI.Vb = VbI

    # define parameters for the Q-MZM:
    paramQ = parameters()
    paramQ.Vpi = Vpi
    paramQ.Vb = VbQ
    sqrt2 = np.sqrt(2)

    return mzm(Ai / sqrt2, u.real, paramI) + pm(
        mzm(Ai / sqrt2, u.imag, paramQ), Vphi * th.ones(u.shape, device=u.device), Vpi
    )


def linearFiberChannel(Ei, param):
    """
    Simulate signal propagation through a linear fiber channel.

    Parameters
    ----------
    Ei : torch.Tensor
        Input optical field.
    param : parameter object  (struct)
        Object with physical/simulation parameters of the optical channel.

        - param.L: total fiber length [km][default: 50 km]

        - param.alpha: fiber attenuation parameter [dB/km][default: 0.2 dB/km]

        - param.D: chromatic dispersion parameter [ps/nm/km][default: 16 ps/nm/km]

        - param.Fc: carrier frequency [Hz] [default: 193.1e12 Hz]

        - param.Fs: sampling frequency [Hz] [default: None]

    Returns
    -------
    Ei : torch.Tensor
        Optical field at the output of the fiber.

    """
    try:
        Fs = param.Fs
    except AttributeError:
        logg.error("Simulation sampling frequency (Fs) not provided.")

    # check input parameters
    L = getattr(param, "L", 50)
    alpha = getattr(param, "alpha", 0.2)
    D = getattr(param, "D", 16)
    Fc = getattr(param, "Fc", 193.1e12)

    c_kms = const.c / 1e3
    λ = c_kms / Fc
    α = alpha / (10 * np.log10(np.exp(1)))
    β2 = -(D * λ**2) / (2 * np.pi * c_kms)

    π = th.acos(th.zeros(1, device=Ei.device)).item() * 2

    if len(Ei.shape) == 1:
        Ei = Ei.view(len(Ei), 1)

    Nfft = Ei.shape[0]

    ω = 2 * π * Fs * fftfreq(Nfft, device=Ei.device)
    ω = ω.reshape(-1, 1)

    Nmodes = Ei.shape[1]

    Ei_fft = fft(Ei, dim=0)
    Ei_fft = Ei_fft * th.exp(-α / 2 * L + 1j * (β2 / 2) * (ω**2) * L)
    Ei = ifft(Ei_fft, dim=0)

    if Nmodes == 1:
        Ei = Ei.reshape(Ei.size())

    return Ei


def photodiode(E, paramPD=None):
    """
    Pin photodiode (PD).

    Parameters
    ----------
    E : torch.Tensor
        Input optical field.
    paramPD : parameter object (struct), optional
        Parameters of the photodiode.

        - paramPD.R: photodiode responsivity [A/W][default: 1 A/W]

        - paramPD.Tc: temperature [°C][default: 25°C]

        - paramPD.Id: dark current [A][default: 5e-9 A]

        - paramPD.RL: impedance load [Ω] [default: 50Ω]

        - paramPD.B bandwidth [Hz][default: 30e9 Hz]

        - paramPD.Fs: sampling frequency [Hz] [default: 60e9 Hz]

        - paramPD.fType: frequency response type [default: 'rect']

        - paramPD.N: number of the frequency resp. filter taps. [default: 8001]

        - paramPD.ideal: ideal PD?(i.e. no noise, no frequency resp.) [default: True]

    Returns
    -------
    ipd : torch.Tensor
          photocurrent.

    """
    if paramPD is None:
        paramPD = {}

    kB = const.Boltzmann
    q = const.elementary_charge

    # check input parameters
    R = getattr(paramPD, "R", 1)
    Tc = getattr(paramPD, "Tc", 25)
    Id = getattr(paramPD, "Id", 5e-9)
    RL = getattr(paramPD, "RL", 50)
    B = getattr(paramPD, "B", 30e9)
    Fs = getattr(paramPD, "Fs", 60e9)
    N = getattr(paramPD, "N", 8000)
    fType = getattr(paramPD, "fType", "rect")
    ideal = getattr(paramPD, "ideal", True)

    assert R > 0, "PD responsivity should be a positive scalar"
    assert Fs >= 2 * B, "Sampling frequency Fs needs to be at least twice of B."

    changeDim = False

    if len(E.shape) == 1:
        changeDim = True
        E = E.view(len(E), 1)

    ipd = R * th.sum(th.abs(E) ** 2, axis=1)  # ideal photocurrent

    if not ideal:
        Pin = th.mean(th.abs(E) ** 2)

        # shot noise
        σ2_s = 2 * q * (R * Pin.cpu().detach().numpy() + Id) * B  # shot noise variance

        # thermal noise
        T = Tc + 273.15  # temperature in Kelvin
        σ2_T = 4 * kB * T * B / RL  # thermal noise variance

        σ_T_sim = np.sqrt(Fs * (σ2_T / (2 * B)))
        σ_s_sim = np.sqrt(Fs * (σ2_s / (2 * B)))

        # add noise sources to the p-i-n receiver
        Is = th.normal(0.0, σ_s_sim, ipd.shape, device=E.device)
        It = th.normal(0.0, σ_T_sim, ipd.shape, device=E.device)

        ipd += Is + It

        # lowpass filtering
        h = lowPassFIR(B, Fs, N, typeF=fType)
        h = h / th.sum(th.abs(h))

        ipd = firFilter(h, ipd)
    else:
        pass  # return ideal photodetected current

    if changeDim:
        E = E.flatten()

    return ipd


def awgn(sig, snrdB, Fs=1, B=1, complexNoise=True):
    """
    Implement a basic AWGN channel model.

    Parameters
    ----------
    sig : torch.Tensor
        Input signal.
    snrdB : scalar
        Signal-to-noise ratio in dB.
    Fs : real scalar
        Sampling frequency. The default is 1.
    B : real scalar
        Signal bandwidth, defined as the length of the frequency interval [-B/2, B/2]. The default is 1.
    complexNoise : bool
        Generate complex-valued noise. The default is True.

    Returns
    -------
    torch.Tensor
        Input signal plus noise.

    """
    snr_lin = th.tensor(10 ** (snrdB / 10), device=sig.device)

    noiseVar = 1 / snr_lin
    σ2 = (Fs / B) * noiseVar

    if complexNoise:
        noise = gaussianComplexNoise(sig.shape, σ2, device=sig.device)
    else:
        noise = gaussianNoise(sig.shape, σ2, device=sig.device)

    return sig + noise


def manakovSSF(Ei, param):
    device = Ei.device  # Get the device of the input tensor Ei

    try:
        Fs = param.Fs
    except AttributeError:
        raise ValueError("Simulation sampling frequency (Fs) not provided.")

    # check input parameters
    param.Ltotal = getattr(param, "Ltotal", 400)
    param.Lspan = getattr(param, "Lspan", 80)
    param.hz = getattr(param, "hz", 0.5)
    param.alpha = getattr(param, "alpha", 0.2)
    param.D = getattr(param, "D", 16)
    param.gamma = getattr(param, "gamma", 1.3)
    param.Fc = getattr(param, "Fc", 193.1e12)
    param.prec = getattr(param, "prec", th.complex64)
    param.amp = getattr(param, "amp", "edfa")
    param.NF = getattr(param, "NF", 4.5)
    param.maxIter = getattr(param, "maxIter", 10)
    param.tol = getattr(param, "tol", 1e-5)
    param.nlprMethod = getattr(param, "nlprMethod", True)
    param.maxNlinPhaseRot = getattr(param, "maxNlinPhaseRot", 2e-2)
    param.prgsBar = getattr(param, "prgsBar", True)
    param.saveSpanN = getattr(param, "saveSpanN", [param.Ltotal // param.Lspan])
    param.returnParameters = getattr(param, "returnParameters", False)

    Ltotal = param.Ltotal
    Lspan = param.Lspan
    hz = param.hz
    alpha = param.alpha
    D = param.D
    gamma = param.gamma
    Fc = param.Fc
    amp = param.amp
    NF = param.NF
    prec = param.prec
    maxIter = param.maxIter
    tol = param.tol
    prgsBar = param.prgsBar
    saveSpanN = param.saveSpanN
    nlprMethod = param.nlprMethod
    maxNlinPhaseRot = param.maxNlinPhaseRot
    returnParameters = param.returnParameters

    # Convert constants to the device of the input tensor
    c_kms = th.tensor(const.c / 1e3, device=device)
    λ = c_kms / Fc
    α = th.tensor(alpha / (10 * np.log10(np.exp(1))), device=device)
    β2 = -(D * λ**2) / (2 * th.pi * c_kms)
    γ = gamma

    # edfa parameters
    paramAmp = parameters()
    paramAmp.G = alpha * Lspan
    paramAmp.NF = NF
    paramAmp.Fc = Fc
    paramAmp.Fs = Fs

    # generate frequency axis
    Nfft = Ei.shape[0]
    ω = 2 * th.pi * Fs * fftfreq(Nfft, device=device)

    Nspans = int(np.floor(Ltotal / Lspan))

    # define static part of the linear operator
    argLimOp = -(α / 2) + 1j * (β2 / 2) * (ω**2)

    if Ei.shape[1] > 1:
        argLimOp = th.tile(argLimOp, (Ei.shape[1], 1)).detach()
    else:
        argLimOp = argLimOp.reshape(1, -1)

    Ei = Ei.T

    for _ in tqdm(range(1, Nspans + 1), disable=not (prgsBar)):
        z_current = 0

        # fiber propagation steps
        while z_current < Lspan:
            Pch = th.abs(Ei[0, :]) ** 2 + th.abs(Ei[1, :]) ** 2

            phiRot = (8 / 9) * γ * Pch  # nlinPhaseRot(Ei[0, :], Ei[1, :], Pch, γ)

            if nlprMethod:
                hz_ = (
                    maxNlinPhaseRot / phiRot.max()
                    if Lspan - z_current >= maxNlinPhaseRot / phiRot.max()
                    else Lspan - z_current
                )
            elif Lspan - z_current < hz:
                hz_ = Lspan - z_current  # check that the remaining
                # distance is not less than hz (due to non-integer
                # steps/span)
            else:
                hz_ = hz

            # define the linear operator
            linOperator = th.exp(argLimOp * (hz_ / 2))

            # First linear step (frequency domain)
            Ei = ifft(fft(Ei) * linOperator)

            # Nonlinear step (time domain)
            Ei = Ei * th.exp(1j * phiRot * hz_)

            # Second linear step (frequency domain)
            Ei = ifft(fft(Ei) * linOperator)

            z_current += hz_  # update propagated distance

        # amplification step
        if amp == "edfa":
            Ei = edfa(Ei.T, paramAmp).T
        elif amp == "ideal":
            Ei = Ei * th.exp(α / 2 * Lspan)
        elif amp is None:
            Ei = Ei * th.exp(0)

    if returnParameters:
        return Ei.T, param
    else:
        return Ei.T


def nlinPhaseRot(Ex, Ey, Pch, γ):
    """
    Calculate nonlinear phase-shift per step for the Manakov SSFM.

    Parameters
    ----------
    Ex : th.Tensor
        Input optical signal field of x-polarization.
    Ey : th.Tensor
        Input optical signal field of y-polarization.
    Pch : th.Tensor
        Input optical power.
    γ : real scalar
        fiber nonlinearity coefficient.

    Returns
    -------
    th.Tensor
        nonlinear phase-shift of each sample of the signal.

    """
    return ((8 / 9) * γ * (Pch + Ex * th.conj(Ex) + Ey * th.conj(Ey)) / 2).real


def convergenceCondition(Ex_fd, Ey_fd, Ex_conv, Ey_conv):
    """
    Verify the convergence condition for the trapezoidal integration.

    Parameters
    ----------
    Ex_fd : th.Tensor
        field of x-polarization fully dispersed and rotated.
    Ey_fd : th.Tensor
        field of y-polarization fully dispersed and rotated.
    Ex_conv : th.Tensor
        field of x-polarization at the beginning of the step.
    Ey_conv : th.Tensor
        field of y-polarization at the beginning of the step.

    Returns
    -------
    scalar
        squared root of the MSE normalized by the power of the fields.

    """
    return th.sqrt(
        th.norm(Ex_fd - Ex_conv) ** 2 + th.norm(Ey_fd - Ey_conv) ** 2
    ) / th.sqrt(th.norm(Ex_conv) ** 2 + th.norm(Ey_conv) ** 2)


def edfa(Ei, param):
    """
    Implement simple EDFA model.

    Parameters
    ----------
    Ei : th.Tensor
        Input signal field.
    param : parameter object (struct), optional
        Parameters of the edfa.

        - param.G : amplifier gain in dB. The default is 20.
        - param.NF : EDFA noise figure in dB. The default is 4.5.
        - param.Fc : central optical frequency. The default is 193.1e12.
        - param.Fs : sampling frequency in samples/second.

    Returns
    -------
    Ei : th.Tensor
        Amplified noisy optical signal.

    """
    device = Ei.device  # Get the device of the input tensor Ei

    try:
        Fs = param.Fs
    except AttributeError:
        raise ValueError("Simulation sampling frequency (Fs) not provided.")

    # check input parameters
    G = getattr(param, "G", 20)
    NF = getattr(param, "NF", 4.5)
    Fc = getattr(param, "Fc", 193.1e12)
    prec = getattr(param, "prec", th.complex64)

    assert G > 0, "EDFA gain should be a positive scalar"
    assert NF >= 3, "The minimal EDFA noise figure is 3 dB"

    NF_lin = 10 ** (NF / 10)
    G_lin = th.tensor(10 ** (G / 10), device=device)
    nsp = (G_lin * NF_lin - 1) / (2 * (G_lin - 1))

    # ASE noise power calculation:
    # Ref. Eq.(54) of R. -J. Essiambre, et al, "Capacity Limits of Optical Fiber
    # Networks," in Journal of Lightwave Technology, vol. 28, no. 4,
    # pp. 662-701, Feb.15, 2010, doi: 10.1109/JLT.2009.2039464.

    N_ase = (G_lin - 1) * nsp * const.h * Fc
    p_noise = N_ase * Fs

    noise = th.randn_like(Ei, dtype=prec, device=device) * th.sqrt(
        p_noise / 2
    ) + 1j * th.randn_like(Ei, dtype=prec, device=device) * th.sqrt(p_noise / 2)

    return Ei * th.sqrt(G_lin) + noise


def kkPhaseRetrieval(Amp, Fs, device="cpu"):
    """
    Calculate the phase of a signal from its amplitude using the Kramers-Kronig Phase Retrieval.

    Parameters
    ----------
    Amp : torch.Tensor
        Signal amplitude values.
    Fs : float
        Signal's sampling frequency.

    Returns
    -------
    torch.Tensor
        Calculated signal's phase values.

    Notes
    -----
    The function applies the Kramers-Kronig transformation to obtain the phase
    of the input signal in the time domain from its amplitude in the frequency domain.

    The input signal can be single-mode or multi-mode, and the function handles
    both cases appropriately.
    """
    Nsamples = Amp.shape[0]

    try:
        Nmodes = Amp.shape[1]
    except IndexError:
        Nmodes = 1
        Amp = Amp.reshape(Nsamples, Nmodes)

    xf = fftfreq(Nsamples, Fs, device=device)
    xf = fftshift(xf)
    xf = xf.reshape(Nsamples, Nmodes)

    # Calculate phase in the frequency domain
    phiOmega = 1j * th.sign(xf) * fft(th.log(Amp), dim=0)

    # Pass phase to the time domain
    phiTime = ifft(phiOmega, dim=0).real

    if Nmodes == 1:
        phiTime = phiTime.reshape(
            -1,
        )

    return phiTime


def adc(Ei, param):
    """
    Analog-to-digital converter (ADC) model.

    Parameters
    ----------
    Ei : th.Tensor
        Input signal.
    param : core.parameter
        Resampling parameters:
            - param.Fs_in  : sampling frequency of the input signal [default: 1 sample/s]
            - param.Fs_out : sampling frequency of the output signal [default: 1 sample/s]
            - param.jitter_rms : root mean square (RMS) value of the jitter in seconds [default: 0 s]
            - param.nBits : number of bits used for quantization [default: 8 bits]
            - param.Vmax : maximum value for the ADC's full-scale range [default: 1V]
            - param.Vmin : minimum value for the ADC's full-scale range [default: -1V]
            - param.AAF : flag indicating whether to use anti-aliasing filters [default: True]
            - param.N : number of taps of the anti-aliasing filters [default: 201]

    Returns
    -------
    Eo : th.Tensor
        Resampled and quantized signal.

    """
    # Check and set default values for input parameters
    param.Fs_in = getattr(param, "Fs_in", 1)
    param.Fs_out = getattr(param, "Fs_out", 1)
    param.jitter_rms = getattr(param, "jitter_rms", 0)
    param.nBits = getattr(param, "nBits", 8)
    param.Vmax = getattr(param, "Vmax", 1)
    param.Vmin = getattr(param, "Vmin", -1)
    param.AAF = getattr(param, "AAF", True)
    param.N = getattr(param, "N", 201)

    # Extract individual parameters for ease of use
    Fs_in = param.Fs_in
    Fs_out = param.Fs_out
    jitter_rms = param.jitter_rms
    nBits = param.nBits
    Vmax = param.Vmax
    Vmin = param.Vmin
    AAF = param.AAF
    N = param.N

    # Apply anti-aliasing filters if AAF is enabled
    if AAF:
        # Anti-aliasing filters:
        Ntaps = min(Ei.shape[0], N)
        hi = lowPassFIR(param.Fs_out / 2, param.Fs_in, Ntaps, typeF="rect").to(
            Ei.device
        )
        ho = lowPassFIR(param.Fs_out / 2, param.Fs_out, Ntaps, typeF="rect").to(
            Ei.device
        )

        Ei = firFilter(hi, Ei)

    # Reshape the input signal if needed to handle single-dimensional inputs
    if len(Ei.shape) == 1:
        Ei = Ei.reshape(len(Ei), 1)
    # Get the number of modes (columns) in the input signal
    nModes = Ei.shape[1]

    print(Ei.shape)

    if th.is_complex(Ei):
        # Signal interpolation to the ADC's sampling frequency
        Eo_real = clockSamplingInterp(Ei.real, Fs_in, Fs_out, jitter_rms)
        Eo_imag = clockSamplingInterp(Ei.imag, Fs_in, Fs_out, jitter_rms)
        Eo = th.stack((Eo_real, Eo_imag), dim=-1)

        # Uniform quantization of the signal according to the number of bits of the ADC
        Eo_real_quantized = quantizer(Eo_real, nBits, Vmax, Vmin)
        Eo_imag_quantized = quantizer(Eo_imag, nBits, Vmax, Vmin)
        Eo = th.stack((Eo_real_quantized, Eo_imag_quantized), dim=-1)
    else:
        # Signal interpolation to the ADC's sampling frequency
        Eo = clockSamplingInterp(Ei, Fs_in, Fs_out, jitter_rms)

        # Uniform quantization of the signal according to the number of bits of the ADC
        Eo = quantizer(Eo, nBits, Vmax, Vmin)

    # Apply anti-aliasing filters to the output if AAF is enabled
    if AAF:
        Eo_real = firFilter(ho, Eo.real)
        Eo_imag = firFilter(ho, Eo.imag)
        Eo = th.stack((Eo_real, Eo_imag), dim=-1)

    return Eo


def powerAmplifier(x, g=16, σ=1.1, c=1.9, α=-345, β=0.17, q=4):
    """
    Power amplifier (PA) model.

    # g=4.65, σ=0.81, c=0.58, α=2560, β=0.114, p=2.4, q=2.3

    Parameters
    ----------
    x : Tensor
        Input signal.
    g : float, optional
        Gain parameter (default is 4.65).
    σ : float, optional
        Shape parameter (default is 0.81).
    c : float, optional
        Offset parameter (default is 0.58).
    α : float, optional
        Scaling parameter (default is 2560).
    β : float, optional
        Scaling parameter (default is 0.114).
    p : float, optional
        Exponent parameter (default is 2.4).
    q : float, optional
        Exponent parameter (default is 2.3).

    Returns
    -------
    Tensor
        Output signal.

    Notes
    -----
    This function models the behavior of a power amplifier.

    The output signal is calculated as follows:
    abs_y = g * abs_x / (1 + g * abs_x / c) ** (1 / (2 * σ))
    phi_y = α * abs_x ** p / (1 + (abs_x / β) ** q)
    y = abs_y * exp(1j * (phi_x + phi_y)), where phi_x is the phase of x.

    References
    ----------
    Put any relevant references or citations here.
    """
    pi = 3.14159265359
    abs_x = th.abs(x)
    phi_x = th.angle(x)

    abs_y = g * abs_x / (1 + th.abs(g * abs_x / c)**(2 * σ) ) ** (1 / (2 * σ))
    phi_y = α * abs_x**q / (1 + (abs_x / β) ** q) * (pi / 180)

    return abs_y * th.exp(1j * (phi_x + phi_y))
