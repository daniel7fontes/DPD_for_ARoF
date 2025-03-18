"""Metrics for signal and performance characterization."""
import logging as logg

import torch as th
from optic_private.torchModulation import (
    grayMapping,
    modulateGray,
    demodulateGray,
    detector,
    dec2bitarray,
    demap,
    getBitMap,
)
from optic_private.torchUtils import signal_power
from optic_private.torchDSP import pnorm


def fastBERcalc(rx, tx, M, constType, px=None, bitMap=None, SNR=None):
    """
    Monte Carlo BER/SER/SNR calculation.

    Parameters
    ----------
    rx : torch.Tensor
        Received symbol sequence.
    tx : torch.Tensor
        Transmitted symbol sequence.
    M : int
        Modulation order.
    constType : string
        Modulation type: 'qam', 'psk', 'pam' or 'ook'.

    Returns
    -------
    BER : torch.Tensor
        Bit-error-rate.
    SER : torch.Tensor
        Symbol-error-rate.
    SNR : torch.Tensor
        Estimated SNR from the received constellation.

    """
    device = rx.device  # Get the device of the input rx

    if M != 2 and constType == "ook":
        logg.warn("OOK has only 2 symbols, but M != 2. Changing M to 2.")
        M = 2
    # Constellation parameters
    constSymb = grayMapping(M, constType, device)

    if px is None:
        # If px was not specified assume uniform distribution
        px = th.ones(M, device=device) / M

    Es = th.sum(th.abs(constSymb) ** 2 * px)
    constSymb = constSymb / th.sqrt(Es)  # normalize average constellation energy to 1

    # We want all the signal sequences to be disposed in columns:
    # try:
    #     if rx.shape[1] > rx.shape[0]:
    #         rx = rx.T
    # except IndexError:
    #     rx = rx.view(len(rx), 1)
    # try:
    #     if tx.shape[1] > tx.shape[0]:
    #         tx = tx.T
    # except IndexError:
    
    # tx = tx.view(len(tx), 1)
    try:
        nModes = tx.shape[1]  # Number of signal modes
    except:
        tx = tx.view(len(tx), 1)
        rx = rx.view(len(rx), 1)
        nModes = tx.shape[1]  # Number of signal modes

    if bitMap is None:
        # get bit to symbol mapping
        bitMap = getBitMap(constSymb)

    BER = th.zeros(nModes, device=device)
    SER = th.zeros(nModes, device=device)
    SNRest = th.zeros(nModes, device=device)

    # Pre-processing
    for k in range(nModes):
        if constType in ["qam", "psk", "apsk"]:
            # find (possible) phase ambiguity (rotation in the complex plane)
            rot = th.mean(tx[:, k] / rx[:, k])
        else:
            rot = 1

    for k in range(nModes):
        # Estimate SNR of the received constellation
        SNRest[k] = 10 * th.log10(
            signal_power(tx[:, k]) / signal_power(rot * rx[:, k] - tx[:, k])
        )
        σ2 = 1 / 10 ** (SNRest[k] / 10) if SNR is None else 1 / 10 ** (SNR / 10)

        ind_rx = detector(rot * rx[:, k], constSymb, σ2=σ2, px=px, rule="MAP")
        ind_tx = detector(pnorm(tx[:, k]), constSymb)

        brx = demap(ind_rx, bitMap)
        btx = demap(ind_tx, bitMap)

        err = th.logical_xor(brx, btx)
        BER[k] = th.mean(err.float())
        SER[k] = th.mean(
            (
                th.sum(
                    err.float().view(-1, int(th.log2(th.tensor(M, device=device)))),
                    dim=1,
                )
                > 0
            ).float()
        )

    return BER, SER, SNRest


# @th.jit.script
# def calcLLR(rxSymb, σ2, constSymb, bitMap, px):
#     """
#     LLR calculation (circular AGWN channel).

#     Parameters
#     ----------
#     rxSymb : torch.Tensor
#         Received symbol sequence.
#     σ2 : scalar
#         Noise variance.
#     constSymb : (M, 1) torch.Tensor
#         Constellation symbols.
#     px : (M, 1) torch.Tensor
#         Prior symbol probabilities.
#     bitMap : (M, log2(M)) torch.Tensor
#         Bit-to-symbol mapping.

#     Returns
#     -------
#     LLRs : torch.Tensor
#         Sequence of calculated LLRs.

#     """
#     M = len(constSymb)
#     b = int(th.log2(th.tensor(M, device=rxSymb.device)).item())

#     LLRs = th.zeros(len(rxSymb) * b, device=rxSymb.device)

#     for i in range(len(rxSymb)):
#         prob = th.exp((-th.abs(rxSymb[i] - constSymb) ** 2) / σ2) * px

#         for indBit in range(b):
#             p0 = th.sum(prob[bitMap[:, indBit] == 0])
#             p1 = th.sum(prob[bitMap[:, indBit] == 1])

#             LLRs[i * b + indBit] = th.log(p0) - th.log(p1)
#     return LLRs


def calcLLR(rxSymb, σ2, constSymb, bitMap, px):
    """
    Vectorized LLR calculation (circular AGWN channel).

    Parameters
    ----------
    rxSymb : torch.Tensor
        Received symbol sequence.
    σ2 : scalar
        Noise variance.
    constSymb : (M, 1) torch.Tensor
        Constellation symbols.
    px : (M, 1) torch.Tensor
        Prior symbol probabilities.
    bitMap : (M, log2(M)) torch.Tensor
        Bit-to-symbol mapping.

    Returns
    -------
    LLRs : torch.Tensor
        Sequence of calculated LLRs.

    """
    M = len(constSymb)
    b = int(th.log2(th.tensor(M, device=rxSymb.device)).item())

    # Calculate the Euclidean distance between rxSymb and constSymb
    distances = th.abs(rxSymb.view(-1, 1) - constSymb.view(1, -1)) ** 2

    # Calculate the probabilities using distances and σ2
    prob = th.exp(-distances / σ2) * px

    # Split bitMap into two masks for 0 and 1 bits
    mask0 = bitMap == 0
    mask1 = bitMap == 1

    # Calculate p0 and p1 using masks and probabilities
    p0 = th.matmul(prob, mask0.float())
    p1 = th.matmul(prob, mask1.float())

    # Compute LLRs using p0 and p1
    LLRs = th.log(p0 + 1e-10) - th.log(p1 + 1e-10)

    # Reshape LLRs to the desired shape
    LLRs = LLRs.view(-1)

    return LLRs


def monteCarloGMI(rx, tx, M=16, constType="qam", constSymb=None, px=None, bitMap=None):
    """
    Monte Carlo based generalized mutual information (GMI) estimation.

    Parameters
    ----------
    rx : torch.Tensor
        Received symbol sequence.
    tx : torch.Tensor
        Transmitted symbol sequence.
    M : int, optional
        Modulation order (default is 16).
    constType : string, optional
        Modulation type: 'qam' or 'psk' (default is 'qam').
    constSymb : torch.Tensor, optional
        Constellation symbols (default is None).
    px : torch.Tensor, optional
        Prior symbol probabilities. The default is [].
    bitMap : torch.Tensor, optional
        Bit-to-symbol mapping. The default is None.

    Returns
    -------
    GMI : torch.Tensor
        Generalized mutual information values.
    NGMI : torch.Tensor
        Normalized generalized mutual information.

    """
    if px is None:
        px = []

    device = rx.device
    b = th.log2(th.tensor(M, device=device)).int()

    if constSymb is None:
        # get constellation
        constSymb = grayMapping(M, constType, device)

    if bitMap is None:
        indMap = detector(constSymb, constSymb)
        bitMap = dec2bitarray(indMap, b)
        bitMap = bitMap.reshape(-1, b)

    # We want all the signal sequences to be disposed in columns:
    try:
        if rx.shape[1] > rx.shape[0]:
            rx = rx.T
    except IndexError:
        rx = rx.reshape(len(rx), 1)
    try:
        if tx.shape[1] > tx.shape[0]:
            tx = tx.T
    except IndexError:
        tx = tx.reshape(len(tx), 1)

    nModes = tx.shape[1]  # number of sinal modes
    GMI = th.zeros(nModes, device=device)
    NGMI = th.zeros(nModes, device=device)

    if len(px) == 0:  # if px is not defined, assume uniform distribution
        px = th.ones(constSymb.shape, device=device) / M
    # Normalize constellation
    Es = th.sum(th.abs(constSymb) ** 2 * px)
    constSymb = constSymb / th.sqrt(Es)

    # Calculate source entropy
    H = th.sum(-px * th.log2(px))

    for k in range(nModes):
        # correct (possible) phase ambiguity with rotation of rx
        rot = th.mean(tx[:, k] / rx[:, k]) if constType in ["qam", "psk"] else 1.0

        # set the noise variance
        σ2 = th.var(rot * rx[:, k] - tx[:, k], axis=0)

        # demodulate transmitted symbol sequence
        ind_tx = detector(tx[:, k], constSymb)
        btx = demap(ind_tx, bitMap)

        # soft demodulation of the received symbols
        LLRs = calcLLR(rot * rx[:, k], σ2, constSymb, bitMap, px)

        # LLR clipping
        LLRs[LLRs == th.inf] = 500
        LLRs[LLRs == -th.inf] = -500

        # Compute bitwise MIs and their sum
        MIperBitPosition = th.zeros(b, device=device)

        for n in range(b):
            MIperBitPosition[n] = H / b - th.mean(
                th.log2(1 + th.exp((2 * btx[n::b] - 1) * LLRs[n::b]))
            )
        GMI[k] = th.sum(MIperBitPosition)
        NGMI[k] = GMI[k] / H
    return GMI, NGMI


def calcMI(rx, tx, σ2, constSymb, pX):
    """
    Mutual information (MI) calculation (circular AGWN channel).

    Parameters
    ----------
    rx : torch.Tensor
        Received symbol sequence.
    tx : torch.Tensor
        Transmitted symbol sequence.
    σ2 : scalar
        Noise variance.
    constSymb : torch.Tensor
        Constellation symbols.
    pX : torch.Tensor
        Probability mass function (p.m.f.) of the constellation symbols.

    Returns
    -------
    torch.Tensor
        Estimated mutual information.

    """
    # H_XgY = th.zeros(1, dtype=th.float64, device=rx.device)
    H_X = th.sum(-pX * th.log2(pX))

    indSymb = th.argmin(th.abs(tx.view(-1, 1) - constSymb), dim=1)

    log2_pYgX = (
        -(1 / σ2)
        * th.abs(rx - tx) ** 2
        * th.log2(th.exp(th.tensor(1.0, device=rx.device)))
    )  # log2 p(Y|X=x)

    pXY = (
        th.exp(-(1 / σ2) * th.abs(rx.view(-1, 1) - constSymb) ** 2) * pX
    )  # p(Y,X) = p(Y|X)*p(X)

    pY = th.sum(pXY, dim=1)  # p(Y) = Σ_x p(Y,X)

    log2_pXgY = log2_pYgX + th.log2(pX[indSymb]) - th.log2(pY)

    H_XgY = th.mean(-log2_pXgY)

    return H_X - H_XgY


def monteCarloMI(rx, tx, M=16, constType="qam", constSymb=None, px=None):
    """
    Monte Carlo based mutual information (MI) estimation.

    Parameters
    ----------
    rx : torch.Tensor
        Received symbol sequence.
    tx : torch.Tensor
        Transmitted symbol sequence.
    M : int
        Modulation order.
    constType : string
        Modulation type: 'qam' or 'psk'
    px : torch.Tensor
        p.m.f. of the constellation symbols. The default is [].

    Returns
    -------
    torch.Tensor
        Estimated MI values.

    """
    if px is None:
        px = 1 / M * th.ones(M, device=rx.device)  # assume uniform distribution

    # We want all the signal sequences to be disposed in columns:
    try:
        if rx.shape[1] > rx.shape[0]:
            rx = rx.T
    except IndexError:
        rx = rx.reshape(len(rx), 1)
    try:
        if tx.shape[1] > tx.shape[0]:
            tx = tx.T
    except IndexError:
        tx = tx.reshape(len(tx), 1)

    if constSymb is None:
        # get constellation
        constSymb = grayMapping(M, constType, rx.device)

    Es = th.sum(th.abs(constSymb) ** 2 * px)
    constSymb = constSymb / th.sqrt(Es)

    nModes = rx.shape[1]  # number of signal modes
    MI = th.zeros(nModes, dtype=th.float64, device=rx.device)

    for k in range(nModes):
        # correct (possible) phase ambiguity with rotation of rx
        rot = th.mean(tx[:, k] / rx[:, k]) if constType in ["qam", "psk"] else 1.0

        # set the noise variance
        σ2 = th.var(rot * rx[:, k] - tx[:, k], axis=0)

        MI[k] = calcMI(rot * rx[:, k], tx[:, k], σ2, constSymb, px)

    return MI

def papr(signal):
    """
    Calculate the Peak-to-Average Power Ratio (PAPR) of a signal in dB.

    Parameters
    ----------
    signal : torch.Tensor
        Input signal.

    Returns
    -------
    float
        Peak-to-Average Power Ratio (PAPR) of the signal in dB.

    Notes
    -----
    The Peak-to-Average Power Ratio (PAPR) is a measure of how much the peak
    power of a signal exceeds its average power. It is often used in
    communication systems to characterize the linearity requirements of
    amplifiers.

    The PAPR is calculated as the ratio of the peak power to the average power
    of the signal, expressed in decibels (dB).
    """
    peak_power = th.max(th.abs(signal)) ** 2
    average_power = th.mean(signal ** 2)
    papr_ratio = peak_power / average_power
    papr_dB = 10 * th.log10(papr_ratio)
    
    return papr_dB.item()


def calcACLR(Psd, freqs, B):
    """
    Calculate the Adjacent Channel Leakage Ratio (ACLR).

    Parameters
    ----------
    Psd : numpy.ndarray
        Power spectral density (Psd) values.
    freqs : numpy.ndarray
        Frequency values corresponding to the Psd array.
    B : float
        Bandwidth of the adjacent channel.

    Returns
    -------
    float
        Calculated ACLR value in decibels (dB).

    Notes
    -----
    The ACLR measures the power leakage from one channel into an adjacent channel. 
    It is calculated as the ratio of the power outside of the adjacent channel 
    bandwidth to the power inside the adjacent channel bandwidth.

    The function computes the ACLR using the following steps:
    1. Compute the frequency resolution (df) as the difference between consecutive frequency values.
    2. Calculate the total power inside and outside the adjacent channel bandwidth.
    3. Compute the ratio of power outside to power inside the adjacent channel bandwidth.
    4. Convert the ratio to decibels (dB) using the `lin2dB` function.

    References
    ----------
    [1] 3GPP TS 36.101: "User Equipment (UE) radio transmission and reception."
        https://www.3gpp.org/ftp/Specs/html-info/36101.htm

    [2] 3GPP TS 38.104: "Base Station (BS) radio transmission and reception."
        https://www.3gpp.org/ftp/Specs/html-info/38104.htm
    """
    df = freqs[1] - freqs[0]
    Pin = np.sum(Psd[freqs >= -B] * df) - np.sum(Psd[freqs >= B] * df)
    Pout = np.sum(Psd[freqs <= -B] * df) + np.sum(Psd[freqs >= B] * df)

    return lin2dB(Pout / Pin)