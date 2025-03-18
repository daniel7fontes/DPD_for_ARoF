import logging as logg

import torch as th
import numpy as np
from optic_private.torchModulation import grayMapping
import torch.fft as fft
from scipy.constants import c
from tqdm.notebook import tqdm


def rde(x, dx=None, paramEq=None):
    if dx is None:
        dx = []
    if paramEq is None:
        paramEq = []

    # check input parameters
    numIter = th.tensor(getattr(paramEq, "numIter", 1), device=x.device)
    nTaps = th.tensor(getattr(paramEq, "nTaps", 15), device=x.device)
    mu = th.tensor(getattr(paramEq, "mu", [1e-3]), device=x.device)
    SpS = th.tensor(getattr(paramEq, "SpS", 2), device=x.device)
    H = th.tensor(getattr(paramEq, "H", []), device=x.device)
    L = th.tensor(getattr(paramEq, "L", []), device=x.device)
    Hiter = th.tensor(getattr(paramEq, "Hiter", []), device=x.device)
    storeCoeff = getattr(paramEq, "storeCoeff", False)
    constType = getattr(paramEq, "constType", "qam")
    M = th.tensor(getattr(paramEq, "M", 4), device=x.device)
    px = getattr(paramEq, "px", th.ones(M, device=x.device) / M)
    prgsBar = getattr(paramEq, "prgsBar", True)

    # We want all the signal sequences to be disposed in columns:
    if not len(dx):
        dx = x.clone()
    try:
        if x.shape[1] > x.shape[0]:
            x = x.mT
    except IndexError:
        x = x.view(len(x), 1)
    try:
        if dx.shape[1] > dx.shape[0]:
            dx = dx.mT
    except IndexError:
        dx = dx.view(len(dx), 1)
    nModes = x.shape[1]  # number of signal modes (order of the MIMO equalizer)

    Lpad = int(th.floor(nTaps / 2))
    zeroPad = th.zeros((Lpad, nModes), dtype=th.complex64, device=x.device)
    x = th.cat(
        (zeroPad, x, zeroPad), dim=0
    )  # pad start and end of the signal with zeros

    # Defining training parameters:
    constSymb = grayMapping(M, constType, x.device)  # constellation
    Es = th.sum(th.abs(constSymb) ** 2 * px)
    constSymb = constSymb / th.sqrt(Es)  # normalized constellation symbols

    totalNumSymb = int(th.fix((len(x) - nTaps) / SpS + 1))

    if not L.numel():  # if L is not defined
        L = totalNumSymb
        # Length of the output (1 sample/symbol) of the training section
    if not H.numel():  # if H is not defined
        H = th.zeros((nModes**2, nTaps), dtype=th.complex64, device=x.device)

        for initH in range(nModes):  # initialize filters' taps
            H[
                initH + initH * nModes,
                int(th.floor(th.tensor(H.shape[1] / 2, device=x.device))),
            ] = 1  # Central spike initialization

    yEq = th.zeros((totalNumSymb, x.shape[1]), dtype=th.complex64, device=x.device)
    errSq = th.zeros((totalNumSymb, x.shape[1]), device=x.device).mT

    Rrde = th.unique(th.abs(constSymb))

    # allocate variables
    nModes = x.shape[1]
    indTaps = th.arange(0, nTaps, device=x.device)
    indMode = th.arange(0, nModes, device=x.device)

    yEq = x[:L].clone()
    yEq[:] = th.nan
    outEq = th.zeros((nModes, 1), dtype=th.complex64, device=x.device)
    decidedR = th.zeros(outEq.shape, dtype=th.complex64, device=x.device)

    for ind in range(L):
        indIn = indTaps + ind * SpS  # simplify indexing and improve speed

        # pass signal sequence through the equalizer:
        for N in range(nModes):
            inEq = x[indIn, N].view(
                len(indIn), 1
            )  # slice input coming from the Nth mode
            outEq += H[indMode + N * nModes, :].mm(
                inEq
            )  # add contribution from the Nth mode to the equalizer's output
        yEq[ind, :] = outEq.view(1, -1)

        outEq = outEq.mT

        for k in range(nModes):
            indR = th.argmin(th.abs(Rrde - th.abs(outEq[0, k])))
            decidedR[0, k] = Rrde[indR]
        err = (
            decidedR**2 - th.abs(outEq) ** 2
        )  # calculate output error for the RDE algorithm

        prodErrOut = th.diag(err[0]) @ th.diag(outEq[0])  # define diagonal matrix

        # update equalizer taps
        for N in range(nModes):
            indUpdTaps = indMode + N * nModes  # simplify indexing
            inAdapt = x[indIn, N].T
            inAdaptPar = (
                inAdapt.repeat(nModes).reshape(len(inAdapt), -1).T
            )  # expand input to parallelize tap adaptation

            H[indUpdTaps, :] += (
                mu * prodErrOut @ th.conj(inAdaptPar)
            )  # gradient descent update

        if not (ind % 1000):
            print(th.abs(err) ** 2)

        errSq[:, ind] = th.abs(err) ** 2

    return yEq


def mimoAdaptEqualizer(x, dx=None, paramEq=None):
    """
    N-by-N MIMO adaptive equalizer.

    Algorithms available: 'cma', 'rde', 'nlms', 'dd-lms', 'da-rde', 'rls', 'dd-rls', 'static'.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    dx : torch.Tensor, optional
        Synchronized exact symbol sequence corresponding to the received input tensor x.
    paramEq : object, optional
        Parameter object containing the following attributes:

        - numIter : int, number of pre-convergence iterations (default: 1)
        - nTaps : int, number of filter taps (default: 15)
        - mu : float or list of floats, step size parameter(s) (default: [1e-3])
        - lambdaRLS : float, RLS forgetting factor (default: 0.99)
        - SpS : int, samples per symbol (default: 2)
        - H : torch.Tensor, coefficient matrix (default: [])
        - L : int or list of ints, length of the output of the training section (default: [])
        - Hiter : list, history of coefficient matrices (default: [])
        - storeCoeff : bool, flag indicating whether to store coefficient matrices (default: False)
        - alg : str or list of strs, specifying the equalizer algorithm(s) (default: ['nlms'])
        - constType : str, constellation type (default: 'qam')
        - M : int, modulation order (default: 4)
        - prgsBar : bool, flag indicating whether to display a progress bar (default: True)

    Returns
    -------
    torch.Tensor
        Equalized output tensor.
    torch.Tensor
        Coefficient matrix.
    torch.Tensor
        Squared absolute error tensor.
    list
        History of coefficient matrices.
    """
    if dx is None:
        dx = []
    if paramEq is None:
        paramEq = []

    # check input parameters
    numIter = th.tensor(getattr(paramEq, "numIter", 1), device=x.device)
    nTaps = th.tensor(getattr(paramEq, "nTaps", 15), device=x.device)
    mu = th.tensor(getattr(paramEq, "mu", [1e-3]), device=x.device)
    lambdaRLS = th.tensor(getattr(paramEq, "lambdaRLS", 0.99), device=x.device)
    SpS = th.tensor(getattr(paramEq, "SpS", 2), device=x.device)
    H = th.tensor(getattr(paramEq, "H", []), device=x.device)
    L = th.tensor(getattr(paramEq, "L", []), device=x.device)
    Hiter = th.tensor(getattr(paramEq, "Hiter", []), device=x.device)
    storeCoeff = getattr(paramEq, "storeCoeff", False)
    alg = getattr(paramEq, "alg", ["nlms"])
    constType = getattr(paramEq, "constType", "qam")
    M = getattr(paramEq, "M", 4)
    px = getattr(paramEq, "px", th.ones(M, device=x.device) / M)
    prgsBar = getattr(paramEq, "prgsBar", True)

    # We want all the signal sequences to be disposed in columns:
    if not len(dx):
        dx = x.clone()
    try:
        if x.shape[1] > x.shape[0]:
            x = x.mT
    except IndexError:
        x = x.view(len(x), 1)
    try:
        if dx.shape[1] > dx.shape[0]:
            dx = dx.mT
    except IndexError:
        dx = dx.view(len(dx), 1)
    nModes = x.shape[1]  # number of signal modes (order of the MIMO equalizer)

    Lpad = int(th.floor(nTaps / 2))
    zeroPad = th.zeros((Lpad, nModes), dtype=th.complex64, device=x.device)
    x = th.cat(
        (zeroPad, x, zeroPad), dim=0
    )  # pad start and end of the signal with zeros

    # Defining training parameters:
    constSymb = grayMapping(M, constType, x.device)  # constellation
    Es = th.sum(th.abs(constSymb) ** 2 * px)
    constSymb = constSymb / th.sqrt(Es)  # normalized constellation symbols

    totalNumSymb = int(th.fix((len(x) - nTaps) / SpS + 1))

    if not L.numel():  # if L is not defined
        L = [
            totalNumSymb
        ]  # Length of the output (1 sample/symbol) of the training section
    if not H.numel():  # if H is not defined
        H = th.zeros((nModes**2, nTaps), dtype=th.complex64, device=x.device)

        for initH in range(nModes):  # initialize filters' taps
            H[
                initH + initH * nModes,
                int(th.floor(th.tensor(H.shape[1] / 2, device=x.device))),
            ] = 1  # Central spike initialization
            
    logg.info(f"Running adaptive equalizer...")
    # Equalizer training:
    if isinstance(alg, list):
        yEq = th.zeros((totalNumSymb, x.shape[1]), dtype=th.complex64, device=x.device)
        errSq = th.zeros((totalNumSymb, x.shape[1]), device=x.device).mT

        nStart = 0
        for indstage, runAlg in enumerate(alg):
            logg.info(f"{runAlg} - training stage #{indstage}")

            nEnd = nStart + L[indstage]

            if indstage == 0:
                for indIter in tqdm(range(numIter), disable=not (prgsBar)):
                    logg.info(f"{runAlg} pre-convergence training iteration #{indIter}")
                    yEq[nStart:nEnd, :], H, errSq[:, nStart:nEnd], Hiter = coreAdaptEq(
                        x[nStart * SpS : (nEnd + 2 * Lpad) * SpS, :],
                        dx[nStart:nEnd, :],
                        SpS,
                        H,
                        L[indstage],
                        mu[indstage],
                        lambdaRLS,
                        nTaps,
                        storeCoeff,
                        runAlg,
                        constSymb,
                        px,
                    )
                    logg.info(f"{runAlg} MSE = {th.nanmean(errSq[:, nStart:nEnd]):.6f}")
            else:
                yEq[nStart:nEnd, :], H, errSq[:, nStart:nEnd], Hiter = coreAdaptEq(
                    x[nStart * SpS : (nEnd + 2 * Lpad) * SpS, :],
                    dx[nStart:nEnd, :],
                    SpS,
                    H,
                    L[indstage],
                    mu[indstage],
                    lambdaRLS,
                    nTaps,
                    storeCoeff,
                    runAlg,
                    constSymb,
                    px,
                )
                logg.info(f"{runAlg} MSE = {th.nanmean(errSq[:, nStart:nEnd]):.6f}")
            nStart = nEnd
    else:
        for indIter in tqdm(range(numIter), disable=not (prgsBar)):
            logg.info(f"{alg} training iteration #{indIter}")
            yEq,

            H, errSq, Hiter = coreAdaptEq(
                x, dx, SpS, H, L, mu, nTaps, storeCoeff, alg, constSymb
            )
            logg.info(f"{alg} MSE = {th.nanmean(errSq):.6f}")

    return yEq, H, errSq, Hiter


def coreAdaptEq(x, dx, SpS, H, L, mu, lambdaRLS, nTaps, storeCoeff, alg, constSymb, px):
    """
    Adaptive equalizer core processing function

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    dx : torch.Tensor
        Exact constellation radius tensor.
    SpS : int
        Samples per symbol.
    H : torch.Tensor
        Coefficient matrix.
    L : int
        Length of the output.
    mu : float
        Step size parameter.
    lambdaRLS : float
        RLS forgetting factor.
    nTaps : int
        Number of taps.
    storeCoeff : bool
        Flag indicating whether to store coefficient tensors.
    alg : str
        Equalizer algorithm.
    constSymb : torch.Tensor
        Constellation symbols.

    Returns
    -------
    torch.Tensor
        Equalized output tensor.
    torch.Tensor
        Coefficient tensor.
    torch.Tensor
        Squared absolute error tensor.
    torch.Tensor
        History of coefficient tensors.

    """
    # allocate variables
    nModes = x.shape[1]
    indTaps = th.arange(0, nTaps, device=x.device)
    indMode = th.arange(0, nModes, device=x.device)

    errSq = th.empty((nModes, L), device=x.device)
    yEq = x[:L].clone()
    yEq[:] = th.nan
    outEq = th.zeros((nModes, 1), dtype=th.complex64, device=x.device)

    if storeCoeff:
        Hiter = th.zeros((nModes**2, nTaps, L), dtype=th.complex64, device=x.device)
    else:
        Hiter = th.zeros((nModes**2, nTaps, 1), dtype=th.complex64, device=x.device)
    if alg == "rls":
        Sd = th.eye(nTaps, dtype=th.complex128, device=x.device)
        a = Sd.clone()
        for _ in range(nTaps - 1):
            Sd = th.cat((Sd, a), dim=0)
    # Radii cma, rde
    Rcma = (
        th.mean(th.abs(constSymb) ** 4) / th.mean(th.abs(constSymb) ** 2)
    ) * th.ones((1, nModes), dtype=th.complex64, device=x.device)
    Rrde = th.unique(th.abs(constSymb))

    for ind in range(L):
        outEq[:] = 0

        indIn = indTaps + ind * SpS  # simplify indexing and improve speed

        # print('ind: ', ind)
        # print('indIn: ', indIn)

        # pass signal sequence through the equalizer:
        for N in range(nModes):
            inEq = x[indIn, N].view(
                len(indIn), 1
            )  # slice input coming from the Nth mode
            outEq += H[indMode + N * nModes, :].mm(
                inEq
            )  # add contribution from the Nth mode to the equalizer's output
        yEq[ind, :] = outEq.view(1, -1)

        # inEq = x[indIn, :nModes].view(len(indIn), nModes)  # Slice input for all modes
        # outEq += th.matmul(H[indMode, :], inEq)  # Add contributions from all modes to the equalizer's output
        # yEq[ind, :] = outEq.view(1, -1)

        # update equalizer taps according to the specified algorithm and save squared error:
        if alg == "nlms":
            H, errSq[:, ind] = nlmsUp(x[indIn, :], dx[ind, :], outEq, mu, H, nModes)
        elif alg == "cma":
            H, errSq[:, ind] = cmaUp(x[indIn, :], Rcma, outEq, mu, H, nModes)
        elif alg == "dd-lms":
            H, errSq[:, ind] = ddlmsUp(x[indIn, :], constSymb, outEq, mu, H, nModes, px)
        elif alg == "rde":
            H, errSq[:, ind] = rdeUp(x[indIn, :], Rrde, outEq, mu, H, nModes)
        elif alg == "da-rde":
            H, errSq[:, ind] = dardeUp(x[indIn, :], dx[ind, :], outEq, mu, H, nModes)
        elif alg == "rls":
            H, Sd, errSq[:, ind] = rlsUp(
                x[indIn, :], dx[ind, :], outEq, lambdaRLS, H, Sd, nModes
            )
        elif alg == "dd-rls":
            H, Sd, errSq[:, ind] = ddrlsUp(
                x[indIn, :], constSymb, outEq, lambdaRLS, H, Sd, nModes
            )
        elif alg == "static":
            errSq[:, ind] = errSq[:, ind - 1]
        else:
            raise ValueError(
                "Equalization algorithm not specified (or incorrectly specified)."
            )
        if storeCoeff:
            Hiter[:, :, ind] = H
        else:
            Hiter[:, :, 0] = H
    return yEq, H, errSq, Hiter


# def nlmsUp(x, dx, outEq, mu, H, nModes):
#     """
#     Coefficient update with the NLMS algorithm.

#     Parameters
#     ----------
#     x : torch.Tensor
#         Input tensor.
#     dx : torch.Tensor
#         Desired output tensor.
#     outEq : torch.Tensor
#         Equalized output tensor.
#     mu : float
#         Step size for the update.
#     H : torch.Tensor
#         Coefficient matrix.
#     nModes : int
#         Number of modes.

#     Returns
#     -------
#     torch.Tensor
#         Updated coefficient matrix.
#     torch.Tensor
#         Squared absolute error.

#     """
#     indMode = th.arange(0, nModes)
#     err = dx - outEq.mT  # calculate output error for the NLMS algorithm

#     errDiag = th.diag(err[0])  # define diagonal matrix from error tensor

#     # update equalizer taps
#     for N in range(nModes):
#         indUpdTaps = indMode + N * nModes  # simplify indexing and improve speed
#         inAdapt = x[:, N].T / th.norm(x[:, N]) ** 2  # NLMS normalization
#         inAdaptPar = inAdapt.repeat(nModes).reshape(len(x), -1).T  # expand input to parallelize tap adaptation
#         H[indUpdTaps, :] += mu * errDiag @ th.conj(inAdaptPar)  # gradient descent update
#     return H, th.abs(err) ** 2


def nlmsUp(x, dx, outEq, mu, H, nModes):
    """
    Coefficient update with the NLMS algorithm.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    dx : torch.Tensor
        Desired output tensor.
    outEq : torch.Tensor
        Equalized output tensor.
    mu : float
        Step size for the update.
    H : torch.Tensor
        Coefficient matrix.
    nModes : int
        Number of modes.

    Returns
    -------
    torch.Tensor
        Updated coefficient matrix.
    torch.Tensor
        Squared absolute error.

    """
    err = dx - outEq.T  # Calculate output error for the NLMS algorithm
    errDiag = th.diag(err[0])  # Define diagonal matrix from error tensor

    # NLMS normalization
    x_norm = th.norm(x, dim=0) ** 2
    inAdapt = x / x_norm
    inAdaptPar = inAdapt.repeat(
        nModes, 1
    ).T  # Expand input to parallelize tap adaptation

    # Update equalizer taps using vectorized operations
    indMode = th.arange(0, nModes)
    indUpdTaps = indMode + indMode * nModes
    # H[indUpdTaps, :] += mu * errDiag @ th.conj(inAdaptPar)  # Gradient descent update
    H[indUpdTaps, :] += mu * th.matmul(
        errDiag, th.conj(inAdaptPar)
    )  # Gradient descent update

    return H, th.abs(err) ** 2


def rlsUp(x, dx, outEq, lambdaRLS, H, Sd, nModes):
    """
    Coefficient update with the RLS algorithm.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    dx : torch.Tensor
        Desired output tensor.
    outEq : torch.Tensor
        Equalized output tensor.
    lambdaRLS : float
        RLS forgetting factor.
    H : torch.Tensor
        Coefficient matrix.
    Sd : torch.Tensor
        Inverse correlation matrix.
    nModes : int
        Number of modes.

    Returns
    -------
    torch.Tensor
        Updated coefficient matrix.
    torch.Tensor
        Updated inverse correlation matrix.
    torch.Tensor
        Squared absolute error.

    """
    nTaps = H.shape[1]
    indMode = th.arange(0, nModes)
    indTaps = th.arange(0, nTaps)

    err = dx - outEq.mT  # calculate output error for the RLS algorithm

    errDiag = th.diag(err[0])  # define diagonal matrix from error tensor

    # update equalizer taps
    for N in range(nModes):
        indUpdModes = indMode + N * nModes
        indUpdTaps = indTaps + N * nTaps

        Sd_ = Sd[indUpdTaps, :]

        inAdapt = th.conj(x[:, N]).reshape(-1, 1)  # input samples
        inAdaptPar = (
            (inAdapt.mT).repeat(nModes).reshape(len(x), -1).mT
        )  # expand input to parallelize tap adaptation

        Sd_ = (1 / lambdaRLS) * (
            Sd_
            - (Sd_ @ (inAdapt @ (th.conj(inAdapt).mT)) @ Sd_)
            / (lambdaRLS + (th.conj(inAdapt).mT) @ Sd_ @ inAdapt)
        )

        H[indUpdModes, :] += errDiag @ (Sd_ @ inAdaptPar.mT).mT

        Sd[indUpdTaps, :] = Sd_
    return H, Sd, th.abs(err) ** 2


def ddlmsUp(x, constSymb, outEq, mu, H, nModes, px, σ2=0.1):
    """
    Coefficient update with the DD-LMS algorithm.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    constSymb : torch.Tensor
        Tensor of constellation symbols.
    outEq : torch.Tensor
        Equalized output tensor.
    mu : float
        Step size for the update.
    H : torch.Tensor
        Coefficient matrix.
    nModes : int
        Number of modes.

    Returns
    -------
    torch.Tensor
        Updated coefficient matrix.
    torch.Tensor
        Squared absolute error.

    """
    indMode = th.arange(0, nModes)
    outEq = outEq.mT
    decided = th.zeros(outEq.shape, dtype=th.complex64, device=x.device)

    for k in range(nModes):
        indSymb = th.argmax(-th.abs(outEq[0, k] - constSymb) ** 2 / σ2 + th.log(px))
        decided[0, k] = constSymb[indSymb]
    err = decided - outEq  # calculate output error for the DD-LMS algorithm

    errDiag = th.diag(err[0])  # define diagonal matrix from error tensor

    # update equalizer taps
    for N in range(nModes):
        indUpdTaps = indMode + N * nModes  # simplify indexing
        inAdapt = x[:, N].T
        inAdaptPar = (
            inAdapt.repeat(nModes).reshape(len(x), -1).T
        )  # expand input to parallelize tap adaptation
        H[indUpdTaps, :] += (
            mu * errDiag @ th.conj(inAdaptPar)
        )  # gradient descent update
    return H, th.abs(err) ** 2


def ddrlsUp(x, constSymb, outEq, lambdaRLS, H, Sd, nModes):
    """
    Coefficient update with the DD-RLS algorithm.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    constSymb : torch.Tensor
        Tensor of constellation symbols.
    outEq : torch.Tensor
        Equalized output tensor.
    lambdaRLS : float
        RLS forgetting factor.
    H : torch.Tensor
        Coefficient matrix.
    Sd : torch.Tensor
        Inverse correlation matrix.
    nModes : int
        Number of modes.

    Returns
    -------
    torch.Tensor
        Updated coefficient matrix.
    torch.Tensor
        Updated inverse correlation matrix.
    torch.Tensor
        Squared absolute error.

    """
    nTaps = H.shape[1]
    indMode = th.arange(0, nModes)
    indTaps = th.arange(0, nTaps)

    outEq = outEq.mT
    decided = th.zeros(outEq.shape, dtype=th.complex64, device=x.device)

    for k in range(nModes):
        indSymb = th.argmin(th.abs(outEq[0, k] - constSymb))
        decided[0, k] = constSymb[indSymb]
    err = decided - outEq  # calculate output error for the DD-RLS algorithm

    errDiag = th.diag(err[0])  # define diagonal matrix from error tensor

    # update equalizer taps
    for N in range(nModes):
        indUpdModes = indMode + N * nModes
        indUpdTaps = indTaps + N * nTaps

        Sd_ = Sd[indUpdTaps, :]

        inAdapt = th.conj(x[:, N]).reshape(-1, 1)  # input samples
        inAdaptPar = (
            (inAdapt.mT).repeat(nModes).reshape(len(x), -1).mT
        )  # expand input to parallelize tap adaptation

        Sd_ = (1 / lambdaRLS) * (
            Sd_
            - (Sd_ @ (inAdapt @ (th.conj(inAdapt).mT)) @ Sd_)
            / (lambdaRLS + (th.conj(inAdapt).mT) @ Sd_ @ inAdapt)
        )

        H[indUpdModes, :] += errDiag @ (Sd_ @ inAdaptPar.mT).mT

        Sd[indUpdTaps, :] = Sd_
    return H, Sd, th.abs(err) ** 2


def cmaUp(x, R, outEq, mu, H, nModes):
    """
    Coefficient update with the CMA algorithm.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    R : torch.Tensor
        Correlation tensor.
    outEq : torch.Tensor
        Equalized output tensor.
    mu : float
        Step size parameter.
    H : torch.Tensor
        Coefficient matrix.
    nModes : int
        Number of modes.

    Returns
    -------
    torch.Tensor
        Updated coefficient matrix.
    torch.Tensor
        Squared absolute error.

    """
    indMode = th.arange(0, nModes)
    outEq = outEq.mT
    err = R - th.abs(outEq) ** 2  # calculate output error for the CMA algorithm

    prodErrOut = th.diag(err[0]) @ th.diag(outEq[0])  # define diagonal matrix

    # update equalizer taps
    for N in range(nModes):
        indUpdTaps = indMode + N * nModes  # simplify indexing
        inAdapt = x[:, N].mT
        inAdaptPar = (
            inAdapt.repeat(nModes).reshape(len(x), -1).mT
        )  # expand input to parallelize tap adaptation
        H[indUpdTaps, :] += (
            mu * prodErrOut @ th.conj(inAdaptPar)
        )  # gradient descent update
    return H, th.abs(err) ** 2


def rdeUp(x, R, outEq, mu, H, nModes):
    """
    Coefficient update with the RDE algorithm.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    R : torch.Tensor
        Constellation radius tensor.
    outEq : torch.Tensor
        Equalized output tensor.
    mu : float
        Step size parameter.
    H : torch.Tensor
        Coefficient matrix.
    nModes : int
        Number of modes.

    Returns
    -------
    torch.Tensor
        Updated coefficient matrix.
    torch.Tensor
        Squared absolute error.

    """
    indMode = th.arange(0, nModes)
    outEq = outEq.mT
    decidedR = th.zeros(outEq.shape, dtype=th.complex64, device=x.device)

    for k in range(nModes):
        indR = th.argmin(th.abs(R - th.abs(outEq[0, k])))
        decidedR[0, k] = R[indR]
    err = (
        decidedR**2 - th.abs(outEq) ** 2
    )  # calculate output error for the RDE algorithm

    prodErrOut = th.diag(err[0]) @ th.diag(outEq[0])  # define diagonal matrix

    # update equalizer taps
    for N in range(nModes):
        indUpdTaps = indMode + N * nModes  # simplify indexing
        inAdapt = x[:, N].T
        inAdaptPar = (
            inAdapt.repeat(nModes).reshape(len(x), -1).T
        )  # expand input to parallelize tap adaptation
        H[indUpdTaps, :] += (
            mu * prodErrOut @ th.conj(inAdaptPar)
        )  # gradient descent update
    return H, th.abs(err) ** 2


def dardeUp(x, dx, outEq, mu, H, nModes):
    """
    Coefficient update with the data-aided RDE algorithm.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    dx : torch.Tensor
        Exact constellation radius tensor.
    outEq : torch.Tensor
        Equalized output tensor.
    mu : float
        Step size parameter.
    H : torch.Tensor
        Coefficient matrix.
    nModes : int
        Number of modes.

    Returns
    -------
    torch.Tensor
        Updated coefficient matrix.
    torch.Tensor
        Squared absolute error.

    """
    indMode = th.arange(0, nModes)
    outEq = outEq.mT
    decidedR = th.zeros(outEq.shape, dtype=th.complex64, device=x.device)

    for k in range(nModes):
        decidedR[0, k] = th.abs(dx[k])
    err = (
        decidedR**2 - th.abs(outEq) ** 2
    )  # calculate output error for the RDE algorithm

    prodErrOut = th.diag(err[0]) @ th.diag(outEq[0])  # define diagonal matrix

    # update equalizer taps
    for N in range(nModes):
        indUpdTaps = indMode + N * nModes  # simplify indexing
        inAdapt = x[:, N].mT
        inAdaptPar = (
            inAdapt.repeat(nModes).reshape(len(x), -1).mT
        )  # expand input to parallelize tap adaptation
        H[indUpdTaps, :] += (
            mu * prodErrOut @ th.conj(inAdaptPar)
        )  # gradient descent update
    return H, th.abs(err) ** 2


def dbp(Ei, Fs, Ltotal, Lspan, hz=0.5, alpha=0.2, gamma=1.3, D=16, Fc=193.1e12):
    """
    Digital backpropagation (symmetric, single-pol.)

    :param Ei: input signal
    :param Ltotal: total fiber length [km]
    :param Lspan: span length [km]
    :param hz: step-size for the split-step Fourier method [km][default: 0.5 km]
    :param alpha: fiber attenuation parameter [dB/km][default: 0.2 dB/km]
    :param D: chromatic dispersion parameter [ps/nm/km][default: 16 ps/nm/km]
    :param gamma: fiber nonlinear parameter [1/W/km][default: 1.3 1/W/km]
    :param Fc: carrier frequency [Hz][default: 193.1e12 Hz]
    :param Fs: sampling frequency [Hz]

    :return Ech: backpropagated signal
    """
    c_kms = c / 1e3
    λ = c_kms / Fc
    α = -alpha / (10 * np.log10(np.exp(1)))
    β2 = (D * λ**2) / (2 * np.pi * c_kms)
    γ = -gamma

    Nfft = len(Ei)

    ω = 2 * np.pi * Fs * fft.fftfreq(Nfft)

    Nspans = int(np.floor(Ltotal / Lspan))
    Nsteps = int(np.floor(Lspan / hz))

    Ech = Ei.reshape(
        len(Ei),
    )
    Ech = fft.fft(Ech)  # single-polarization field

    linOperator = np.exp(-(α / 2) * (hz / 2) + 1j * (β2 / 2) * (ω**2) * (hz / 2))

    for _ in tqdm(range(Nspans)):
        Ech = Ech * np.exp((α / 2) * Nsteps * hz)

        for _ in range(Nsteps):
            # First linear step (frequency domain)
            Ech = Ech * linOperator

            # Nonlinear step (time domain)
            Ech = fft.ifft(Ech)
            Ech = Ech * np.exp(1j * γ * (Ech * np.conj(Ech)) * hz)

            # Second linear step (frequency domain)
            Ech = fft.fft(Ech)
            Ech = Ech * linOperator
    Ech = fft.ifft(Ech)

    return Ech.reshape(
        len(Ech),
    )
