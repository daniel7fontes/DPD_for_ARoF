import torch as th
import numpy as np


def dec2bitarray(in_number, bit_width):
    """
    Converts a positive integer or an array-like of positive integers to PyTorch tensor of the specified size containing
    bits (0 and 1).

    Parameters
    ----------
    in_number : int or array-like of int
        Positive integer to be converted to a bit array.

    bit_width : int
        Size of the output bit array.

    Returns
    -------
    bitarray : 1D tensor of torch.int
        tensor containing the binary representation of all the input decimal(s).

    """

    if isinstance(in_number, int):
        return decimal2bitarray(in_number, bit_width).clone()
    result = th.zeros(bit_width * len(in_number), dtype=th.int, device=in_number.device)
    for pox, number in enumerate(in_number):
        result[pox * bit_width : (pox + 1) * bit_width] = decimal2bitarray(
            number, bit_width
        ).clone()
    return result


def decimal2bitarray(number, bit_width):
    """
    Converts a positive integer to PyTorch tensor of the specified size containing bits (0 and 1). This version is slightly
    quicker that dec2bitarray but only works for one integer.

    Parameters
    ----------
    in_number : int
        Positive integer to be converted to a bit array.

    bit_width : int
        Size of the output bit array.

    Returns
    -------
    bitarray : 1D tensor of th.int
        tensor containing the binary representation of all the input decimal(s).

    """
    result = th.zeros(bit_width, dtype=th.int, device=number.device)
    i = 1
    pox = 0
    while i <= number:
        if i & number:
            result[bit_width - pox - 1] = 1
        i <<= 1
        pox += 1
    return result


def bitarray2dec(in_bitarray):
    """
    Converts an input PyTorch tensor of bits (0 and 1) to a decimal integer.

    Parameters
    ----------
    in_bitarray : 1D tensor of ints
        Input PyTorch tensor of bits.

    Returns
    -------
    number : int
        Integer representation of input bit array.
    """

    number = th.zeros(1, dtype=th.int, device=in_bitarray.device)

    for i in range(len(in_bitarray)):
        number = number + in_bitarray[i] * pow(2, len(in_bitarray) - 1 - i)

    return number


# def minEuclid(symb, const):
#     """
#     Find minimum Euclidean distance.

#     Find closest constellation symbol w.r.t the Euclidean distance in the
#     complex plane.

#     Parameters
#     ----------
#     symb : torch.tensor
#         Received constellation symbols.
#     const : torch.tensor
#         Reference constellation.

#     Returns
#     -------
#     torch.tensor
#         Indexes of the closest constellation symbols.

#     """
#     ind = th.zeros(symb.shape, dtype=th.int64, device=symb.device)
#     for ii in range(len(symb)):
#         ind[ii] = th.abs(symb[ii] - const).argmin()
#     return ind


def minEuclid(symb, const):
    """
    Vectorized minimum Euclidean distance calculation.

    Find the indexes of the closest constellation symbols w.r.t. the Euclidean distance in the
    complex plane.

    Parameters
    ----------
    symb : torch.Tensor
        Received constellation symbols.
    const : torch.Tensor
        Reference constellation.

    Returns
    -------
    torch.Tensor
        Indexes of the closest constellation symbols.

    """
    # Compute the Euclidean distance between each received symbol and all reference constellation symbols
    dist = th.abs(symb.view(-1, 1) - const.view(1, -1))

    return th.argmin(dist, dim=1)


def detector(r, constSymb, σ2=None, px=None, rule="ML"):
    """
    Perform symbol detection using either the MAP (Maximum A Posteriori) or ML (Maximum Likelihood) rule using PyTorch.

    Parameters
    ----------
    r : th.Tensor
        The received signal.
    σ2 : float
        The noise variance.
    constSymb : th.Tensor
        The constellation symbols.
    px : th.Tensor, optional
        The prior probabilities of each symbol. If None, uniform priors are assumed.
    rule : str, optional
        The detection rule to use. Either 'MAP' (default) or 'ML'.

    Returns
    -------
    th.Tensor
        The detected symbols.
    th.Tensor
        The indices of the detected symbols in the constellation.

    """
    if px is None and rule == "MAP":
        M = len(constSymb)
        # Assume uniform distribution
        px = th.ones(M, device=r.device) / M

    # Find the constellation symbol with the largest P(sm|r) or smallest distance metric
    if rule == "MAP":
        if σ2 is None:
            raise ValueError(
                "Noise variance σ2 needs to be specified if using the 'MAP' rule."
            )

        # Calculate the probability metric
        log_probMetric = -th.abs(r.unsqueeze(1) - constSymb) ** 2 / σ2 + th.log(px)
        indDec = th.argmax(log_probMetric, dim=1)

    elif rule == "ML":
        distMetric = th.abs(r.unsqueeze(1) - constSymb) ** 2
        indDec = th.argmin(distMetric, dim=1)
    else:
        raise ValueError("Detection rule should be either 'MAP' or 'ML'.")

    return indDec


def GrayCode(n):
    """
    Gray code generator.

    Parameters
    ----------
    n : int
        length of the codeword in bits.

    Returns
    -------
    code : list
           list of binary strings of the gray code.

    """
    code = []

    for i in range(1 << n):
        val = i ^ (i >> 1)
        s = bin(val)[2:].zfill(n)
        code.append(s)
    return code


def grayMapping(M, constType, device="cpu"):
    """
    Gray Mapping for digital modulations.

    Parameters
    ----------
    M : int
        modulation order
    constType : 'qam', 'psk', 'pam' or 'ook'.
        type of constellation.

    Returns
    -------
    const : th.Tensor
        tensor with constellation symbols (sorted according to their corresponding
        Gray bit sequence as integer decimal).

    """
    if M != 2 and constType == "ook":
        print("OOK has only 2 symbols, but M != 2. Changing M to 2.")
        M = 2

    bitsSymb = int(np.log2(M))

    code = GrayCode(bitsSymb)
    if constType == "ook":
        const = th.arange(0, 2, device=device)
    elif constType == "pam":
        const = pamConst(M).to(device)
    elif constType == "qam":
        const = qamConst(M).to(device)
    elif constType == "psk":
        const = pskConst(M).to(device)
    elif constType == "apsk":
        const = apskConst(M).to(device)

    const = const.view(M, 1)
    const_ = th.zeros((M, 2), dtype=th.complex64, device=device)

    for ind in range(M):
        const_[ind, 0] = const[ind, 0]  # complex constellation symbol
        const_[ind, 1] = int(code[ind], 2)  # mapped bit sequence (as integer decimal)

    # Sort complex symbols column according to their mapped bit sequence (as integer decimal)
    sorted_indices = th.argsort(const_[:, 1].real)
    const = const_[sorted_indices][:, 0]

    if constType in ["pam", "ook"]:
        const = const.real

    return const


def pamConst(M):
    """
    Generate a Pulse Amplitude Modulation (PAM) constellation.

    Parameters
    ----------
    M : int
        Number of symbols in the constellation. It must be an integer.

    Returns
    -------
    th.Tensor
        1D PAM constellation.
    """
    L = int(M - 1)
    return th.arange(-L, L + 1, 2)


def qamConst(M):
    """
    Generate a Quadrature Amplitude Modulation (QAM) constellation.

    Parameters
    ----------
    M : int
        Number of symbols in the constellation. It must be a perfect square.

    Returns
    -------
    const : th.Tensor
        Complex square M-QAM constellation.
    """
    L = int(np.sqrt(M) - 1)

    # generate 1D PAM constellation
    PAM = np.arange(-L, L + 1, 2)
    PAM = np.array([PAM])

    # generate complex square M-QAM constellation
    const = np.tile(PAM, (L + 1, 1))
    const = const + 1j * np.flipud(const.T)

    for ind in np.arange(1, L + 1, 2):
        const[ind] = np.flip(const[ind], 0)

    return th.tensor(const, dtype=th.complex64)


def pskConst(M):
    """
    Generate a Phase Shift Keying (PSK) constellation.

    Parameters
    ----------
    M : int
        Number of symbols in the constellation. It must be a power of 2 positive integer.

    Returns
    -------
    th.Tensor
        Complex M-PSK constellation.
    """
    pskPhases = th.arange(0, 2 * np.pi, 2 * np.pi / M)
    return th.exp(1j * pskPhases)


def apskConst(M, m1=None, phaseOffset=None):
    """
    Generate an APSK modulated constellation.

    Parameters
    ----------
    M : int
        Constellation order.
    m1 : int
        Number of bits used to index the radii of the constellation.

    Returns
    -------
    th.Tensor
        APSK constellation

    References
    ----------
    Z. Liu, et al "APSK Constellation with Gray Mapping," IEEE Communications Letters, vol. 15, no. 12, pp. 1271-1273, December 2011
    """
    if m1 is None:
        if M == 16:
            m1 = 1
        elif M == 32:
            m1 = 2
        elif M == 64:
            m1 = 2
        elif M == 128:
            m1 = 3
        elif M == 256:
            m1 = 3
        elif M == 512:
            m1 = 4
        elif M == 1024:
            m1 = 4

    nRings = int(2**m1)
    m2 = int(np.log2(M) - m1)
    symbolsPerRing = int(2**m2)

    const = th.zeros((M,), dtype=th.complex64)

    if phaseOffset is None:
        phaseOffset = np.pi / symbolsPerRing

    for idx in range(nRings):
        radius = np.sqrt(-np.log(1 - ((idx + 1) - 0.5) * symbolsPerRing / M))

        if (idx + 1) % 2 == 1:
            const[idx * symbolsPerRing : (idx + 1) * symbolsPerRing] = radius * th.flip(
                pskConst(symbolsPerRing), [0]
            )
        else:
            const[
                idx * symbolsPerRing : (idx + 1) * symbolsPerRing
            ] = radius * pskConst(symbolsPerRing)

    return const * np.exp(1j * phaseOffset)


# def demap(indSymb, bitMap):
#     """
#     Constellation symbol index to bit sequence demapping.

#     Parameters
#     ----------
#     indSymb : torch.tensor of ints
#         Indexes of received symbol sequence.
#     bitMap : torch.tensor (M, log2(M))
#         bit-to-symbol mapping.

#     Returns
#     -------
#     torch.tensor
#         Sequence of demapped bits.

#     """
#     M = bitMap.shape[0]
#     b = int(th.log2(th.tensor([M], device=indSymb.device)))

#     decBits = th.zeros(len(indSymb) * b, dtype=th.int64, device=indSymb.device)

#     for i in range(len(indSymb)):
#         decBits[i * b : i * b + b] = bitMap[indSymb[i], :]
#     return decBits


def demap(indSymb, bitMap):
    """
    Vectorized constellation symbol index to bit sequence demapping.

    Parameters
    ----------
    indSymb : torch.Tensor of ints
        Indexes of received symbol sequence.
    bitMap : torch.Tensor (M, log2(M))
        bit-to-symbol mapping.

    Returns
    -------
    torch.Tensor
        Sequence of demapped bits.

    """
    M = bitMap.shape[0]
    b = int(th.log2(th.tensor([M], device=indSymb.device)))

    # Create an index tensor for bit mapping
    index = indSymb.view(-1, 1).expand(-1, b)

    # Gather bits from the bitMap using the index tensor
    decBits = th.gather(bitMap, 0, index)

    # Reshape decBits to the desired shape
    decBits = decBits.view(-1)

    return decBits


def modulateGray(bits, M, constType):
    """
    Modulate bit sequences to constellation symbol sequences (w/ Gray mapping).

    Parameters
    ----------
    bits : torch.tensor of ints
        Sequence of data bits.
    M : int
        Order of the modulation format.
    constType : string
        'qam', 'psk', 'pam' or 'ook'.

    Returns
    -------
    torch.tensor of complex constellation symbols
        Bits modulated to complex constellation symbols.

    """
    if M != 2 and constType == "ook":
        print("OOK has only 2 symbols, but M != 2. Changing M to 2.")
        M = 2
    bitsSymb = int(np.log2(M))
    const = grayMapping(M, constType, bits.device)

    symb = bits.view(-1, bitsSymb).T
    symbInd = bitarray2dec(symb)

    return const[symbInd]


def demodulateGray(symb, M, constType, bitMap=None):
    """
    Demodulate symbol sequences to bit sequences (w/ Gray mapping).

    Hard demodulation is based on minimum Euclidean distance.

    Parameters
    ----------
    symb : torch.tensor of complex constellation symbols
        Sequence of constellation symbols to be demodulated.
    M : int
        Order of the modulation format.
    constType : string
        'qam', 'psk', 'pam' or 'ook'.

    Returns
    -------
    torch.tensor of ints
        Sequence of demodulated bits.

    """
    if M != 2 and constType == "ook":
        print("OOK has only 2 symbols, but M != 2. Changing M to 2.")
        M = 2
    const = grayMapping(M, constType, symb.device)

    if bitMap is None:
        bitMap = getBitMap(const)

    # Demodulate received symbol sequence
    indrx = detector(symb, const)

    return demap(indrx, bitMap)


def getBitMap(constellation):
    """
    Generate a bit-to-symbol mapping for a given constellation.

    Parameters
    ----------
    constellation : torch.Tensor
        A tensor containing the constellation symbols.

    Returns
    -------
    torch.Tensor
        A tensor representing the bit-to-symbol mapping for the given constellation.
        Each row in the tensor corresponds to a symbol, and each column represents a bit position.
    """

    # Get the device of the input constellation
    dev = constellation.device

    # Determine the modulation order (number of symbols)
    M = len(constellation)

    # Get bit-to-symbol mapping
    indMap = detector(constellation, constellation)
    bitMap = dec2bitarray(indMap, int(th.log2(th.tensor([M], device=dev))))

    # Calculate the number of bits required to represent each symbol
    b = int(th.log2(th.tensor([M], device=dev)))

    # Reshape the bit-to-symbol mapping into a 2D tensor
    bitMap = bitMap.view(-1, b)

    return bitMap


def sample_pmf(px, num_samples=1):
    """
    Sample random integers from 0 to M-1 with a given probability mass function px.

    Parameters
    ----------
    px : th.Tensor
        Probability mass function. Should sum to 1.

    Returns
    -------
    th.Tensor
        Randomly sampled integers.

    """
    px = px / th.sum(px)

    return th.multinomial(px, num_samples, replacement=True)


def maxwell_boltzmann_pmf(λ, const):
    """
    Generate a discrete Maxwell-Boltzmann probability mass function (PMF) using PyTorch.

    Parameters
    ----------
    lmbda : float
        Lambda parameter for the Maxwell-Boltzmann distribution.
    const : th.Tensor
        Tensor containing the discrete values at which to compute the PMF.

    Returns
    -------
    th.Tensor
        Discrete PMF for the given values.

    """
    if λ < 0:
        raise ValueError("Lambda (λ) must be a non-negative float.")

    p = th.exp(-λ * th.abs(const) ** 2)
    p = p / th.sum(p)

    return p
