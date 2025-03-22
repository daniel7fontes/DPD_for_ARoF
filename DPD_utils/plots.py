# -*- coding: utf-8 -*-
"""
======================================================================
Funções para geração de figuras associadas à análise de resultados de DPD
======================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


def plotDPD_const(symbTx, symbRx, symbRx_DLA, symbRx_ILA, show = False, savefig = False, file_path = None):
    
    fig, axs = plt.subplots(1, 2, figsize = (16, 8))
    
    axs[0].plot(symbRx.real, symbRx.imag, "o", color = "r", ms = 2, label = "Sem DPD")
    axs[0].plot(symbRx_DLA.real, symbRx_DLA.imag, "o", color = "b", ms = 2, label = "Com DPD")
    axs[0].plot(symbTx.real, symbTx.imag, "o", color = "k",  ms = 3, label = "SymbTx")
    axs[0].set_title("DPD - DLA", fontsize = 18)
    axs[0].set_ylabel("Quadrature - Q")
    axs[0].set_xlabel("In-Phase - I")
    axs[0].set_xlim(-1.5, 1.5)
    axs[0].set_ylim(-1.5, 1.5)
    axs[0].grid()
    
    axs[1].plot(symbRx.real, symbRx.imag, "o", color = "r", ms = 2, label = "Sem DPD")
    axs[1].plot(symbRx_ILA.real, symbRx_ILA.imag, "o", color = "b", ms = 2, label = "Com DPD")
    axs[1].plot(symbTx.real, symbTx.imag, "o", color = "k",  ms = 3, label = "SymbTx")
    axs[1].set_title("DPD - ILA", fontsize = 18)
    axs[1].set_ylabel("Quadrature - Q")
    axs[1].set_xlabel("In-Phase - I")
    axs[1].set_xlim(-1.5, 1.5)
    axs[1].set_ylim(-1.5, 1.5)
    axs[1].grid()
    
    plt.tight_layout()
    if savefig:
        plt.savefig(file_path)
    
    if not(show):
        plt.close()


def plotDPD_spec(freq, P_sigTx, P_sigRx_PA, P_sigRx_PA_DLA, P_sigRx_PA_ILA, Rs, show = False, savefig = False, file_path = None):    
    fig, axs = plt.subplots(1, 2, figsize = (16, 6))
    
    axs[0].plot(freq/1e9, 10*np.log10(P_sigTx), color = "k", ls = "-", lw = 2, label = "Tx")
    axs[0].plot(freq/1e9, 10*np.log10(P_sigRx_PA), color = "r", ls = "--", lw = 2, label = "Sem DPD")
    axs[0].plot(freq/1e9, 10*np.log10(P_sigRx_PA_DLA), color = "b", ls = "--", lw = 2, label = r"Com DPD-DLA")
    axs[0].set_title("DPD - DLA", fontsize = 18)
    axs[0].set_xlim(-1.5*Rs/1e9, 1.5*Rs/1e9)
    axs[0].set_ylim(-125, -80)
    axs[0].set_xlabel("f [GHz]")
    axs[0].set_ylabel("Power Spectral Density [dB/Hz]")
    axs[0].legend(fontsize = 14, framealpha = 1)
    axs[0].grid()
    
    axs[1].plot(freq/1e9, 10*np.log10(P_sigTx), color = "k", ls = "-", lw = 2, label = "Tx")
    axs[1].plot(freq/1e9, 10*np.log10(P_sigRx_PA), color = "r", ls = "--", lw = 2, label = "Sem DPD")
    axs[1].plot(freq/1e9, 10*np.log10(P_sigRx_PA_ILA), color = "b", ls = "--", lw = 2, label = r"Com DPD-ILA")
    axs[1].set_title("DPD - ILA", fontsize = 18)
    axs[1].set_xlim(-1.5*Rs/1e9, 1.5*Rs/1e9)
    axs[1].set_ylim(-125, -80)
    axs[1].set_xlabel("f [GHz]")
    axs[1].set_ylabel("Power Spectral Density [dB/Hz]")
    axs[1].legend(fontsize = 14, framealpha = 1)
    axs[1].grid()
    
    plt.tight_layout()
    if savefig:
        plt.savefig(file_path)
    
    if not(show):
        plt.close()


def plotDPD_SNR(SNR_per_carrier, SNR_per_carrier_DPD, Ns, DPD_type, show = False, savefig = False, file_path = None):    
    fig, axs = plt.subplots(figsize = (10, 5))
    
    axs.plot(SNR_per_carrier, color = "r", label = "Sem DPD")
    axs.plot(SNR_per_carrier_DPD, color = "b", label = f"Com DPD - {DPD_type}")
    axs.set_xlim(0, Ns)
    axs.set_ylabel("SNR [dB]")
    axs.set_xlabel("Carrier")
    axs.legend(framealpha = 1, fontsize = 14)
    axs.grid()
    plt.tight_layout()
    
    if savefig:
        plt.savefig(file_path)
    
    if not(show):
        plt.close()