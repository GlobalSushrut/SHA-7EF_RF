"""
Future-Proof Cryptographic System (FPCS)
A next-generation post-quantum, AI-resistant encryption model.
"""

from .core import FRH, ARH, CSB, QuantumInspiredKey
from .utils import generate_dynamic_key, measure_entropy

__version__ = "0.1.0"
__all__ = ['FRH', 'ARH', 'CSB', 'QuantumInspiredKey', 'generate_dynamic_key', 'measure_entropy']
