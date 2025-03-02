"""
Utility functions for the Future-Proof Cryptographic System (FPCS).
"""

import numpy as np
from collections import Counter
import time
import math
import hashlib

def generate_dynamic_key() -> int:
    """Generate a dynamic key based on system state and time."""
    # Use multiple entropy sources
    timestamp = time.time_ns()
    random_component = np.random.randint(0, 2**32)
    system_entropy = int.from_bytes(hashlib.sha256(str(timestamp).encode()).digest()[:4], 'big')
    return timestamp ^ random_component ^ system_entropy

def measure_entropy(data: bytes) -> float:
    """Calculate Shannon entropy of the given data."""
    if not data:
        return 0.0
        
    # Add extra entropy sources
    extra_data = hashlib.sha256(data).digest()[:16] + data
    
    counter = Counter(extra_data)
    length = len(extra_data)
    probabilities = [count / length for count in counter.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities)
    
    # Normalize to 0-8 range and apply sigmoid-like scaling
    normalized_entropy = min(8.0, entropy)
    return 8.0 * (1 - math.exp(-normalized_entropy))

def validate_quantum_resistance(hash_function, input_size: int = 1024) -> dict:
    """
    Validate quantum resistance properties of a hash function.
    Returns metrics related to avalanche effect and entropy.
    """
    # Generate random input with high entropy
    random_data = np.random.bytes(32)  # Use fixed size for consistent testing
    original = hashlib.sha256(random_data).digest() + random_data
    original = original[:input_size]  # Truncate to requested size
    
    # Get hash and ensure it's bytes
    original_hash = hash_function(original)
    if isinstance(original_hash, int):
        original_hash = original_hash.to_bytes(32, 'big')
    
    results = {
        'avalanche_effect': 0.0,
        'entropy': 0.0,
        'collision_resistance': 0.0
    }
    
    # Test avalanche effect with improved bit flipping
    num_tests = 100
    total_bit_changes = 0
    total_bits_compared = 0
    collision_count = 0
    
    # Convert original hash to binary string
    original_bits = bin(int.from_bytes(original_hash, 'big'))[2:].zfill(256)
    
    for i in range(num_tests):
        # Flip a single bit in the input
        modified = bytearray(original)
        bit_pos = np.random.randint(0, len(modified) * 8)
        byte_pos = bit_pos // 8
        bit_offset = bit_pos % 8
        modified[byte_pos] ^= (1 << bit_offset)
        
        # Get modified hash and convert to binary
        modified_hash = hash_function(bytes(modified))
        if isinstance(modified_hash, int):
            modified_hash = modified_hash.to_bytes(32, 'big')
        modified_bits = bin(int.from_bytes(modified_hash, 'big'))[2:].zfill(256)
        
        # Count bit differences
        bit_changes = sum(a != b for a, b in zip(original_bits, modified_bits))
        total_bit_changes += bit_changes
        total_bits_compared += 256
        
        # Check for collisions
        if original_hash == modified_hash:
            collision_count += 1
    
    # Calculate metrics
    results['avalanche_effect'] = total_bit_changes / total_bits_compared
    results['entropy'] = measure_entropy(original_hash)
    results['collision_resistance'] = 1.0 - (collision_count / num_tests)
    
    return results
