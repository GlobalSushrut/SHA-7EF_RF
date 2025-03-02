"""
Test suite for the Future-Proof Cryptographic System (FPCS).
"""

import pytest
import numpy as np
import hashlib
import random
from fpcs import FRH, ARH, CSB, QuantumInspiredKey
from fpcs.utils import measure_entropy, validate_quantum_resistance

def test_frh_basic_properties():
    """Test basic properties of Factorial-Refractive Hashing."""
    frh = FRH()
    
    # Test deterministic output
    input_data = b"test_data"
    assert frh.hash(input_data) == frh.hash(input_data)
    
    # Test different inputs produce different hashes
    assert frh.hash(b"data1") != frh.hash(b"data2")
    
    # Test entropy of hash output
    hash_value = frh.hash(b"test_entropy")
    entropy = measure_entropy(hash_value.to_bytes(32, 'big'))
    assert entropy > 7.0  # Good entropy should be close to 8 bits per byte

def test_arh_cryptographic_security():
    """Comprehensive cryptographic security test suite."""
    arh = ARH()
    
    def bit_diff(a, b):
        """Calculate bit difference ratio between two byte sequences."""
        return sum(bin(a_i ^ b_i).count('1') for a_i, b_i in zip(a, b)) / (len(a) * 8)
    
    def calculate_entropy(data):
        """Calculate Shannon entropy of byte sequence."""
        counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        probs = counts / len(data)
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    print("\n=== Comprehensive Cryptographic Security Tests ===")
    
    # Test 1: Avalanche Effect (Single-bit Changes)
    print("\n1. Testing Avalanche Effect (Single-bit Changes)...")
    avalanche_scores = []
    for size in [64, 128, 256, 512, 1024]:
        total_avalanche = 0
        num_bits = 20  # Test multiple bit positions
        
        input_data = bytearray(random.getrandbits(8) for _ in range(size))
        original_hash = arh.hash(input_data)
        
        for bit in range(num_bits):
            test_data = bytearray(input_data)
            byte_idx = random.randint(0, len(test_data) - 1)
            bit_idx = random.randint(0, 7)
            test_data[byte_idx] ^= (1 << bit_idx)
            
            modified_hash = arh.hash(test_data)
            avalanche = bit_diff(original_hash, modified_hash)
            total_avalanche += avalanche
            
        avg_avalanche = total_avalanche / num_bits
        avalanche_scores.append(avg_avalanche)
        print(f"Size {size}: Avalanche Effect = {avg_avalanche:.4f}")
    
    # Test 2: Collision Resistance (Birthday Attack Simulation)
    print("\n2. Testing Collision Resistance...")
    num_hashes = 10000
    hash_set = set()
    collisions = 0
    
    for i in range(num_hashes):
        data = str(i).encode() + bytes([random.getrandbits(8) for _ in range(16)])
        hash_val = arh.hash(data)
        if hash_val in hash_set:
            collisions += 1
        hash_set.add(hash_val)
    
    collision_rate = collisions / num_hashes
    print(f"Collision Rate: {collision_rate:.6f}")
    
    # Test 3: Distribution Analysis
    print("\n3. Testing Hash Distribution...")
    hash_bytes = bytearray()
    num_samples = 1000
    
    for i in range(num_samples):
        data = str(i).encode()
        hash_bytes.extend(arh.hash(data))
    
    entropy = calculate_entropy(hash_bytes)
    print(f"Distribution Entropy: {entropy:.4f} bits/byte")
    
    # Test 4: Length Extension Attack Resistance
    print("\n4. Testing Length Extension Attack Resistance...")
    original = b"original message"
    extension = b"extended data"
    
    hash1 = arh.hash(original + extension)
    hash2 = arh.hash(arh.hash(original) + extension)
    
    length_ext_resistance = bit_diff(hash1, hash2)
    print(f"Length Extension Difference: {length_ext_resistance:.4f}")
    
    # Test 5: Preimage Attack Resistance
    print("\n5. Testing Preimage Attack Resistance...")
    target_data = b"target message"
    target_hash = arh.hash(target_data)
    best_match = float('inf')
    
    for i in range(1000):  # Try 1000 random inputs
        test_data = bytes([random.getrandbits(8) for _ in range(len(target_data))])
        test_hash = arh.hash(test_data)
        diff = bit_diff(target_hash, test_hash)
        best_match = min(best_match, diff)
    
    print(f"Best Preimage Match: {best_match:.4f}")
    
    # Test 6: Input Length Variability
    print("\n6. Testing Input Length Variability...")
    prev_hash = None
    length_sensitivity = []
    
    for length in [1, 10, 100, 1000]:
        data = bytes([random.getrandbits(8) for _ in range(length)])
        curr_hash = arh.hash(data)
        
        if prev_hash is not None:
            diff = bit_diff(prev_hash, curr_hash)
            length_sensitivity.append(diff)
            print(f"Length {length}: Hash Difference = {diff:.4f}")
        
        prev_hash = curr_hash
    
    # Assertions for all tests
    print("\n=== Test Results ===")
    
    # 1. Avalanche Effect
    avg_avalanche = sum(avalanche_scores) / len(avalanche_scores)
    print(f"\nAverage Avalanche Effect: {avg_avalanche:.4f}")
    assert 0.45 <= avg_avalanche <= 0.55, "Avalanche effect outside acceptable range"
    
    # 2. Collision Resistance
    assert collision_rate < 0.01, "Too many collisions detected"
    
    # 3. Distribution
    assert entropy > 7.0, "Poor entropy in hash distribution"
    
    # 4. Length Extension
    assert length_ext_resistance > 0.4, "Weak length extension resistance"
    
    # 5. Preimage Resistance
    assert best_match > 0.3, "Weak preimage resistance"
    
    # 6. Length Sensitivity
    avg_length_sensitivity = sum(length_sensitivity) / len(length_sensitivity)
    assert 0.4 <= avg_length_sensitivity <= 0.6, "Poor length sensitivity"

def test_csb_substitution():
    """Test Chaotic S-Box Substitution properties."""
    csb = CSB()
    
    # Test substitution uniqueness
    input_data = b"test_substitution"
    substituted = csb.substitute(input_data)
    assert input_data != substituted
    
    # Test S-box properties
    sbox = csb.generate_sbox()
    assert len(set(sbox)) == 256  # All values should be unique
    assert all(0 <= x < 256 for x in sbox)  # All values should be in valid range

def test_quantum_inspired_key_evolution():
    """Test Quantum Entanglement-Inspired Key Evolution."""
    key_gen = QuantumInspiredKey()
    
    # Generate initial key
    key1, entropy1 = key_gen.generate_key()
    assert len(key1) == 256
    assert entropy1 > 7.0
    
    # Test key evolution
    key2, entropy2 = key_gen.evolve_key(key1)
    assert key1 != key2  # Evolved key should be different
    assert entropy2 > 7.0  # Evolved key should maintain high entropy

def test_system_integration():
    """Test integration of all FPCS components."""
    # Initialize components
    frh = FRH()
    arh = ARH()
    csb = CSB()
    key_gen = QuantumInspiredKey()
    
    # Test data flow through system
    input_data = b"test_integration"
    
    # Generate key
    key, _ = key_gen.generate_key()
    
    # Apply hashing and substitution
    hash1 = frh.hash(input_data)
    hash2 = arh.hash(input_data)
    substituted = csb.substitute(input_data)
    
    # Verify each step produces unique outputs
    assert len(set([hash1, hash2])) == 2
    assert input_data != substituted
    
    # Test entropy at each stage
    stages = [
        hash1.to_bytes(32, 'big'),
        hash2.to_bytes(32, 'big'),
        substituted
    ]
    
    for stage_data in stages:
        entropy = measure_entropy(stage_data)
        assert entropy > 7.0  # Ensure high entropy throughout the system

if __name__ == '__main__':
    pytest.main([__file__])
