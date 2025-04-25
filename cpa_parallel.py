import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import multiprocessing as mp
from functools import partial
import time

# AES S-box table for SubBytes operation
SBOX = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
]

def load_cleartext(filename):
    """Load cleartext values from file"""
    with open(filename, 'r') as f:
        content = f.read()
    
    # Parse the file content into integers
    values = [int(val) for val in content.split()]
    
    # Reshape into rows where each row is a block of 16 bytes
    return np.array(values).reshape(-1, 16)

def load_trace_file(filename):
    """Load power trace values from file"""
    with open(filename, 'r') as f:
        trace_data = np.loadtxt(f)
    return trace_data

def calculate_hamming_weight(value):
    """Calculate the Hamming Weight (number of set bits) in a byte"""
    return bin(value).count('1')

def perform_cpa_attack(trace_file_path, cleartext_data, byte_index):
    """
    Perform CPA attack on a specific byte position
    
    Args:
        trace_file_path: Path to the trace file
        cleartext_data: Cleartext input data
        byte_index: Index of the byte to attack (0-15)
        
    Returns:
        Tuple of (byte_index, best_key_byte, max_correlation_value, all_correlations)
    """
    print(f"Starting attack on byte {byte_index}...")
    
    # Load the power traces for the specific byte
    traces = load_trace_file(trace_file_path)

    # Get the plaintext bytes at the target position
    plaintext_bytes = cleartext_data[:, byte_index]
    
    num_traces = len(traces)
    num_samples = len(traces[0])
    
    # Array to store correlation for each key guess
    max_correlations = np.zeros(256)
    
    # For each possible key byte value (0-255)
    for key_guess in range(256):
        # Create hypothetical power consumption model using hamming weight
        hw_values = np.zeros(num_traces)
        
        # For each trace/plaintext
        for i in range(num_traces):
            # Calculate SubBytes output: S-box[plaintext XOR key]
            sbox_output = SBOX[plaintext_bytes[i] ^ key_guess]
            
            # Calculate Hamming Weight of the output
            hw_values[i] = calculate_hamming_weight(sbox_output)
        
        # Calculate correlation between our model and the actual power traces
        # for each sample point in the trace
        correlations = np.zeros(num_samples)
        
        for i in range(num_samples):
            # Extract power values for this sample point across all traces
            power_values = traces[:, i]
            
            # Calculate Pearson correlation
            correlation, _ = pearsonr(hw_values, power_values)
            correlations[i] = abs(correlation)  # Take absolute value
        
        # Store the maximum correlation for this key guess
        max_correlations[key_guess] = np.max(correlations)
        print(f"[debug] Max correlation for guess [{byte_index}][{key_guess}]: {max_correlations[key_guess]}")
    
    # Find the key byte with highest correlation
    best_key_byte = np.argmax(max_correlations)
    max_correlation = np.max(max_correlations)
    
    print(f"Finished byte {byte_index}: Best key 0x{best_key_byte:02x} with correlation {max_correlation:.4f}")
    
    return byte_index, best_key_byte, max_correlation, max_correlations

def attack_single_byte_wrapper(args):
    """Wrapper function for multiprocessing pool"""
    dataset_prefix, cleartext_data, byte_index = args
    trace_file = f"{dataset_prefix}{byte_index}.txt"
    return perform_cpa_attack(trace_file, cleartext_data, byte_index)

def attack_all_bytes_parallel(dataset_num=1, num_processes=None):
    """
    Attack all 16 bytes of the AES key in parallel
    
    Args:
        dataset_num: Dataset number (1 or 2)
        num_processes: Number of processes to use (defaults to CPU count)
        
    Returns:
        The full AES key
    """
    start_time = time.time()
    
    # Dataset paths
    dataset_prefix = f"dataset{dataset_num}/trace"
    cleartext_file = f"dataset{dataset_num}/cleartext.txt"
    
    # Load cleartext data
    cleartext_data = load_cleartext(cleartext_file)
    
    # Prepare arguments for the parallel processing
    args_list = [(dataset_prefix, cleartext_data, byte_index) for byte_index in range(16)]
    
    # Use multiprocessing pool to attack all bytes in parallel
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    print(f"Starting parallel attack using {num_processes} processes...")
    
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(attack_single_byte_wrapper, args_list)
    
    # Sort results by byte index
    results.sort(key=lambda x: x[0])
    
    # Extract key bytes and correlations
    recovered_key = [res[1] for res in results]
    correlations = [res[3] for res in results]
    
    # Convert to numpy array for easy manipulation
    recovered_key = np.array(recovered_key)
    
    # Generate plots for each byte
    for byte_index, corr in enumerate(correlations):
        plt.figure(figsize=(10, 6))
        plt.bar(range(256), corr)
        plt.xlabel('Key Byte Guess')
        plt.ylabel('Maximum Correlation')
        plt.title(f'Correlation for Byte {byte_index}')
        plt.savefig(f'correlation_byte_{byte_index}.png')
        plt.close()
    
    # Verify key by summing all bytes
    key_sum = np.sum(recovered_key)
    expected_sum = 1712 if dataset_num == 1 else 1434
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Print results
    print("\n============ RESULTS ============")
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"Recovered Key: {[hex(b) for b in recovered_key]}")
    print(f"Key Sum: {key_sum} (Expected: {expected_sum})")
    
    if key_sum == expected_sum:
        print("Key sum verification PASSED!")
    else:
        print("Key sum verification FAILED!")
    
    return recovered_key

def main():
    cleartext_file = "dataset1/cleartext.txt"
    
    # For comparison purposes, you can uncomment this to run in single-process mode
    # print("\nPerforming sequential key recovery for dataset 1...")
    # start_time = time.time()
    # # Use the original attack function
    # from original_cpa import attack_all_bytes
    # full_key = attack_all_bytes(dataset_num=1)
    # end_time = time.time()
    # print(f"Sequential execution time: {end_time - start_time:.2f} seconds")
    
    # Run the parallel version
    print("\nPerforming parallel key recovery for dataset 1...")
    full_key = attack_all_bytes_parallel(dataset_num=1)
    
    # You can also test on dataset 2 if available
    # print("\nPerforming parallel key recovery for dataset 2...")
    # full_key2 = attack_all_bytes_parallel(dataset_num=2)

if __name__ == "__main__":
    main()