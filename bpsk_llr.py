import numpy as np

def generate_bpsk_symbols(num_symbols):
    """
    Gera símbolos BPSK (1 -> +1, 0 -> -1)
    """
    # Gera bits aleatórios
    # bits = np.random.randint(0, 2, num_symbols)
    bits = np.zeros(num_symbols)
    # Converte para símbolos BPSK
    symbols = - 2 * bits + 1
    return bits, symbols

def add_awgn_noise(symbols, eb_n0_db):
    """
    Adiciona ruído Gaussiano branco aos símbolos
    eb_n0_db: Eb/N0 em dB
    """
    # Converte Eb/N0 de dB para linear
    eb_n0_linear = 10 ** (eb_n0_db / 10)
    
    # Calcula a variância do ruído
    noise_variance = 1 / (2 * eb_n0_linear)
    
    # Gera ruído Gaussiano
    noise = np.random.normal(0, np.sqrt(noise_variance), len(symbols))
    
    # Adiciona ruído aos símbolos
    received_symbols = symbols + noise
    
    return received_symbols

def calculate_llr(received_symbols, eb_n0_db):
    """
    Calcula os LLRs para os símbolos recebidos
    """
    # Converte Eb/N0 de dB para linear
    eb_n0_linear = 10 ** (eb_n0_db / 10)
    
    # Calcula os LLRs
    llrs = 4 * eb_n0_linear * received_symbols
    
    return llrs

def simulate_bpsk_awgn(num_symbols, eb_n0_db):
    """
    Simula a transmissão BPSK através de um canal AWGN e retorna os LLRs
    """
    # Gera símbolos BPSK
    bits, symbols = generate_bpsk_symbols(num_symbols)
    
    # Adiciona ruído
    received_symbols = add_awgn_noise(symbols, eb_n0_db)
    
    # Calcula LLRs
    llrs = calculate_llr(received_symbols, eb_n0_db)
    
    return bits, symbols, received_symbols, llrs

# Exemplo de uso
if __name__ == "__main__":
    # Parâmetros
    num_symbols = 10
    eb_n0_db = 1  # dB
    
    # Simula o sistema
    bits, symbols, received_symbols, llrs = simulate_bpsk_awgn(num_symbols, eb_n0_db)
    
    # Imprime alguns resultados
    print(f"Primeiros 10 bits originais: {bits[:10]}")
    print(f"Primeiros 10 símbolos BPSK: {symbols[:10]}")
    print(f"Primeiros 10 símbolos recebidos: {received_symbols[:10]}")
    print(f"Primeiros 10 LLRs: {llrs[:10]}")