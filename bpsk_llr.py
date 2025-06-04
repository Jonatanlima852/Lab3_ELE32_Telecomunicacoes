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

def add_awgn_noise(symbols, eb_n0_db, R):
    """
    Adiciona ruído Gaussiano branco aos símbolos
    eb_n0_db: Eb/N0 em dB
    R: Taxa do código (não usado na variância do ruído)
    """
    # Converte Eb/N0 de dB para linear
    eb_n0_linear = 10 ** (eb_n0_db / 10)
    
    # Calcula a variância do ruído (não depende de R)
    noise_variance = 1 / (2 * R * eb_n0_linear)
    
    # Gera ruído Gaussiano
    noise = np.random.normal(0, np.sqrt(noise_variance), len(symbols))
    
    # Adiciona ruído aos símbolos
    received_symbols = symbols + noise
    
    return received_symbols

def calculate_llr(received_symbols, eb_n0_db, R):
    """
    Calcula os LLRs para os símbolos recebidos
    eb_n0_db: Eb/N0 em dB
    R: Taxa do código (não usado no cálculo do LLR)
    """
    # Converte Eb/N0 de dB para linear
    eb_n0_linear = 10 ** (eb_n0_db / 10)
    
    # Calcula os LLRs (não depende de R)
    llrs = 4 * R * eb_n0_linear * received_symbols
    
    return llrs

def simulate_bpsk_awgn(num_symbols, eb_n0_db, R):
    """
    Simula a transmissão BPSK através de um canal AWGN e retorna os LLRs
    """
    # Gera símbolos BPSK
    bits, symbols = generate_bpsk_symbols(num_symbols)
    
    # Adiciona ruído
    received_symbols = add_awgn_noise(symbols, eb_n0_db, R)
    
    # Calcula LLRs
    llrs = calculate_llr(received_symbols, eb_n0_db, R)
    
    return bits, symbols, received_symbols, llrs

# Exemplo de uso
if __name__ == "__main__":
    # Parâmetros
    num_symbols = 10
    eb_n0_db = 1  # dB
    R = 1  # Assuming R is provided
    
    # Simula o sistema
    bits, symbols, received_symbols, llrs = simulate_bpsk_awgn(num_symbols, eb_n0_db, R)
    
    # Imprime alguns resultados
    print(f"Primeiros 10 bits originais: {bits[:10]}")
    print(f"Primeiros 10 símbolos BPSK: {symbols[:10]}")
    print(f"Primeiros 10 símbolos recebidos: {received_symbols[:10]}")
    print(f"Primeiros 10 LLRs: {llrs[:10]}")