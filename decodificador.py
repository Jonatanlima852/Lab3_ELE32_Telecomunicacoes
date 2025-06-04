import numpy as np
import matplotlib.pyplot as plt
from bpsk_llr import add_awgn_noise, calculate_llr
from gerar_grafo_ldpc import converter_H_para_AB, criar_matriz_verificacao_ldpc, encontrar_N_proximo, exportar_grafo_csv

def calcular_mensagem_v_node(mensagens_entrada, llr_canal):
    """
    Calcula a mensagem de saída de um v-node para um c-node
    
    Args:
        mensagens_entrada: Lista de mensagens LLR de outros c-nodes conectados
        llr_canal: LLR do canal para este v-node
        
    Returns:
        Mensagem LLR de saída
    """
    # A mensagem de saída é a soma das mensagens de entrada dos outros ramos + LLR do canal
    mensagem_saida = llr_canal + np.sum(mensagens_entrada)
    return mensagem_saida

def calcular_mensagem_c_node(mensagens_entrada):
    """
    Calcula a mensagem de saída de um c-node para um v-node
    
    Args:
        mensagens_entrada: Lista de mensagens LLR de outros v-nodes conectados
        
    Returns:
        Mensagem LLR de saída
    """
    # Converte LLRs para sinal e magnitude
    sinais = np.sign(mensagens_entrada)
    magnitudes = np.abs(mensagens_entrada)
    
    # Sinal = produto dos sinais das mensagens de entrada
    sinal_saida = np.prod(sinais)
    
    # Magnitude = menor das magnitudes das mensagens de entrada
    magnitude_saida = np.min(magnitudes)
    
    return sinal_saida * magnitude_saida

def verificar_palavra_codigo(v_to_c, B, A, llrs_canal):
    """
    Verifica se as restrições de paridade estão satisfeitas usando
    a decisão hard baseada no LLR total
    """
    M, dc = B.shape # M = número de c-nodes, dc = grau dos c-nodes
    N = A.shape[0]
    
    # Calcula o LLR total para cada bit
    llr_total = np.zeros(N)
    for n in range(N):
        llr_total[n] = llrs_canal[n]
        for j in range(A.shape[1]):
            c_node = A[n, j]
            i = np.where(B[c_node, :] == n)[0][0]
            llr_total[n] += v_to_c[n, j]
    
    # Toma a decisão hard
    bits = np.zeros(N, dtype=int)
    bits[llr_total < 0] = 1
    
    # Verifica se H·bits^T = 0
    for m in range(M):
        soma = 0
        for i in range(dc):
            v_node = B[m, i]
            soma = (soma + bits[v_node]) % 2
        if soma != 0:
            return False
    
    return True

def decodificar_ldpc(llrs_canal, A, B, max_iteracoes=50):
    """
    Decodifica LDPC usando algoritmo de propagação de crenças (belief propagation)
    
    Args:
        llrs_canal: LLRs iniciais do canal
        H: Matriz de verificação de paridade
        max_iteracoes: Número máximo de iterações
        
    Returns:
        bits_decodificados: Bits decodificados
        iteracoes: Número de iterações executadas
    """
    # Descontinuado: Passando A e B invés de H
    # M, N = H.shape  # M = número de c-nodes, N = número de v-nodes

    # Obtendo dimensões da da forma correta
    N = A.shape[0]
    M = B.shape[0]
    
    # Converte H para matrizes A e B
    dv = A.shape[1]  # grau dos v-nodes
    dc = B.shape[1]  # grau dos c-nodes
    
    # Inicializa mensagens v-node para c-node (N x dv)
    v_to_c = np.zeros((N, dv))
    
    # Inicializa mensagens c-node para v-node (M x dc)
    c_to_v = np.zeros((M, dc))
    
    # Inicializa as mensagens v-to-c com os LLRs do canal
    for n in range(N):
        v_to_c[n, :] = llrs_canal[n]
    
    # Loop de iterações
    for iteracao in range(max_iteracoes):
        # print("------------- printando v_to_c -------------")
        # print(v_to_c, c_to_v)

        # Passo 4: Atualiza as mensagens dos v-nodes para c-nodes
        for n in range(N):  # Para cada v-node
            for j in range(dv):  # Para cada c-node conectado
                c_node = A[n, j]  # Índice do c-node
                
                # Encontra o índice do v-node n na lista de conexões do c-node
                i = np.where(B[c_node, :] == n)[0][0]
                
                # Obtém as mensagens de entrada para este v-node (excluindo a mensagem do c-node atual)
                mensagens_entrada = []
                for k in range(dv):
                    if k != j:
                        c_conectado = A[n, k]
                        idx = np.where(B[c_conectado, :] == n)[0][0]
                        mensagens_entrada.append(c_to_v[c_conectado, idx])
                
                # Calcula a mensagem de saída usando a regra do v-node
                v_to_c[n, j] = calcular_mensagem_v_node(mensagens_entrada, llrs_canal[n])
        
        # print("------------- printando v_to_c -------------")
        # print(v_to_c, c_to_v)
        
        # Passo 6: Atualiza as mensagens dos c-nodes para v-nodes
        for m in range(M):  # Para cada c-node
            for i in range(dc):  # Para cada v-node conectado
                v_node = B[m, i]  # Índice do v-node
                
                # Encontra o índice do c-node m na lista de conexões do v-node
                j = np.where(A[v_node, :] == m)[0][0]
                
                # Obtém as mensagens de entrada para este c-node (excluindo a mensagem do v-node atual)
                mensagens_entrada = []
                for k in range(dc):
                    if k != i:
                        v_conectado = B[m, k]
                        idx = np.where(A[v_conectado, :] == m)[0][0]
                        mensagens_entrada.append(v_to_c[v_conectado, idx])
                
                # Calcula a mensagem de saída usando a regra do c-node
                c_to_v[m, i] = calcular_mensagem_c_node(mensagens_entrada)

        # Passo 5: Verifica se encontrou palavra-código válida
        if verificar_palavra_codigo(v_to_c, B, A, llrs_canal):
            break
    
    # Passo 8: Toma a decisão final para cada bit
    bits_decodificados = np.zeros(N, dtype=int)
    for n in range(N):
        # Soma todas as mensagens LLR dos c-nodes conectados + LLR do canal
        soma_llr = llrs_canal[n]
        for j in range(dv):
            c_node = A[n, j]
            i = np.where(B[c_node, :] == n)[0][0]
            soma_llr += c_to_v[c_node, i]
        
        # Decide o bit: LLR > 0 => bit = 0, LLR < 0 => bit = 1
        bits_decodificados[n] = 0 if soma_llr > 0 else 1
    
    return bits_decodificados, iteracao + 1

def simular_sistema_ldpc(bits_info, A, B, eb_n0_db, max_iteracoes=50):
    """
    Simula o sistema completo de transmissão e decodificação LDPC
    
    Args:
        bits_info: Bits de informação (K bits) -> Aqui usaremos vetor todo de zeros e de tamanho K
        A, B: Matrizes de conexão do grafo Tanner
        eb_n0_db: Eb/N0 em dB
        max_iteracoes: Número máximo de iterações do decodificador
        
    Returns:
        bits_decodificados: Bits decodificados
        ber: Taxa de erro de bit
        iteracoes: Número de iterações executadas
    """
    # Como estamos usando vetor nulo, não precisamos codificar
    N = A.shape[0]  # Número total de bits
    M = B.shape[0]       # número de restrições (nós c)
    K = bits_info.shape[0]  # número de bits de informação

    # 1) Gera bits de informação (vetor nulo)
    bits = np.zeros(N, dtype=int)  
    
    # Converte bits para símbolos BPSK
    simbolos = - 2 * bits + 1
    
    # 2) Calcula a taxa R = K/N para passar ao canal
    R = K / N
    
    # Simula o canal AWGN
    simbolos_recebidos = add_awgn_noise(simbolos, eb_n0_db, R)
    
    # Calcula os LLRs do canal
    llrs_canal = calculate_llr(simbolos_recebidos, eb_n0_db, R)
    
    # Decodifica usando o algoritmo LDPC
    bits_decodificados, iteracoes = decodificar_ldpc(llrs_canal, A, B, max_iteracoes)
    
    # Calcula a taxa de erro de bit
    ber = np.mean(bits != bits_decodificados)
    
    return bits_decodificados, ber, iteracoes

if __name__ == "__main__":
    # Teste do decodificador
    
    # Parâmetros
    valor_alvo = 1000
    dv = 3
    dc = 7  # Reduzido para melhor convergência
    max_iteracoes = 50
    num_trials = 1000  # ex.: 100 rodadas Monte Carlo por Eb/N0

    
    # Gera a matriz H
    print("Gerando matriz LDPC...")
    N = encontrar_N_proximo(valor_alvo, dv, dc)
    H = criar_matriz_verificacao_ldpc(N, dv, dc)
    M = H.shape[0]
    A, B = converter_H_para_AB(H)
    
    # Exporta o grafo em formato CSV
    exportar_grafo_csv(A, N, f"grafo_ldpc_N{N}_dv{dv}_dc{dc}.csv")
    print(f"Grafo exportado em: grafo_ldpc_N{N}_dv{dv}_dc{dc}.csv")
    
    # Gera bits de informação (vetor nulo)
    K = N - M
    bits_info = np.zeros(K, dtype=int)
    
    # Imprime informações sobre o código
    print(f"\nParâmetros do código:")
    print(f"N = {N} (tamanho do código)")
    print(f"M = {M} (número de restrições)")
    print(f"K = {K} (bits de informação)")
    print(f"Code Rate (R) = {K/N:.3f}")
    
    # Testa para diferentes valores de Eb/N0
    eb_n0_db_range = np.array([-1, -0.5,0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5])
    ber_values = []
    iter_values = []
    
    # Define número de trials para cada Eb/N0
    # Mais loops para Eb/N0 altos, pois assim a simulação é mais precisa já que o ruído será mínimo.

    # num_trials = np.array([25, 50, 50, 100, 100, 200, 200, 500, 500, 1000, 1000, 2000, 2000, 5000])
    num_trials = np.array([10, 10, 10, 10, 10, 10, 10, 20, 20, 50, 50, 100, 100, 200, 200, 500])
    
    print("\n=== Iniciando simulações com número variável de réplicas por Eb/N0 ===")
    for idx, eb_n0_db in enumerate(eb_n0_db_range):
        soma_ber = 0.0
        soma_iter = 0
        trials_atual = num_trials[idx]
        print(f"\nEb/N0 = {eb_n0_db:.1f} dB - Executando {trials_atual} simulações...")
        
        for trial in range(trials_atual):
            _, ber, iters = simular_sistema_ldpc(bits_info, A, B, eb_n0_db, max_iteracoes)
            soma_ber += ber
            soma_iter += iters
            
        # Cálculo das médias
        media_ber = soma_ber / trials_atual
        media_iter = soma_iter / trials_atual

        ber_values.append(media_ber)
        iter_values.append(media_iter)
        print(f"Eb/N0 = {eb_n0_db:.1f} dB → BER médio = {media_ber:.6f}, Iterações médias = {media_iter:.1f}")

    
    # Plota os resultados
    print("\n=== Gerando gráficos ===")
    plt.figure(figsize=(12, 6))  # Ajustei o tamanho para um único gráfico
    
    # Gráfico da BER
    plt.semilogy(eb_n0_db_range, ber_values, 'bo-', linewidth=2)
    plt.grid(True, which="both", ls="-", alpha=0.2)  # Adiciona grid para ambas as escalas
    plt.minorticks_on()  # Ativa ticks menores
    plt.xlabel('Eb/N0 (dB)', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.title(f'Desempenho do Código LDPC\nN={N}, dv={dv}, dc={dc}, R={K/N:.3f}', fontsize=14)
    plt.ylim(1e-6, 1)
    
    # Adiciona linhas horizontais de referência
    for ber in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        plt.axhline(y=ber, color='gray', linestyle='--', alpha=0.3)
    
    # Adiciona informações sobre o número de trials
    trials_info = f"Trials: {num_trials[0]}-{num_trials[-1]}"
    plt.figtext(0.02, 0.02, trials_info, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('ldpc_performance_N1000_dv3_dc7.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Simulação concluída! Resultados salvos em 'ldpc_performance.png'")
