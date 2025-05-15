import numpy as np
import matplotlib.pyplot as plt
from bpsk_llr import add_awgn_noise, calculate_llr
from gerar_grafo_ldpc import converter_H_para_AB, criar_matriz_verificacao_ldpc, encontrar_N_proximo

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

def verificar_palavra_codigo(v_to_c, B, A):
    """
    Verifica se as restrições de paridade estão satisfeitas
    
    Args:
        v_to_c: Mensagens dos v-nodes para os c-nodes
        B: Matriz de conexões (c-nodes para v-nodes)
        A: Matriz de conexões (v-nodes para c-nodes)
        
    Returns:
        True se todas as restrições estão satisfeitas, False caso contrário
    """
    M, dc = B.shape  # M = número de c-nodes, dc = grau dos c-nodes
    
    for m in range(M):  # Para cada c-node
        sinais_mensagens = []
        for i in range(dc):  # Para cada v-node conectado
            v_node = B[m, i]  # Índice do v-node
            
            # Encontra o índice do c-node m na lista de conexões do v-node
            j = np.where(A[v_node, :] == m)[0][0]
            
            sinais_mensagens.append(np.sign(v_to_c[v_node, j]))
        
        # Produto dos sinais deve ser positivo (número par de mensagens negativas)
        if np.prod(sinais_mensagens) < 0:
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
        if verificar_palavra_codigo(v_to_c, B, A):
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

def simular_sistema_ldpc(bits, A, B, eb_n0_db, max_iteracoes=50):
    """
    Simula o sistema completo de transmissão e decodificação LDPC
    
    Args:
        bits: Bits de informação
        H: Matriz de verificação de paridade
        eb_n0_db: Eb/N0 em dB
        max_iteracoes: Número máximo de iterações do decodificador
        
    Returns:
        bits_decodificados: Bits decodificados
        ber: Taxa de erro de bit
        iteracoes: Número de iterações executadas
    """
    
    # Converte bits para símbolos BPSK
    simbolos = - 2 * bits + 1
    
    # Simula o canal AWGN
    simbolos_recebidos = add_awgn_noise(simbolos, eb_n0_db)
    
    # Calcula os LLRs do canal
    llrs_canal = calculate_llr(simbolos_recebidos, eb_n0_db)
    
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
    dc = 7
    max_iteracoes = 50
    
    # Gera a matriz H
    print("Gerando matriz LDPC...")
    N = encontrar_N_proximo(valor_alvo, dv, dc)
    H = criar_matriz_verificacao_ldpc(N, dv, dc)
    A, B = converter_H_para_AB(H)
    
    # Gera bits aleatórios
    print("Testando decodificador...")
    # bits = np.random.randint(0, 2, N)
    bits = np.zeros(N)
    
    # Testa para diferentes valores de Eb/N0
    eb_n0_db_range = np.arange(0, 6.1, 0.5)
    # eb_n0_db_range = np.arange(3, 5, 0.5)
    ber_values = []
    iter_values = []
    
    print("\n=== Iniciando simulações ===")
    for eb_n0_db in eb_n0_db_range:
        print(f"\nSimulando Eb/N0 = {eb_n0_db} dB")
        _, ber, iteracoes = simular_sistema_ldpc(bits, A, B, eb_n0_db, max_iteracoes)
        ber_values.append(ber)
        iter_values.append(iteracoes)
        print(f"Resultado: BER = {ber:.6f}, Iterações = {iteracoes}")
    
    # Plota os resultados
    print("\n=== Gerando gráficos ===")
    plt.figure(figsize=(12, 10))
    
    # Gráfico da BER
    plt.subplot(2, 1, 1)
    plt.semilogy(eb_n0_db_range, ber_values, 'bo-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.title(f'Desempenho do Código LDPC (N={N}, dv={dv}, dc={dc})')
    
    # Gráfico das iterações
    plt.subplot(2, 1, 2)
    plt.plot(eb_n0_db_range, iter_values, 'ro-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('Número de Iterações')
    plt.title('Convergência do Algoritmo')
    
    plt.tight_layout()
    plt.savefig('ldpc_performance.png')
    plt.show()
    
    print("Simulação concluída! Resultados salvos em 'ldpc_performance.png'")
