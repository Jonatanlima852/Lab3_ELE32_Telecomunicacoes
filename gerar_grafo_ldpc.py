import numpy as np
from typing import Tuple

def criar_matriz_verificacao_ldpc(N: int, dv: int, dc: int) -> np.ndarray:
    """
    Cria uma matriz de verificação de paridade H para um código LDPC regular.
    
    Args:
        N: Número de colunas (comprimento da palavra código)
        dv: Grau dos nós variáveis (número de 1s por coluna)
        dc: Grau dos nós de verificação (número de 1s por linha)
        
    Returns:
        H: Matriz de verificação de paridade (M x N)
    """
    # Calcula o número de linhas M baseado nos parâmetros
    # N*dv = M*dc (número total de 1s deve ser igual olhando por linhas ou colunas)
    M = (N * dv) // dc
    
    if (N * dv) % dc != 0:
        raise ValueError(f"Parâmetros inválidos: N*dv ({N*dv}) deve ser divisível por dc ({dc})")
    
    # Inicializa matriz com zeros
    H = np.zeros((M, N), dtype=int)
    
    # Para cada coluna (v-node)
    for j in range(N):
        # Lista de linhas disponíveis (onde ainda podemos colocar 1s)
        linhas_disponiveis = [i for i in range(M) if np.sum(H[i]) < dc]
        
        # Se não há linhas suficientes disponíveis, recomeça
        if len(linhas_disponiveis) < dv:
            return criar_matriz_verificacao_ldpc(N, dv, dc)
        
        # Escolhe aleatoriamente dv linhas para colocar 1s
        linhas_escolhidas = np.random.choice(linhas_disponiveis, size=dv, replace=False)
        
        for i in linhas_escolhidas:
            H[i, j] = 1
    
    # Verifica se todas as restrições foram atendidas
    if not _verificar_matriz(H, dv, dc):
        # Se não foram, tenta criar novamente
        return criar_matriz_verificacao_ldpc(N, dv, dc)
    
    return H

def _verificar_matriz(H: np.ndarray, dv: int, dc: int) -> bool:
    """
    Verifica se a matriz H atende a todas as restrições do código LDPC regular.
    """
    M, N = H.shape
    
    # Verifica grau dos v-nodes (colunas)
    if not all(np.sum(H[:, j]) == dv for j in range(N)):
        return False
    
    # Verifica grau dos c-nodes (linhas)
    if not all(np.sum(H[i, :]) == dc for i in range(M)):
        return False
    
    return True

def calcular_taxa_codigo(H: np.ndarray) -> float:
    """
    Calcula a taxa do código LDPC.
    
    Args:
        H: Matriz de verificação de paridade
        
    Returns:
        Taxa do código (R = k/n)
    """
    M, N = H.shape
    return (N - M) / N

def converter_H_para_AB(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converte a matriz de verificação de paridade H para as matrizes A e B.
    
    Args:
        H: Matriz de verificação de paridade (M x N)
        
    Returns:
        A: Matriz N x dv onde cada linha i contém os índices dos c-nodes conectados ao v-node i
        B: Matriz M x dc onde cada linha i contém os índices dos v-nodes conectados ao c-node i
    """
    M, N = H.shape
    
    # Encontra dv e dc a partir da matriz H
    dv = np.sum(H[:, 0])  # número de 1s na primeira coluna
    dc = np.sum(H[0, :])  # número de 1s na primeira linha
    
    # Inicializa as matrizes A e B
    A = np.zeros((N, dv), dtype=int)
    B = np.zeros((M, dc), dtype=int)
    
    # Preenche a matriz A
    for i in range(N):
        # Encontra os índices dos c-nodes conectados ao v-node i
        c_nodes = np.where(H[:, i] == 1)[0]
        if len(c_nodes) != dv:
            raise ValueError(f"V-node {i} não tem exatamente {dv} conexões")
        A[i, :] = c_nodes
    
    # Preenche a matriz B
    for i in range(M):
        # Encontra os índices dos v-nodes conectados ao c-node i
        v_nodes = np.where(H[i, :] == 1)[0]
        if len(v_nodes) != dc:
            raise ValueError(f"C-node {i} não tem exatamente {dc} conexões")
        B[i, :] = v_nodes
    
    return A, B

def encontrar_N_proximo(valor_alvo: int, dv: int, dc: int) -> int:
    """
    Encontra o valor de N mais próximo do valor alvo que satisfaz as restrições do código LDPC.
    
    Args:
        valor_alvo: Valor desejado para N
        dv: Grau dos nós variáveis
        dc: Grau dos nós de verificação
        
    Returns:
        Valor válido de N mais próximo do alvo
    """
    # N*dv deve ser divisível por dc para termos um número inteiro de linhas
    # Encontra o múltiplo mais próximo
    N = valor_alvo
    while (N * dv) % dc != 0:
        N += 1
    return N

def exportar_grafo_csv(A: np.ndarray, N: int, nome_arquivo: str = "grafo_ldpc.csv"):
    """
    Exporta o grafo LDPC para um arquivo CSV.
    Cada linha representa um v-node e seus c-nodes conectados.
    
    Args:
        A: Matriz de conexões v-node para c-node
        N: Número de v-nodes
        nome_arquivo: Nome do arquivo CSV de saída
    """
    with open(nome_arquivo, 'w') as f:
        for i in range(N):
            # Converte a linha para string e remove os colchetes
            linha = str(A[i].tolist())[1:-1]
            f.write(linha + '\n')

def gerar_grafo_ldpc():
    """
    Gera grafos LDPC com taxa aproximada de 4/7 para diferentes valores de N.
    Usa dv=6 e dc=14 para atingir a taxa desejada.
    """
    # Parâmetros fixos
    dv = 6  # grau dos nós variáveis
    dc = 14  # grau dos nós de verificação
    
    # Taxa teórica = 1 - dv/dc = 1 - 6/14 ≈ 0.571 (próximo de 4/7 ≈ 0.571)
    taxa_teorica = 1 - dv/dc
    
    # Valores alvo para N
    valores_alvo = [100]
    
    resultados = []
    
    # Para cada valor alvo, encontra o N mais próximo válido e gera a matriz
    for valor_alvo in valores_alvo:
        N = encontrar_N_proximo(valor_alvo, dv, dc)
        
        print(f"\nGerando matriz para N ≈ {valor_alvo}")
        print(f"N ajustado = {N}")
        
        try:
            # Cria a matriz H
            H = criar_matriz_verificacao_ldpc(N, dv, dc)
            
            # Calcula e exibe as propriedades da matriz
            M = H.shape[0]
            taxa_real = calcular_taxa_codigo(H)
            
            print(f"Dimensões da matriz H: {M} x {N}")
            print(f"Taxa teórica: {taxa_teorica:.6f}")
            print(f"Taxa real: {taxa_real:.6f}")
            
            # Converte H para matrizes A e B (grafo de Tanner)
            A, B = converter_H_para_AB(H)
            print(f"Matrizes de conexão geradas: A ({A.shape}) e B ({B.shape})")
            
            # Salva as matrizes em arquivos
            nome_arquivo_H = f"matriz_ldpc_H_N{N}.npy"
            nome_arquivo_A = f"matriz_ldpc_A_N{N}.npy"
            nome_arquivo_B = f"matriz_ldpc_B_N{N}.npy"
            
            np.save(nome_arquivo_H, H)
            np.save(nome_arquivo_A, A)
            np.save(nome_arquivo_B, B)
            
            # Exporta o grafo em formato CSV
            exportar_grafo_csv(A, N, f"grafo_ldpc_N{N}.csv")
            
            print(f"Matrizes salvas em: {nome_arquivo_H}, {nome_arquivo_A}, {nome_arquivo_B}")
            print(f"Grafo exportado em: grafo_ldpc_N{N}.csv")
            
            # Verificações adicionais
            print("\nVerificações:")
            print(f"- Todos v-nodes têm grau {dv}:", all(np.sum(H[:, j]) == dv for j in range(N)))
            print(f"- Todos c-nodes têm grau {dc}:", all(np.sum(H[i, :]) == dc for i in range(M)))
            
            resultados.append((N, M, H, A, B))
            
        except ValueError as e:
            print(f"Erro ao gerar matriz: {e}")
    
    return resultados


# Geração de matrizes LDPC para teste
if __name__ == "__main__":
    print("=== Gerando grafos LDPC ===")
    resultados = gerar_grafo_ldpc()
    print("\n=== Geração concluída ===")
