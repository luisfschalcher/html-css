import numpy
import random
import matplotlib.pyplot as plt

# Definições do ambiente
num_linhas = 9
num_colunas = 14
robo_pos_inicial = (3, 1)
obstaculo_pos_inicial = (1, 7)
destino = (3, 11)

# Definições das constantes
alpha = 0.05
gamma = 0.5
epsilon = 0.05
episodios = 1000

# Inicializando a tabela Q
Q = numpy.zeros((num_linhas, num_colunas, num_linhas, num_colunas, 9))  # 9 ações possíveis, duas de cada (num linhas e colunas) para representar as posições do robô e obstáculo

# Função de recompensa
def recompensa(pos_robo, destino, colidiu):
  if pos_robo == destino:
    return 10
  elif colidiu:
    return -10
  else:
    return -1  # Pequena penalidade, para que o robô não vague sem rumo

# Função para escolher ação usando política epsilon-greedy
def escolher_acao(estado):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 8)
    else:
        return numpy.argmax(Q[estado])

# Função para mover o obstáculo
def mover_obstaculo(pos):
    velocidade = random.randint(1, 3)
    nova_linha = (pos[0] + velocidade) % num_linhas
    nova_pos = (nova_linha, pos[1])
    return nova_pos

# Função para mover o robô
def mover_robo(pos, acao):
    if acao == 0 and pos[0] > 0:  # cima
        return (pos[0] - 1, pos[1])
    elif acao == 1 and pos[0] < num_linhas - 1:  # baixo
        return (pos[0] + 1, pos[1])
    elif acao == 2 and pos[1] > 0:  # esquerda
        return (pos[0], pos[1] - 1)
    elif acao == 3 and pos[1] < num_colunas - 1:  # direita
        return (pos[0], pos[1] + 1)
    elif acao == 4 and pos[0] > 0 and pos[1] > 0:  # noroeste
        return (pos[0] - 1, pos[1] - 1)
    elif acao == 5 and pos[0] > 0 and pos[1] < num_colunas - 1:  # nordeste
      return (pos[0] - 1, pos[1] + 1)
    elif acao == 6 and pos[0] < num_linhas - 1 and pos[1] > 0:  # sudoeste
      return (pos[0] + 1, pos[1] - 1)
    elif acao == 7 and pos[0] < num_linhas - 1 and pos[1] < num_colunas - 1:  # sudeste
      return (pos[0] + 1, pos[1] + 1)
    elif acao == 8:  # parar
      return pos
    return pos

# Função para imprimir o mapa com o estado atual
def imprimir_mapa(pos_robo, pos_obstaculo):
    mapa = [['_' for _ in range(num_colunas)] for _ in range(num_linhas)]
    mapa[pos_robo[0]][pos_robo[1]] = 'R'  # Robô
    mapa[pos_obstaculo[0]][pos_obstaculo[1]] = 'O'  # Obstáculo
    mapa[destino[0]][destino[1]] = 'D'  # Destino

    for linha in mapa:
        print(' '.join(linha))
    print('\n')










# Treinamento
recompensas_episodio = []
for episodio in range(episodios):
    pos_robo = robo_pos_inicial[:]
    pos_obstaculo = obstaculo_pos_inicial[:]
    episodio_recompensa = 0
    while pos_robo != destino:
        estado = (*pos_robo, *pos_obstaculo)
        acao = escolher_acao(estado)
        nova_pos_robo = mover_robo(pos_robo, acao)
        pos_obstaculo = mover_obstaculo(pos_obstaculo)
        colidiu = nova_pos_robo == pos_obstaculo
      
        r = recompensa(nova_pos_robo, destino, colidiu)
        novo_estado = (*nova_pos_robo, *pos_obstaculo)

        Q[estado][acao] = Q[estado][acao] + alpha * (r + gamma * numpy.max(Q[novo_estado]) - Q[estado][acao])

        episodio_recompensa += r

        if colidiu:
            break
        pos_robo = nova_pos_robo

        recompensas_episodio.append(episodio_recompensa)  # Adiciona a recompensa do episódio à lista

# Plotando as recompensas ao longo dos episódios
plt.figure(figsize=(10, 6))
plt.plot(numpy.arange(len(recompensas_episodio)), recompensas_episodio)
plt.xlabel('Episódios')
plt.ylabel('Recompensa')
plt.title('Evolução da Recompensa ao Longo dos Episódios')
plt.grid(True)
plt.show()

# Testando o treinamento
pos_robo = robo_pos_inicial[:]
pos_obstaculo = obstaculo_pos_inicial[:]
imprimir_mapa(pos_robo, pos_obstaculo)
caminho_robo = [pos_robo]
caminho_obstaculo = [pos_obstaculo]

while pos_robo != destino:
    estado = (*pos_robo, *pos_obstaculo)
    acao = numpy.argmax(Q[estado])
    pos_robo = mover_robo(pos_robo, acao)
    pos_obstaculo = mover_obstaculo(pos_obstaculo)
    imprimir_mapa(pos_robo, pos_obstaculo)
    caminho_robo.append(pos_robo)
    caminho_obstaculo.append(pos_obstaculo)