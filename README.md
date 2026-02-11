# RL Marble Game - Comprehensive Documentation

Benvenuto nel progetto **RL Marble Game**. Questa documentazione è progettata per fornirti una visione profonda del sistema, permettendoti di capire non solo *cosa* fa il codice, ma *perché* lo fa, e come puoi modificarlo in sicurezza.

---

## 1. Visione d'Insieme del Sistema

Il progetto è una simulazione fisica in cui un agente di Reinforcement Learning (RL) deve imparare a inclinare una tavola per guidare una pallina attraverso un labirinto dinamico fino a un obiettivo, evitando di cadere nei buchi o fuori dai bordi.

### Componenti Principali:
- **Physics Engine (`marble_game.py`)**: Utilizza **PyBullet** per gestire le collisioni, la gravità e l'attrito.
- **Gym Environment (`marble_env.py`)**: Un wrapper che trasforma la simulazione fisica in un problema di RL compatibile con Gymnasium.
- **Deep Q-Network Agent (`dqn_agent.py`)**: Il "cervello" che sceglie le azioni basandosi sulla situazione attuale.
- **Maze Generator (`maze_generator.py`)**: Crea labirinti casuali (ma risolvibili) e calcola le mappe di distanza BFS.

---

## 2. Il "Cervello": Deep Q-Network (DQN)

L'agente utilizza un algoritmo **DQN** con alcune estensioni moderne per stabilità ed efficienza.

### L'Architettura della Rete (`QNetwork`):
Riceve in input lo stato (8 valori) e restituisce 9 valori di "Q-value" (uno per ogni possibile azione).
- **Livello Input**: 8 neuroni (posizione, velocità, posizione obiettivo, prossima cella BFS).
- **Hidden Layers**: 2 livelli lineari da **128 neuroni** ciascuno con attivazione **ReLU**.
- **Livello Output**: 9 neuroni (corrispondenti alle inclinazioni della tavola).

### Funzionamento:
- **Target Network**: Usiamo una seconda rete (identica alla prima) per calcolare i target di apprendimento. Viene aggiornata ogni `target_update` (2000 step) per evitare che la rete "insegua la sua stessa coda", rendendo il training molto più stabile.
- **Replay Buffer**: L'agente salva le esperienze passate (`stato, azione, reward, next_state`) e ne pesca blocchi casuali (batch da 64) per imparare dai propri successi e fallimenti.
- **Oversampling**: Per accelerare l'apprendimento, le esperienze che portano alla vittoria sono salvate in un buffer speciale e riutilizzate più frequentemente.

---

## 3. Lo Stato e le Azioni

### Lo Stato (Observation Space):
L'agente vede 8 valori normalizzati:
1. **Palla (x, y)**: Coordinate relative al centro della tavola.
2. **Velocità (vx, vy)**: Quanto velocemente si muove la pallina.
3. **Goal (target_x, target_y)**: La posizione finale da raggiungere.
4. **Prossima Cella BFS (nx, ny)**: La direzione suggerita dall' algoritmo di pathfinding per uscire dal labirinto.

### Le Azioni (Action Space):
9 azioni discrete che corrispondono ai tasti direzionali:
- `(0,0)`: Mantieni piatta.
- `(1,0), (-1,0), (0,1), (0,-1)`: Inclina in una direzione.
- `(1,1), (-1,1), (1,-1), (-1,-1)`: Inclinazioni diagonali.

---

## 4. Il Sistema di Reward (Il "Motivatore")

Questo è il cuore del comportamento dell'agente. Il reward totale è la somma di due componenti:

### A. Rewards Estrinseci (Task-based):
1. **Successo (+50.0)**: Assegnato quando la pallina tocca la cella obiettivo.
2. **Caduta (-20.0)**: Penalità per finire in un buco o fuori bordo.
3. **BFS Progression (+5.0)**: Quando l'agente "salta" da una cella del labirinto a quella successiva più vicina al goal.
4. **Local Centering (+1.0 * miglioramento)**: Un piccolo premio per avvicinarsi al centro della cella target locale (mantiene il movimento fluido).
5. **Stagnation Penalty (-0.01)**: Se l'agente sta fermo troppo a lungo.

### B. Rewards Intrinseci (Curiosità):
- **Discovery Bonus (2.0 / visite)**: Assegnato **solo la prima volta** che l'agente visita una cella in un episodio. Più una cella è stata visitata globalmente nel tempo, più questo bonus diminuisce. Questo spinge l'agente ad andare dove non è quasi mai stato.

---

## 5. Strategie Avanzate di Esplorazione

### Novelty-Based Epsilon
In un labirinto grande, l'epsilon standard (casuale) decade troppo in fretta. Noi usiamo un **Effective Epsilon**:
- `effective_epsilon = max(global_epsilon, 1.0 / (1.0 + ln(visits)))`
- **Risultato**: Anche se siamo all' episodio 1000, se l'agente entra in una zona nuova, l'epsilon sale istantaneamente, costringendo l'agente a esplorare invece di seguire la sua policy (che lì non conosce ancora).

### Action Masking
Per evitare che l'alto epsilon causi cadute continue, l'agente riceve una **maschera**:
- Le azioni che porterebbero la pallina contro un muro vengono "nascoste" all'agente durante la selezione dell'azione casuale. Questo rende l'esplorazione sicura ed efficiente.

---

## 6. Guida ai File

- `train.py`: Il motore del training. Qui regoli gli iperparametri come `learning_rate`, `epsilon_decay` e la frequenza di salvataggio.
- `marble_env.py`: Qui modifichi le regole del gioco e i pesi dei reward. Se vuoi che l'agente sia più prudente, alza la `fall_penalty`.
- `dqn_agent.py`: Contiene la logica neurale. Modifica `QNetwork` se vuoi una rete più profonda o larga.
- `marble_game.py`: Gestisce la fisica. Qui puoi cambiare la gravità, la velocità di inclinazione o le dimensioni del labirinto.
- `maze_generator.py`: Logica di creazione del labirinto.

---

## 7. Consigli per il Debugging

1. **Heatmaps**: Controlla `checkpoints/heatmap_ep_*.png` per vedere se l'agente sta esplorando tutto il maze o se si blocca in un angolo.
2. **Epsilon Map**: Controlla `checkpoints/epsilon_ep_*.png` per capire dove l'agente si sente ancora insicuro.
3. **TensorBoard**: Lancia `tensorboard --logdir checkpoints/runs` per analizzare curve di reward, loss e progressione in tempo reale.
4. **Reward Explosion**: Se vedi reward altissimi improvvisi, controlla che non ci siano "salti" nei valori BFS o cicli infiniti di reward di scoperta.

---

## 8. Come Iniziare

### Training
Per far ripartire l'addestramento da zero (o dai pesi salvati):
```bash
python train.py
```

### Testing (Visualizzazione)
Per vedere l'agente in azione con la grafica 3D attiva:
```bash
python test_agent.py --episodes 5
```

### Analisi
Per vedere i progressi con TensorBoard:
```bash
tensorboard --logdir checkpoints/runs
```

---
*Questa documentazione riflette lo stato attuale del sistema dopo le ottimizzazioni di esplorazione e mapping.*
