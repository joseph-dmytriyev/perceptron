# Perceptron — Réseau de neurones en Python pur

Implémentation d'un réseau de neurones multicouche (MLP) from scratch, sans framework de deep learning. Le projet illustre les mécanismes fondamentaux d'un perceptron : calcul pondéré, biais, fonctions d'activation et propagation avant.

---

## Architecture

Le projet suit une composition hiérarchique en trois niveaux :

```
Network
  └─ Layer(s)
      └─ Neuron(s)
```

| Classe | Fichier | Rôle |
|---|---|---|
| `Neuron` | `neuron.py` | Unité de base : calcul pondéré + biais |
| `Layer` | `layer.py` | Groupe de neurones recevant les mêmes entrées |
| `Network` | `network.py` | Chaîne de couches avec propagation avant |

---

## Fonctionnement

### 1. Neurone (`neuron.py`)

Chaque neurone applique la formule du perceptron :

```
z = Σ(wᵢ × xᵢ) + b
```

- `wᵢ` : poids associés à chaque entrée
- `xᵢ` : valeurs d'entrée
- `b`  : biais

### 2. Couche (`layer.py`)

Une couche contient plusieurs neurones. Toutes les entrées sont transmises à chaque neurone, et la couche retourne un vecteur de sorties (une par neurone).

### 3. Réseau (`network.py`)

Le réseau enchaîne plusieurs couches. À chaque couche, la fonction d'activation est appliquée sur les sorties brutes avant de les passer à la couche suivante :

```
a⁽ˡ⁾ = activation( Layer_l.forward( a⁽ˡ⁻¹⁾ ) )
```

---

## Fonctions d'activation (`activation.py`)

| Fonction | Formule | Usage |
|---|---|---|
| Identity | `f(x) = x` | Sortie brute sans transformation |
| Threshold | `f(x) = 1 si x ≥ 0, sinon 0` | Perceptron binaire classique |
| ReLU | `f(x) = max(0, x)` | Couches cachées de réseaux profonds |
| Sigmoid* | `f(x) = 1 / (1 + e⁻ˣ)` | Utilisée dans la démo (`main.py`) |

> *La sigmoid est définie dans `main.py` via la bibliothèque standard `math`.

---

## Exemple — réseau 3 couches

```python
from network import Network
import math

sigmoid = lambda x: 1 / (1 + math.exp(-x))
net = Network(input_size=3, activation=sigmoid)

# Couche 1 : 2 neurones, 3 entrées chacun
net.add(weights=[[0.2, -0.5, 0.1], [0.4, 0.3, -0.2]],
        biases=[0.1, -0.1])

# Couche 2 : 3 neurones, 2 entrées chacun
net.add(weights=[[0.3, -0.4], [0.5, 0.2], [-0.1, 0.6]],
        biases=[0.0, 0.1, -0.2])

# Couche 3 : 2 neurones (sortie), 3 entrées chacun
net.add(weights=[[0.7, -0.3, 0.2], [-0.5, 0.4, 0.1]],
        biases=[0.1, -0.1])

output = net.feedforward([1.0, 0.5, -0.3])
print(output)  # ex: [0.623, 0.489]
```

---

## Structure des fichiers

```
perceptron/
├── neuron.py        # Classe Neuron
├── layer.py         # Classe Layer
├── network.py       # Classe Network
├── activation.py    # Fonctions d'activation
└── main.py          # Démonstration et tests
```

---

## Limites

Ce projet implémente uniquement la **propagation avant** (feedforward). Il n'y a pas de :
- rétropropagation (backpropagation)
- algorithme d'entraînement (gradient descent, Adam…)
- gestion des données (datasets, batches)

L'objectif est pédagogique : comprendre et manipuler les briques fondamentales d'un réseau de neurones.
