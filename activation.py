def act_identity(x):
    """
    Fonction d'activation identité.
    Retourne la valeur telle quelle.
    
    Args:
        x (float): Valeur d'entrée
        
    Returns:
        float: x (inchangé)
    """
    return x


def act_threshold(x):
    """
    Fonction d'activation seuil (step function).
    Retourne 1 si x >= 0, sinon 0.
    
    Args:
        x (float): Valeur d'entrée
        
    Returns:
        int: 1 si x >= 0, sinon 0
    """
    return 1 if x >= 0 else 0


def act_relu(x):
    """
    Fonction d'activation ReLU (Rectified Linear Unit).
    Retourne max(0, x).
    
    Args:
        x (float): Valeur d'entrée
        
    Returns:
        float: max(0, x)
    """
    return max(0, x)