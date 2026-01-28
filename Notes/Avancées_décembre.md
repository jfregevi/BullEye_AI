# État d'avancement : Sélection et optimisation du modèle

Le problème principal évoqué lors de mes précédentes avancées étaient le choix d'un modèle efficace. L'objectif de Décembre a donc été de comparer des modèles afin de sélectionner le plus optimal.

### Méthodologie de comparaison

La méthode de comparaison est la suivante (voir le Notebook) :

1. **Fixer les variables nécessaires** (données financières, taille des fenêtres, horizon de prédiction, etc.) ;
2. **Entraîner** plusieurs modèles ;
3. **Stocker la RMSE** de chacun de ces modèles sur la BDD de test ;
4. **Comparer les RMSE** de tous les modèles ;
5. **Sélectionner** les 5 meilleurs ;
6. **Demander à Gemini** de générer des modèles à ces 5 derniers ;
7. **Itérer** le processus à partir du point 2.

Pour le premier tour de test, il fallait sélectionner différents types classiques de modèles. Pour ce faire, je me suis renseigné sur les différentes familles de modèle. Vous trouverez le document issu de nombreux échanges avec Gemini pour générer un prompt adapté à envoyer à Perplexity dans le sous-dossier *documentation* du dossier *modèles*.

---

### Architecture du modèle sélectionné

Le modèle sectionné est donc :

```python
def create_attention_tcn_gru():
    inputs = Input(shape=(window_size, num_features))
    
    # Branche 1 : TCN
    # Correction : On utilise 'dropout_rate' au lieu de 'dropout'
    tcn_out = TCN(nb_filters=64, 
                  kernel_size=3, 
                  dilations=[1, 2, 4], 
                  return_sequences=True, 
                  dropout_rate=0.1)(inputs)
    
    # Branche 2 : BiGRU
    gru_out = Bidirectional(GRU(64, return_sequences=True))(inputs)
    
    # Mécanisme d'Attention
    # L'attention compare les deux branches pour extraire les caractéristiques saillantes
    query = Dense(128)(tcn_out)
    value = Dense(128)(gru_out)
    attn_out = Attention()([query, value])
    
    # Global Pooling pour réduire la dimension temporelle avant la sortie
    avg_pool = GlobalAveragePooling1D()(attn_out)
    
    dense = Dense(32, activation='leaky_relu')(avg_pool)
    dense = Dropout(0.2)(dense)
    output = Dense(1)(dense)
    
    model = Model(inputs=inputs, outputs=output)
    
    return model

```

---

### Problème rencontré : Le biais de normalisation (Data Leakage)

Pb recontré : dans tous mes essaies, je rescale les valeurs de X et de Y sur l'ensemble des données étudiées (même la partie test). Or en pratique, je ne pourrai pas utiliser des valeurs futures pour rescale, donc il se peut que mon modèle délire en voyant des valeurs plus grande dans le futur.

#### Pourquoi c'est de la "triche" sur ton graphique actuel :

Dans ta Cellule 2, tu fais :

1. Téléchargement de 10 ans (ex: 2014 à 2024).
2. `fit_transform` sur **toute** la période.
3. Affichage du graphique sur les 5% derniers (le `X_test`).

**Le problème :** Ton `X_test` (les données récentes) a été utilisé pour calculer le Min et le Max qui servent à normaliser ton `X_train` (le passé).

* Si le point le plus haut de l'action était en 2024, ton scaler le sait déjà quand il normalise l'année 2018.
* Ton graphique de test est "trop beau pour être vrai" car chaque point de test a été normalisé avec une connaissance globale de la période.

> **Conclusion :** cela démontre en problème intrinsèque à la volonté de travailler avec des valeurs de cours plutôt qu'avec des variations.
