# Note de recherche - 03/01/2026

## Réflexions sur la structure des données

* **Passage aux rendements :** Il est impératif de passer en **% de variation** plutôt qu'en valeurs de prix brutes. C'est mathématiquement plus robuste et plus parlant pour l'analyse.
* **Adaptabilité du modèle :** Maintenant qu'un modèle performant a été identifié, il faudra valider s'il reste pertinent pour prédire des variations ou s'il doit être ajusté.

## Évolution de l'objectif métier

L'objectif de mon interface est d'indiquer à l'investisseur si le prix va **monter ou descendre**. La valeur exacte du prix importe peu en soi.

## Conclusion : Pivot vers la Classification

L'idée est de passer sur un modèle de **classification** qui prédit, par exemple :

* **Classe 0 :** Hausse (> +2%)
* **Classe 1 :** Baisse (> -2%)
* **Classe 2 :** Stabilité (entre -2% et +2%)



# Avis de Gemini sur ce changement de paradigme :

### Mon avis : Est-ce plus pertinent et moins "hallucinant" ?

**OUI, mille fois oui.** Passer à la classification est une décision très mature techniquement pour trois raisons majeures :

#### 1. Tu élimines "l'effet miroir" (Lagging)

En régression, les modèles de Deep Learning ont tendance à "tricher" en prédisant simplement pour demain le prix d'aujourd'hui (avec un léger décalage). C'est ce qui donne des courbes magnifiques mais inutilisables. En classification, le modèle est forcé de choisir une direction. Soit il a raison, soit il a tort. Il ne peut plus se contenter de "suivre" la courbe.

#### 2. C'est beaucoup moins sensible au bruit

Prédire si une action va faire $152.43 ou $152.47 est impossible (c'est du bruit pur). Prédire si elle va prendre plus de 2% est un signal statistique beaucoup plus "solide". En créant une zone "Stabilité" (ta classe 2), tu filtres tout le bruit de marché qui ne sert à rien à l'investisseur.

#### 3. Ton modèle actuel est-il encore bon ?

Bonne nouvelle : Ton modèle **Attention + TCN + GRU** est en réalité **meilleur** pour la classification que pour la régression.

* Le **TCN** capte les motifs de formes (ex: une "tasse avec anse").
* Le **GRU** capte la tendance lourde.
* L'**Attention** va focus sur les pics de volume qui précèdent souvent les gros mouvements de classe 0 ou 1.

**Le seul changement technique :**

* Ta dernière couche sera `Dense(3, activation='softmax')` (pour tes 3 classes).
* Ta perte sera `categorical_crossentropy` au lieu de `mse`.

### Verdict

C'est beaucoup plus "pertinent" et surtout plus "réaliste". Un modèle qui te dit "J'ai 70% de probabilité qu'on monte de plus de 2%" est bien plus utile qu'un modèle qui te dit "Le prix sera de 154.22$" (ce qui est presque toujours faux).
