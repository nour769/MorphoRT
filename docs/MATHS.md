# Formules mathématiques de MorphoRT

Les éléments structurants sont des boules discrètes définies en millimètres :
le vecteur de voxel `s` convertit les coordonnées d'indice en distance réelle.
L'implémentation est dans `morphort/morphology.py`.

## Morphologie binaire

### Érosion

**A ⊖ B = {x | x + B ⊆ A}.** Un pixel reste dans le masque si tout son
voisinage structurant tient dans la région.

### Dilatation

**A ⊕ B = ⋃ᵦ (A + b).** La dilatation ajoute autour de la région les positions
couvertes par l'élément structurant.

### Ouverture

**A ○ B = (A ⊖ B) ⊕ B.** L'érosion enlève les petites structures puis la
dilatation redonne leur taille aux formes qui ont résisté.

### Fermeture

**A ● B = (A ⊕ B) ⊖ B.** La dilatation comble les petits creux puis l'érosion
restaure les dimensions générales de la région.

### Gradient morphologique

**G(A) = (A ⊕ B) \ (A ⊖ B).** La différence entre expansion et contraction
met en évidence la frontière de la forme.

### Top-hat blanc et noir

**TH_blanc = A \ (A ○ B)** et **TH_noir = (A ● B) \ A**. Ils extraient
respectivement les détails clairs supprimés par ouverture et les petits trous
comblés par fermeture.

### Reconstruction géodésique

**R_M(X) = ⋃ₖ ((X ⊕ kB) ∩ M).** Le marqueur s'étend dans le masque sans franchir
sa frontière, jusqu'à ce qu'aucun pixel ne change.

### Ouverture par reconstruction

**O_R(A) = R_A(A ⊖ B).** L'érosion choisit les marqueurs, puis la reconstruction
restaure les contours des composantes conservées.

### Remplissage des trous

**Trous(A) = Aᶜ \ R_{Aᶜ}(bord(Aᶜ)).** On reconstruit le fond depuis le bord ;
les régions de fond non atteintes sont les trous internes.

### Sélection des composantes par segments

**Rᵢⱼ = [aᵢⱼ, bᵢⱼ]** représente une suite contiguë de voxels vrais sur le dernier
axe. Deux segments de lignes voisines sont reliés si leurs intervalles se
touchent à un voxel près ; une fusion d'ensembles sélectionne la composante.
Cette représentation calcule les mêmes connexités 8/26 qu'un parcours voxel
par voxel, avec beaucoup moins de travail Python sur les volumes BraTS.

### Plus grande composante connexe

**C_max = argmax_C |C|.** Parmi les ensembles de pixels voisins, on conserve
celui qui contient le plus de pixels.

### Watershed

**coût(x) = min_chemins max_{y du chemin} relief(y).** Chaque marqueur propage
son étiquette par le chemin dont le point le plus haut est le moins élevé.

### Squelette morphologique

**S(A) = ⋃ₖ [(A ⊖ kB) \ ((A ⊖ kB) ○ B)].** Les couches qui disparaissent sous
ouverture gardent les centres des formes à chaque échelle.

### Granulométrie

**G(d) = |A ○ B_d| / |A|.** La fraction restante après ouverture décrit la
quantité de structure conservée lorsque le diamètre `d` augmente.

## Segmentation et mesures

### Normalisation robuste

**z(x) = (I(x) − médiane(I_cerveau)) / (1,4826 × MAD).** Le MAD est l'écart
absolu médian ; il limite l'influence des intensités extrêmes.

### Seuil d'Otsu

**t* = argmax_t ω₀(t) ω₁(t) [μ₀(t) − μ₁(t)]².** Le seuil est celui qui sépare
au mieux les moyennes des deux classes de l'histogramme.

### Tumeur rehaussée

**R₄ = C_max((T ∩ {z_T1ce > t_Otsu}) ○ B).** Un seuil T1ce normalisé au cerveau
est nettoyé par ouverture, puis seule sa plus grande composante est conservée.

### Noyau nécrotique

**R₁ = [fill_holes(R₄ • B) \ R₄] ∩ T ∩ {z_T1ce ≤ t_Otsu}.** Les cavités
reconstruites à l'intérieur de la région rehaussée ne sont gardées que si leur
signal T1ce est sous le seuil.

### Œdème

**R₂ = R_T([T ∩ {z_FLAIR ≥ P₅₀(z_FLAIR|T)}] \ (R₄ ∪ R₁)).** Le seuil FLAIR est
calculé dans la tumeur entière ; l'ouverture par reconstruction nettoie le
résidu sans modifier les formes conservées.

### Seuil par percentile

**t = P₉₀(z(x) : x ∈ cerveau).** Cette seconde méthode choisit le 90e percentile
des intensités normalisées dans le masque anatomique.

### Choix des paramètres

**θ* = argmax_θ moyenne(Diceᵢ(θ)), i ∈ groupe réglage.** Le meilleur triplet
(rayon d'ouverture, rayon de fermeture, méthode de seuil) est choisi parmi huit
combinaisons fixes, sans utiliser le groupe de test.

Les 20 identifiants BraTS triés sont répartis de façon déterministe par indices
pairs et impairs : 10 au réglage et 10 au test indépendant. Les scores du test
sont calculés une seule fois après le choix des paramètres.

### Volume physique

**V = N_voxels × dx × dy × dz.** Le produit de la taille des voxels par leur
nombre donne le volume ; 1 mL = 1 cm³ = 1000 mm³.

### Surface

**S = somme des aires des faces exposées.** Chaque face de voxel voisine du
fond contribue selon les deux dimensions physiques correspondantes.

### Sphéricité

**Ψ = π^(1/3) (6V)^(2/3) / S.** C'est le rapport entre la surface de la sphère
de même volume et la surface mesurée.

### Compacité

**C = S³ / (36πV²).** Cette mesure sans dimension vaut 1 pour une sphère idéale
et augmente lorsque la forme s'écarte de la sphéricité.

### Dice

**Dice(P,G) = 2|P ∩ G| / (|P| + |G|).** Le recouvrement entre masque prédit `P`
et masque expert `G` varie de 0 à 1.

### Jaccard

**J(P,G) = |P ∩ G| / |P ∪ G|.** La mesure compare l'intersection au volume total
occupé par les deux masques.

### Distance de Hausdorff

**H(P,G) = max{supₚ infᵍ d(p,g), supᵍ infₚ d(g,p)}.** La distance maximale
entre les surfaces est calculée en millimètres avec l'espacement des voxels.

### Hausdorff 95

**HD95(P,G) = max(P95(d(P,G)), P95(d(G,P))).** Le percentile 95 des distances
symétriques de surface réduit l'influence d'un point extrême isolé.

Le pipeline actuel prédit uniquement la tumeur entière. Les sous-régions BraTS
(labels 1, 2 et 4) sont affichées, mais ne reçoivent pas de score sans
prédictions morphologiques séparées.

## Croissance

### Gompertz

**V(t) = K exp(ln(V₀/K) exp(−rt)).** `r` décrit le taux de croissance et `K`
la capacité limite du modèle.

### Logistique

**V(t) = K / [1 + ((K−V₀)/V₀) exp(−rt)].** La croissance ralentit au fur et à
mesure que le volume approche `K`.

Les paramètres sont ajustés par moindres carrés non linéaires. Au moins trois
examens avec jours distincts sont exigés ; cela ne garantit pas la qualité d'un
modèle ni sa valeur clinique.

## Simulation radiobiologique

### Survie linéaire-quadratique

**S(d) = exp[−(αd + βd²)].** Le modèle associe un terme linéaire et un terme
quadratique pour décrire une survie cellulaire théorique.

### BED du tissu sain

**BED = Σᵢ dᵢ [1 + dᵢ/(α/β)_sain].** Les doses par fraction sont transformées en
une quantité biologique simplifiée et contrainte pendant l'optimisation.

### Optimisation

**min log(V_final), sous BED_sain ≤ BED_max et 0 ≤ dᵢ ≤ d_max.** SLSQP cherche
des fractions respectant les bornes tout en minimisant le volume simulé.
Le scénario est pédagogique, non calibré ni validé pour le traitement.
