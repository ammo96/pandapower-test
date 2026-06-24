# KI-gestützte Mustererkennung zur Charakterisierung von Netzrückwirkungen beim Laden von Elektrofahrzeugen

Dieses Repository enthält den offiziellen Quellcode für die Forschungsarbeit **"Entwicklung einer KI-gestützten Mustererkennung zur Charakterisierung von Netzrückwirkungen beim Laden von Elektrofahrzeugen"**. 

Das System implementiert eine vollständige Pipeline zur Erkennung, Klassifizierung und Interpretierbarkeit von 19 verschiedenen Klassen von Netzqualitätsstörungen (Power Quality Disturbances, PQDs). Die Pipeline reicht von der fortschrittlichen Signalverarbeitung auf Basis der generalisierten S-Transformation über das CNN-Training inklusive Hyperparameter-Optimierung bis hin zur modellinternen Visualisierung mittels Grad-CAM.

---

## 📊 Pipeline-Übersicht

```mermaid
graph TD
    %% Nodes
    A[Rohe 1D MATLAB/HDF5-Signale<br/>'pq_daten_19_klassen_sauber.mat'] 
    B[Fast Generalized S-Transform<br/>& AWGN Rauscherweiterung]
    C[Generierte 2D-Datensätze<br/>'.npz'-Dateien / 120x120]
    D[Hyperparameter-Optimierung<br/>Bayesian Sweep]
    E[Core Modell-Training<br/>NetzstoerungsCNN]
    F[Standalone Inferenz & Export<br/>Druckgrafiken / Grad-CAM Ordner]

    %% Pipeline Flow
    A -->|STFINAL.py| B
    B --> C
    C -->|wandB.py| D
    C -->|CNN-Modell.py| E
    E -->|Auswertung.py| F

    %% Styling
    classDef process fill:#f7fafc,stroke:#4a5568,stroke-width:2px,color:#2d3748;
    classDef data fill:#ebf8ff,stroke:#3182ce,stroke-width:2px,color:#2b6cb0;
    classDef output fill:#f0fff4,stroke:#38a169,stroke-width:2px,color:#276749;
    
    class A,C data;
    class B,D,E process;
    class F output;

---

## 🛠️ Beschreibung der Skripte

### 1. `STFINAL.py` — Signalverarbeitung & Datensatzgenerierung
Dieses Skript bildet das Fundament der Pipeline. Es transformiert die rohen 1D-Zeitsignale aus der MATLAB-Simulationsumgebung (`.mat`) in hochqualitative 2D-Zeit-Frequenz-Repräsentationen.

* **Zentrale Funktionen:**
  * **Fast Generalized S-Transform (GST):** Berechnet die S-Transformation mit dynamisch skalierten Gauß-Fenstern zur optimalen Zeit-Frequenz-Auflösung.
  * **Logarithmische Kompression & Perzentil-Skalierung:** Reduziert die Dynamik und bereinigt Ausreißer, um die Merkmale für das CNN zu optimieren.
  * **Daten-Augmentation (AWGN):** Beaufschlagt die sauberen Signale physikalisch korrekt mit additivem weißem gaußschem Rauschen für verschiedene Szenarien (Clean, Mixed 20–50 dB, Festes Rauschen bei 20 dB und 50 dB).
  * **Parallelisierung:** Nutzt `joblib`, um die Transformation aller Signale effizient über mehrere CPU-Kerne zu parallelisieren.
  * **Output:** Generiert komprimierte `.npz`-Dateien (Shape: `[N, 1, 120, 120]`) für das Training und Testen.

---

### 2. `CNN-Modell.py` — Core Training Pipeline
Dieses Skript definiert die neuronale Netzwerkarchitektur und steuert den primären Trainingsprozess über 40 Epochen auf den rauschbeaufschlagten Trainingsdaten.

* **Modell-Architektur (`NetzstoerungsCNN`):**
  * Drei aufeinanderfolgende Faltungsblöcke (Conv2D $\rightarrow$ ReLU $\rightarrow$ MaxPool2D) zur progressiven Feature-Extraktion.
  * **Spatial Dropout (`Dropout2d`, p=0.2):** Verhindert das gemeinsame Anpassen von Feature-Maps und wirkt Overfitting effektiv entgegen.
  * **Dense-Dropout (p=0.5):** Deaktiviert vor der finalen Klassifikationsschicht 50 % der Neuronen zur strikten Regularisierung.
* **Zentrale Funktionen:**
  * Trainiert das Modell mit dem Adam-Optimizer ($lr=0.0005$) und Cross-Entropy-Verlustfunktion.
  * Live-Metriken-Tracking (Loss, Accuracy) und automatische Generierung interaktiver Konfusionsmatrizen pro Epoche in **Weights & Biases (WandB)**.
  * Führt nach dem Training eine Post-hoc-Erklärbarkeitsanalyse mittels **Grad-CAM** durch und sichert die Gewichte (`.pth`) als WandB-Artefakt.

---

### 3. `wandB.py` — Hyperparameter-Sweep (Optimierung)
Zur systematischen Maximierung der Modellperformance implementiert dieses Skript eine automatisierte Hyperparameter-Suche.

* **Zentrale Funktionen:**
  * **Bayesianische Optimierung:** Durchsucht den vordefinierten Suchraum intelligent, um die Validierungsgenauigkeit (`val_acc`) zu maximieren.
  * **Dynamische Konfiguration:** Steuert Variablen wie Lernrate ($0.001, 0.0005, 0.0001$), Batch-Size ($16, 32, 64$) und den FC-Dropout-Faktor ($0.3, 0.4, 0.5, 0.6$) direkt über den WandB-Agenten.
  * Führt standardmäßig 10 aufeinanderfolgende Testläufe (Runs) durch, loggt für jeden Versuch detaillierte Klassifikationsberichte sowie Grad-CAM-Tabellen und speichert die jeweiligen Modellgewichte separat ab.

---

### 4. `Auswertung.py` — Standalone Inferenz & Druckgrafik-Export
Dieses Skript dient der finalen Validierung trainierter Modelle auf unabhängigen Test-Szenarien und erzeugt hochauflösende Visualisierungen für die schriftliche Ausarbeitung.

* **Zentrale Funktionen:**
  * **Umfassende Inferenz:** Berechnet Vorhersagen, Wahrscheinlichkeiten und speichert rohe Vorhersagedaten lokal für post-hoc Analysen.
  * **Klassifikationsbericht:** Gibt Präzision, Recall und F1-Score differenziert für alle 19 diskreten Störungsklassen aus (inkl. Kombinationsstörungen wie *Harmonics+Sag+Flicker*).
  * **High-Res Vektor-Export:** Generiert eine hochauflösende Vektor-PDF der Konfusionsmatrix (`Konfusionsmatrix_Druckqualitaet.pdf`) im passenden Farbschema (`Blues`).
  * **Optimiertes Grad-CAM Visualisierungs-System:** Berechnet die Aktivierungs-Heatmaps der letzten Faltungsschicht (`conv3`) für dedizierte Schlüsselindizes. Wendet eine **Gamma-Korrektur** ($\gamma = 0.5$) auf das S-Transformationsbild an, um schwache Frequenzkomponenten für das menschliche Auge im finalen PNG-Overlay sichtbar zu machen.

---

## 📋 Voraussetzungen & Installation

Das Projekt basiert auf **PyTorch** und nutzt **CUDA** zur Hardwarebeschleunigung auf NVIDIA-GPUs (getestet mit PyTorch 2.6.0 und CUDA 12.4).

### Abhängigkeiten installieren:
```bash
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
pip install numpy scipy scikit-image scikit-learn matplotlib joblib h5py wandb pytorch-grad-cam

