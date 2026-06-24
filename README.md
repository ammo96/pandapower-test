# KI-gestützte Mustererkennung zur Charakterisierung von Netzrückwirkungen beim Laden von Elektrofahrzeugen

Dieses Repository enthält den offiziellen Quellcode für die Forschungsarbeit **"Entwicklung einer KI-gestützten Mustererkennung zur Charakterisierung von Netzrückwirkungen beim Laden von Elektrofahrzeugen"**. 

Das System implementiert eine vollständige Pipeline zur Erkennung, Klassifizierung und Interpretierbarkeit von 19 verschiedenen Klassen von Netzrückwirkungen (Power Quality Disturbances, PQDs). Die Pipeline reicht von der mathematisch-synthetischen Signalgenerierung über die Signalverarbeitung auf Basis der generalisierten S-Transformation und das CNN-Training inklusive Hyperparameter-Optimierung bis hin zur modellinternen Visualisierung mittels Grad-CAM.

---

## 📊 Pipeline-Übersicht

```mermaid
graph TD
    %% Nodes
    A0[Synthetische Signalgenerierung<br/>PQDModell.m]
    A[Rohe 1D MATLAB<br/>'pq_daten_19_klassen_sauber.mat'] 
    B[Fast Generalized S-Transform<br/>& AWGN Rauscherweiterung]
    C[Generierte 2D-Datensätze<br/>'.npz'-Dateien / 120x120]
    D[Hyperparameter-Optimierung<br/>Bayesian Sweep]
    E[Modell-Training<br/>NetzstoerungsCNN]
    F[Export<br/>Druckgrafiken / Grad-CAM Ordner]

    %% Pipeline Flow
    A0 -->|Matlab Simulation| A
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
    class A0,B,D,E process;
    class F output;

```

---

## 🛠️ Beschreibung der Skripte

### 1. `PQDModell.m` — Mathematische Signalgenerierung (MATLAB)

Dieses Skript bildet die fundamentale Datenbasis des Projekts. Es implementiert ein integrales mathematisches Modell (basierend auf *Igual et al., 2017*) zur stochastischen Erzeugung von 19 diskreten Netzstörungsklassen.

```mermaid
graph TD
    A[Start: Funktionsaufruf mit ns, fs, f, n, A] --> B[Grenzwertelektierung & Parameter-Validierung]
    B --> C[Definition der stochastischen Zufallsbereiche<br/>Phasen, Amplituden, Zeitfenster]
    C --> D[Initialisierung der 3D-Ausgangsmatrix<br/>Shape: ns x Punkte x 29]
    
    subgraph Iterative Generierung pro Klasse 1-19
        D --> E[Klasse 1-4: Reine Sinuswelle, Sags, Swells, Interruptions]
        D --> F[Klasse 5-6: Impulsive & Oszillierende Transienten]
        D --> G[Klasse 7-9: Harmonics & Kombinationen]
        D --> H[Klasse 10-12: Flicker & Kombinationen]
        D --> I[Klasse 13-16: Komplexe transiente/harmonische Überlagerungen]
        D --> J[Klasse 17-19: Spannungseinbuchtungen / Notches & Multistörungen]
    end

    E & F & G & H & I & J --> K[Zuweisung an Speicher-Matrix]
    K --> L[Ende: Export der Rohsignale<br/>'pq_daten_19_klassen_sauber.mat']

    style A fill:#f7fafc,stroke:#4a5568,stroke-width:2px
    style L fill:#ebf8ff,stroke:#3182ce,stroke-width:2px

```

* **Zentrale Funktionen:**
* **Stochastische Modellierung:** Parameter wie Einbruchstiefe ($\alpha$), Überhöhungsfaktor ($\beta$), Transientendämpfung ($\tau$) oder Notch-Anzahl ($c$) werden innerhalb physikalisch realistischer Grenzen für jedes Signal zufällig gewürfelt.
* **Umfassender Klassenspiegel:** Erzeugt transiente Phänomene, periodische Störungen (Harmonische), Amplitudenschwankungen (Flicker, Sags, Swells) sowie komplexe, in der Praxis auftretende Kombinationsstörungen (z. B. *Harmonics+Swell+Flicker*).
* **Parametrierbarkeit:** Volle Kontrolle über Signalanzahl ($ns$), Abtastrate ($fs$), Netzfrequenz ($f$) und Zyklenanzahl ($n$).
* **Lizenz- & Änderungshinweis:** Modifizierte und übersetzte Version des Originalmodells von 2017, lizenziert unter *Creative Commons Attribution 4.0 International (CC BY 4.0)*.



---

### 2. `STFINAL.py` — Signalverarbeitung & Datensatzgenerierung

Dieses Python-Skript transformiert die stochastisch generierten 1D-Zeitsignale aus MATLAB (`.mat`) mittels fortschrittlicher Zeit-Frequenz-Analyse in hochauflösende 2D-Spektrogramme.

```mermaid
graph TD
    A[Start: pq_daten_19_klassen_sauber.mat] --> B[Datenimport & Reshaping in 1D-Arrays]
    B --> C[Stratified Train/Test Split 80/20]
    C --> D[Loop über Rauschszenarien<br/>clean, mixed, 50dB, 20dB]
    D --> E[Parallelisierte Verarbeitung via joblib]
    
    subgraph Signal-Transformations-Pipeline pro Sample
        E --> F[Optional: AWGN-Rauschen addieren]
        F --> G[1D Fourier-Transformation FFT]
        G --> H[Frequenzverschiebung & Skaliertes Gauß-Fenster]
        H --> I[Inverse FT IFT]
        I --> J[Amplitudenspektrum + Logarithmische Kompression]
        J --> K[Robustes Perzentil-Scaling 1% bis 99%]
        K --> L[Skalierung auf Zielgröße 120x120]
    end

    L --> M[CNN-Kanal-Dimension hinzufügen<br/>Shape: N, 1, 120, 120]
    M --> N[Ende: Komprimierter Export als .npz Datei]

    style A fill:#ebf8ff,stroke:#3182ce,stroke-width:2px
    style N fill:#f0fff4,stroke:#38a169,stroke-width:2px

```

* **Zentrale Funktionen:**
* **Generalized S-Transform (GST):** Berechnet die S-Transformation mit frequenzabhängig skalierter Fensterbreite zur Maximierung der Zeit-Frequenz-Auflösung.
* **Logarithmische Kompression & Perzentil-Skalierung:** Reduziert die Signaldynamik und eliminiert statistische Ausreißer, um Gradienteninstabilitäten beim CNN-Training vorzubeugen.
* **Daten-Augmentation (AWGN):** Beaufschlagt Signale mit additivem weißem gaußschem Rauschen für unterschiedliche Testszenarien (Clean, Mixed 20–50 dB, Feste Pegel bei 20 dB und 50 dB).
* **Parallelisierung:** Nutzt `joblib` für hocheffiziente Multi-Core-Verarbeitung der rechenintensiven Integraltransformationen.



---

### 3. `CNN-Modell.py` — Core Training Pipeline

Dieses Skript definiert die neuronale Netzwerkarchitektur und steuert den primären Trainingsprozess über 40 Epochen auf den rauschbeaufschlagten Spektrogrammdaten.

```mermaid
graph TD
    A[Start: mixed Trainings- & Testdaten .npz] --> B[DataLoader initialisieren<br/>Batch Size: 16]
    B --> C[Modellarchitektur instanziieren]
    
    subgraph NetzstoerungsCNN Architektur
        C --> C1[3x Conv2D + MaxPool2D Blöcke<br/>Feature Extraktion]
        C1 --> C2[Spatial Dropout 2D<br/>p=0.2 gegen Overfitting]
        C2 --> C3[Flatten & FC1 Schicht<br/>128 Neuronen + ReLU]
        C3 --> C4[Dense Dropout<br/>p=0.5 Regularisierung]
        C4 --> C5[FC2 Ausgangsschicht<br/>19 diskrete Klassen]
    end

    C5 --> D[Trainings-Schleife 40 Epochen]
    
    subgraph Epochen-Verarbeitung
        D --> D1[Forward Pass & Cross-Entropy Loss]
        D1 --> D2[Backward Pass & Adam-Optimizer]
        D2 --> D3[Modell-Evaluierung auf Testdaten]
        D3 --> D4[Live-Metriken & Konfusionsmatrix an WandB]
    end

    D4 --> E[Modellgewichte lokal sichern<br/>.pth Datei]
    E --> F[Post-hoc Grad-CAM Analyse der conv3 Schicht]
    F --> G[Ende: Gewichte als WandB-Artefakt hochladen]

    style A fill:#ebf8ff,stroke:#3182ce,stroke-width:2px
    style G fill:#f0fff4,stroke:#38a169,stroke-width:2px

```

* **Modell-Architektur (`NetzstoerungsCNN`):**
* Drei aufeinanderfolgende Faltungsblöcke (Conv2D $\rightarrow$ ReLU $\rightarrow$ MaxPool2D) extrahieren raum-zeitliche Muster aus den Spektrogrammen.
* **Spatial Dropout (`Dropout2d`, p=0.2):** Nullt ganze Feature-Maps aus, um Co-Adaptionen der Faltungskerne zu verhindern.
* **Dense Dropout (p=0.5):** Rigide Regularisierung der Klassifikationsschicht gegen strukturelles Auswendiglernen.


* **Zentrale Funktionen:**
* Trainiert das Modell mit dem Adam-Optimizer ($lr=0.0005$) und Cross-Entropy-Verlustfunktion.
* Live-Metriken-Tracking (Loss, Accuracy) und automatische Generierung interaktiver Konfusionsmatrizen pro Epoche in **Weights & Biases (WandB)**.



---

### 4. `wandB.py` — Hyperparameter-Sweep (Optimierung)

Zur systematischen Maximierung der Modellperformance implementiert dieses Skript eine automatisierte, bayesianische Hyperparameter-Suche.

```mermaid
graph TD
    A[Start: Globale Datensätze laden] --> B[Sweep-Konfiguration definieren<br/>Methode: Bayes / Ziel: max val_acc]
    B --> C[Suchraum festlegen<br/>Learning Rate, Batch Size, FC-Dropout]
    C --> D[Sweep auf WandB-Server initialisieren]
    D --> E[WandB-Agent starten<br/>Limit: 10 Runs]
    
    subgraph train_sweep Funktion pro Run
        E --> F[Run-spezifische Hyperparameter anfordern]
        F --> G[Dynamischer DataLoader & Modell-Setup]
        G --> H[Kompaktes Training über 10 Epochen]
        H --> I[Live-Logging der Validierungsmetriken]
    end

    I --> J[Run-Abschluss & Post-Auswertung]
    J --> K[Lokaler Gewichte-Export + Classification Report]
    K --> L[Grad-CAM Tabelle generieren & an WandB senden]
    L --> M[Ende: Run-Gewichte als Artefakt hochladen]

    style A fill:#ebf8ff,stroke:#3182ce,stroke-width:2px
    style M fill:#f0fff4,stroke:#38a169,stroke-width:2px

```

* **Zentrale Funktionen:**
* **Bayesianische Optimierung:** Durchsucht den vordefinierten Suchraum intelligent auf Basis vorheriger Läufe, um die Validierungsgenauigkeit (`val_acc`) effizient zu maximieren.
* **Dynamische Konfiguration:** Steuert Variablen wie Lernrate ($0.001, 0.0005, 0.0001$), Batch-Size ($16, 32, 64$) und den FC-Dropout-Faktor ($0.3, 0.4, 0.5, 0.6$) direkt über den cloudbasierten WandB-Agenten.



---

### 5. `Auswertung.py` — Standalone Inferenz & Druckgrafik-Export

Dieses Skript dient der finalen Validierung trainierter Modelle auf unabhängigen Test-Szenarien und erzeugt hochauflösende Visualisierungen für die schriftliche Ausarbeitung.

```mermaid
graph TD
    A[Start: Testdaten .npz & Gewichte .pth laden] --> B[Modell in eval-Modus versetzen<br/>Dropout deaktivieren]
    B --> C[Inferenz-Schleife über kompletten Test-Loader]
    C --> D[Softmax-Wahrscheinlichkeiten & Hard-Predictions extrahieren]
    D --> E[Lokale Sicherung: rohe_vorhersagen.npz]
    E --> F[Berechnung des classification_report & Matrix]
    
    subgraph Cloud Logging & Lokaler Grafik-Export
        F --> G[WandB-Upload: Präzision, Recall, F1, ROC-Kurve]
        F --> H[Matplotlib-Export: Konfusionsmatrix als High-Res Vektor-PDF]
    end

    H --> I[Grad-CAM Erklärbarkeits-Analyse]
    
    subgraph Visualisierungs-Optimierung pro Schlüsselindex
        I --> J[Aktivierungs-Heatmap der conv3 Schicht berechnen]
        J --> K[Gamma-Korrektur des S-Transform Bildes]
        K --> L[RGB-Overlay der Heatmap auf Originalbild]
        L --> M[Speichern als druckfähiges PNG & WandB-Tabelle]
    end

    M --> N[Ende: Auswertung vollständig abgeschlossen]

    style A fill:#ebf8ff,stroke:#3182ce,stroke-width:2px
    style N fill:#f0fff4,stroke:#38a169,stroke-width:2px

```

* **Zentrale Funktionen:**
* **High-Res Vektor-Export:** Generiert eine druckreife, hochauflösende PDF der Konfusionsmatrix (`Konfusionsmatrix_Druckqualitaet.pdf`) im passenden wissenschaftlichen Farbschema (`Blues`).
* **Optimiertes Grad-CAM Visualisierungs-System:** Berechnet Aktivierungs-Heatmaps der letzten Faltungsschicht (`conv3`) für dedizierte Schlüsselindizes. Wendet eine **Gamma-Korrektur** ($\gamma = 0.5$) auf das S-Transformationsbild an, um feine, schwache Frequenzkomponenten für das menschliche Auge im finalen PNG-Overlay sichtbar zu machen.



---

## 📋 Voraussetzungen & Installation

Das Projekt basiert auf **PyTorch** und nutzt **CUDA** zur Hardwarebeschleunigung auf NVIDIA-GPUs (getestet mit PyTorch 2.6.0 und CUDA 12.4).

### Abhängigkeiten installieren:

```bash
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
pip install numpy scipy scikit-image scikit-learn matplotlib joblib h5py wandb pytorch-grad-cam

```

```

```
