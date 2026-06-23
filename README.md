#Drug Safety signal detection from adverse event 

```mermaid

graph LR
    %% CONFIGURAZIONE STILI
    classDef data fill:none,stroke:#00d2ff,color:#fff,stroke-width:2px;
    classDef standard fill:none,stroke:#ff007f,color:#fff,stroke-width:2px;
    classDef stats fill:none,stroke:#9b51e0,color:#fff,stroke-width:2px;
    classDef out fill:none,stroke:#00ff88,color:#fff,stroke-width:2px;
    classDef matrix fill:none,stroke:#fff,color:#fff,stroke-dasharray:5 5;

    %% BLOCCHI PRINCIPALI
    subgraph Data_Ingestion ["Data Ingestion & Extraction"]
        A[(FAERS / VigiBase / RWD)]:::data -->|Raw ICSRs| B(Data Cleaning & Filtering):::data
    end

    subgraph Preprocessing ["Harmonization"]
        C(Drug Vocabulary Mapping):::standard --> D(Adverse Event Mapping):::standard
    end

    subgraph Statistical_Core ["Disproportionality Analysis"]
        E(Frequentist Metrics):::stats
        F(Bayesian Metrics):::stats
    end

    subgraph Signal_Prioritization ["Evaluation & Output"]
        G(FDR Control & Thresholding):::out --> H[Signal Dashboard]:::out
    end

    %% FLUSSO E CONNESSIONI
    Data_Ingestion --> Preprocessing
    
    D --> Matrix_Label[2x2 Contingency Matrix]:::matrix
    D --> Bayesian_Label[Bayesian Shrinkage]:::matrix
    
    Matrix_Label --> E
    Bayesian_Label --> F
    
    E --> G
    F --> G

    %% COLORI DEI CONTENITORI
    style Data_Ingestion fill:none,stroke:#00d2ff,stroke-width:1px,color:#fff
    style Preprocessing fill:none,stroke:#ff007f,stroke-width:1px,color:#fff
    style Statistical_Core fill:none,stroke:#9b51e0,stroke-width:1px,color:#fff
    style Signal_Prioritization fill:none,stroke:#00ff88,stroke-width:1px,color:#fff
