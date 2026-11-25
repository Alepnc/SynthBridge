H-TDBU Forest — note

Riferimento: Nie, Jin, Chen, Hierarchical Synthetic Tabular Data Generation: A Hybrid Top-Down and Bottom-Up Framework, ICML 2026, arXiv:2605.28198.
Codice originale: https://github.com/junfengn-ctrl/hierarchical-synthetic-tabular (licenza MIT).
Autori dell'integrazione: Alessandro Panico, Cataldo Parente

Cosa è stato integrato

Il generatore "RandomForest conditional synthesis" del repository: sintesi sequenziale colonna per colonna, ogni colonna predetta da una Random Forest sulle colonne già generate. Non è stata integrata la parte "top-down" del paper (regole costruite a mano o con un LLM) né la componente testuale cross-modale.

Scelte rispetto al codice originale
Lavora su DataFrame invece che su file CSV.
Campionamento delle classi vettorizzato (molto più veloce con tante righe).
Interi con pochi valori distinti (soglia 20, come nell'originale) trattati come categorici, ad esempio un target 0/1.
Valori mancanti: i categorici diventano una categoria speciale e sono riprodotti come NA. Nelle numeriche i mancanti sono imputati con la mediana solo come input delle altre colonne; non entrano nell'addestramento della colonna a cui appartengono (altrimenti il generatore imparerebbe un picco sulla mediana) e la loro percentuale è riprodotta a caso nei dati sintetici (parametro reproduce_missing). Il meccanismo reale dei mancanti non è riprodotto.
Parametri aggiunti: profondità, foglia minima, soglia degli interi, ordine delle colonne, riproduzione dei mancanti.
Limiti noti
Il risultato dipende dall'ordine delle colonne (la prima è solo ricampionata).
I residui numerici sono calcolati sui dati di addestramento (in-sample), quindi il rumore può essere sottostimato.
I valori numerici non sono limitati all'intervallo osservato.
Con molte colonne di condizionamento e variabili categoriche ad alta cardinalità la memoria può crescere.
Test eseguiti
Rscript tools/run_algorithm.R h_tdbu iris 200
mtcars (32 x 11): 1.4 s. Scoperto che R passa cyl, vs, am, gear, carb come decimali: ora le colonne con soli valori interi e pochi valori distinti sono trattate come categoriche (prima uscivano valori come vs = -0.13).
airquality (153 x 6, con NA): 1.6 s. Scoperto che l'imputazione con la mediana creava un picco nei dati sintetici (25.6% delle righe sulla mediana in un test con 26% di mancanti, contro 1.8% reale): corretto, ora 1.5%.
ADD: iris, tempi, check_algorithms.R