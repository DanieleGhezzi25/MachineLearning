# Importazione delle librerie necessarie
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Definizione dei percorsi delle cartelle che contengono le immagini di training e validazione
train_dir = 'path/to/train/'  # Cartella principale che contiene le immagini di training organizzate per classi
val_dir = 'path/to/validation/'  # Cartella principale che contiene le immagini di validazione organizzate per classi

# Creazione di un generatore di immagini per il training con data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,            # Normalizzazione dei pixel tra 0 e 1 (originalmente tra 0 e 255)
    rotation_range=20,            # Rotazione casuale delle immagini fino a 20 gradi
    width_shift_range=0.2,        # Traslazione orizzontale casuale delle immagini fino al 20% della larghezza
    height_shift_range=0.2,       # Traslazione verticale casuale delle immagini fino al 20% dell'altezza
    shear_range=0.2,              # Applicazione di una distorsione angolare (shear) fino al 20%
    zoom_range=0.2,               # Zoom casuale delle immagini fino al 20%
    horizontal_flip=True,         # Ribaltamento orizzontale casuale delle immagini
    fill_mode='nearest'           # Riempimento dei pixel mancanti dopo le trasformazioni usando i pixel più vicini
)

# Creazione di un generatore di immagini per la validazione (senza data augmentation)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)  # Solo normalizzazione delle immagini

# Creazione del generatore per le immagini di training
train_generator = train_datagen.flow_from_directory(
    train_dir,                    # Percorso della cartella contenente le immagini di training
    target_size=(128, 128),       # Dimensione a cui ridimensionare tutte le immagini
    batch_size=32,                # Numero di immagini da processare in un batch
    class_mode='categorical'      # Tipo di etichettatura; 'categorical' per problemi multiclasse (one-hot encoding)
)

# Creazione del generatore per le immagini di validazione
val_generator = val_datagen.flow_from_directory(
    val_dir,                      # Percorso della cartella contenente le immagini di validazione
    target_size=(128, 128),       # Dimensione a cui ridimensionare tutte le immagini
    batch_size=32,                # Numero di immagini da processare in un batch
    class_mode='categorical'      # Tipo di etichettatura; 'categorical' per problemi multiclasse
)

# Creazione del modello della rete neurale convoluzionale (CNN)
model = Sequential([
    # Primo livello convoluzionale con 32 filtri, kernel 3x3, e funzione di attivazione ReLU
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),  # Livello di pooling per ridurre la dimensionalità delle feature map
    # Secondo livello convoluzionale con 64 filtri
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),  # Pooling per ridurre ulteriormente la dimensionalità
    # Terzo livello convoluzionale con 128 filtri
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),  # Ancora pooling per ridurre la dimensionalità
    Flatten(),                       # Appiattimento delle feature map in un vettore 1D
    Dense(128, activation='relu'),   # Livello denso completamente connesso con 128 neuroni
    Dropout(0.5),                    # Dropout per ridurre l'overfitting disattivando casualmente il 50% dei neuroni
    Dense(4, activation='softmax')   # Livello d'uscita con 4 neuroni (uno per classe) e softmax per la classificazione multiclasse
])

# Compilazione del modello con funzione di perdita, ottimizzatore e metriche di valutazione
model.compile(
    loss='categorical_crossentropy',  # Funzione di perdita per classificazione multiclasse
    optimizer='adam',                 # Ottimizzatore Adam per aggiornare i pesi
    metrics=['accuracy']              # Metrica di valutazione: accuratezza
)

# Addestramento del modello utilizzando i generatori di training e validazione
history = model.fit(
    train_generator,                  # Generatore per i dati di training
    steps_per_epoch=len(train_generator),  # Numero di batch per epoca
    epochs=20,                        # Numero di epoche per l'addestramento
    validation_data=val_generator,    # Generatore per i dati di validazione
    validation_steps=len(val_generator)  # Numero di batch per validazione
)

# Salvataggio del modello addestrato su disco
model.save('model.h5')  # Salva il modello in un file .h5 per futuri utilizzi o ulteriori addestramenti
