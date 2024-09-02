# Importiamo le librerie necessarie
from tensorflow.keras.models import load_model  # Per caricare un modello salvato
from tensorflow.keras.preprocessing import image  # Per la gestione delle immagini
import numpy as np  # Per le operazioni numeriche

# 1. Carica il modello salvato
# Utilizziamo la funzione load_model per caricare il modello precedentemente salvato nel file 'model.h5'.
# Questo file contiene la struttura del modello, i pesi addestrati, e le configurazioni.
model = load_model('model.h5')  # Sostituisci con il percorso corretto del file se necessario

# 2. Prepara l'immagine da classificare
# Specifica il percorso dell'immagine che vuoi classificare.
img_path = 'path/to/your/test_image.png'  # Cambia questo con il percorso effettivo della tua immagine

# Carica l'immagine dal disco usando image.load_img() specificando la dimensione target (128x128).
# Questo ridimensionamento è necessario per garantire che l'immagine abbia le stesse dimensioni usate durante l'addestramento.
img = image.load_img(img_path, target_size=(128, 128))

# Converte l'immagine caricata in un array numpy, che è il formato richiesto dalla rete neurale.
# L'array risultante avrà dimensioni (128, 128, 3), dove 3 indica i canali di colore RGB.
img_array = image.img_to_array(img)

# Aggiungi una dimensione extra all'array per rappresentare il batch.
# Le reti neurali in Keras si aspettano input di forma (batch_size, height, width, channels).
# Qui aggiungiamo una dimensione per rappresentare un batch singolo (batch_size=1).
img_array = np.expand_dims(img_array, axis=0)  # Ora l'array avrà la forma (1, 128, 128, 3)

# Normalizza i valori dei pixel dell'immagine.
# Dividiamo i valori dei pixel (originariamente tra 0 e 255) per 255.0 per ottenere un range tra 0 e 1.
# Questa normalizzazione è necessaria perché il modello è stato addestrato con immagini normalizzate, migliorando l'efficacia dell'addestramento.
img_array = img_array / 255.0

# 3. Effettua la predizione
# Usa il modello caricato per fare una predizione sull'immagine preparata.
# La funzione predict restituisce un array di probabilità per ciascuna classe.
# Ad esempio, se hai 4 classi, l'output potrebbe essere qualcosa come [0.1, 0.3, 0.5, 0.1].
prediction = model.predict(img_array)

# Mostra l'output della predizione.
# Questo è un array che contiene le probabilità assegnate a ciascuna classe dall’immagine input.
print("Predizione (probabilità per ciascuna classe):", prediction)

# 4. Identifica la classe con la probabilità più alta
# np.argmax() restituisce l'indice dell'elemento con la probabilità più alta nell'array di predizioni.
# Questo indice rappresenta la classe predetta dal modello.
predicted_class = np.argmax(prediction, axis=1)  # axis=1 indica che cerchiamo lungo la dimensione delle classi
print("Classe Predetta:", predicted_class)

# Nota: il valore di `predicted_class` sarà un indice numerico. Per ottenere il nome effettivo della classe,
# è necessario mappare questi indici con i nomi delle classi (es. 0 -> 'alpha', 1 -> 'beta', ecc.),
# utilizzando il dizionario di classi creato durante il training.
