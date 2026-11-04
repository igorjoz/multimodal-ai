# Converted from triplet loss.ipynb


# %% [markdown]
# ##### AI TECH - Akademia Innowacyjnych Zastosowań Technologii Cyfrowych. Programu Operacyjnego Polska Cyfrowa na lata 2014-2020

# %% [markdown]
# AI-TECH
# DO LABORATORIUM 3
# z Przetwarzania multimediów w systemach decyzyjnych
# Przegląd algorytmów uczących się metryk dystansu dla
# danych wizyjnych
# Adam Kurowski
# algorytm straty trójkowej (triplet loss)

# %% [markdown]
# ![stopka.svg](stopka.svg)

# %% [markdown]
# Projekt współfinansowany ze środków Unii Europejskiej w ramach Europejskiego Funduszu Rozwoju Regionalnego
# Program Operacyjny Polska Cyfrowa na lata 2014-2020,
# Oś Priorytetowa nr 3 "Cyfrowe kompetencje społeczeństwa" Działanie  nr 3.2 "Innowacyjne rozwiązania na rzecz aktywizacji cyfrowej"
# Tytuł projektu:  „Akademia Innowacyjnych Zastosowań Technologii Cyfrowych (AI Tech)”

# %% [markdown]
# # Trening sieci algorytmem straty trójkowej

# %% [markdown]
# Algorytm straty trójkowej jest mniej "zachłanny" jeśli chodzi o szybkie umieszczanie punktów w przestrzeni decyzyjnej. Zmniejsza to ryzyko słabej generalizacji modelu, jednak okupione jest wolniejszym treningiem.

# %%
# Import bibliotek
import numpy as np
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers     import Input, Dense, Conv2D, Flatten, PReLU, MaxPooling2D, Lambda
from tensorflow.keras.models     import Model
from tensorflow.keras.utils      import to_categorical
from tensorflow.keras            import backend as K
from numpy.linalg                import norm
from sklearn.decomposition       import PCA
from tqdm                        import tqdm

# %% [markdown]
# # Nastawy pracy algorytmu:

# %%
# Nastawy algorytmu treningu
learning_rate       = 0.001
batches_per_epoch   = 10
batch_size          = 16
num_training_epochs = 10

# Nastawy algorytmu uczącego się metryk odległości
embedding_length  = 20 # długość wektora reprezentacji
margin            = 1  # ograniczenie na docelową odległość między klastrami punktów
distance_norm     = 'euclidean' # metryka odległości Ln, może być też 1,2, czy np.inf

# Nazwy folderów na dane:
script_dir = Path(__file__).resolve().parent
folder_model_weights    = script_dir / "_weights"
folder_repspace_preview = script_dir / "_previews"
dataset_cache          = script_dir.parent / "_datasets"

# %% [markdown]
# Odczyt danych z reporytorium TensorFlow Datasets

# %%
# Wykorzystamy zbiór MNIST.
trn_dst, test_dst = tfds.load('mnist', data_dir=str(dataset_cache), split=['train','test'], shuffle_files=True, as_supervised = True, batch_size=-1)

# Skrypt napisany jest z myślą, że zbiór danych podzielony jest
# na standardowe cztery tablice (X - wzorce wejściowe, Y - etykiety).
trn_X,   trn_Y  = tfds.as_numpy(trn_dst)
test_X,  test_Y = tfds.as_numpy(test_dst)

# Konwertujemy dane wejściowe na format zmiennoprzecinkowy:
trn_X  = trn_X.astype(np.float16)
test_X = test_X.astype(np.float16)

# Pobieramy informację o rozmiarze wejścia.
# (dla MNIST jest to (28, 28, 1))
input_dims = trn_X[0].shape

# Normalizacja danych:

def normalize_dataset(dataset):
    for i in range(len(dataset)):
        dataset[i] = dataset[i]/np.max(np.abs(dataset[i]))
    return dataset

trn_X  = normalize_dataset(trn_X)
test_X = normalize_dataset(test_X)

# %% [markdown]
# # Definicja pojedynczej gałęzi sieci.
# Teoretycznie w kontekście algorytmów uczących się metryk dystansu mówi się o sieci syjamskiej, jednak definiowanie jej struktury wprost w kodzie jest nadmiarowe.
# W praktyce definiuje się strukturę tylko jednej z odnóg sieci - druga jest identyczna, więc ta pojedyncza sieć jest w procesie treningu uruchamiana dwa razy.
# Moduł obliczający odległość na podstawie dwóch reprezentacji zrealizowany jest w funkcji straty zdefiniowanej w dalszej części skryptu

# %%
network_input = Input(shape=input_dims, name="left_input")
x = Conv2D(8,(2,2), padding='same')(network_input)
x = PReLU()(x)
x = Conv2D(8,(2,2), padding='same')(x)
x = PReLU()(x)
x = Flatten()(x)
x = Dense(128)(x)
embedding_vector = Dense(embedding_length, activation=None)(x)
encoder_model    = Model(inputs=network_input, outputs=embedding_vector)

print("Struktura kodera:")
encoder_model.summary()
print()

# %% [markdown]
# # Definicje funkcji składowych algorytmu triplet loss i trening

# %%
# Funkcja straty trójkowej (ang. triplet loss). Struktura samej funkcji straty jest dość prosta, a jej
# główna idea zawarta jest w linijce obliczającej wartość zmiennej partial_losses.
def triplet_loss_function(anchor_repr, positive_repr, negative_repr, distance_norm='euclidean', margin=1, return_partials = False):
    distance_ap    = tf.norm(anchor_repr - positive_repr, ord=distance_norm, axis=1)
    distance_an    = tf.norm(anchor_repr - negative_repr, ord=distance_norm, axis=1)
    partial_losses = tf.math.maximum(0, distance_ap - distance_an + margin)
    loss_value     = tf.math.reduce_mean(partial_losses)
    
    if return_partials:
        return partial_losses
    else:
        return loss_value

# Poważnym problemem do rozwiązania w trakcie treningu sieci jest pozyskiwanie trójek reprezentacji,
# które nie generują wartości funkcji straty równej zeru. Dlatego też generowanie trójek przykładów
# (ang. triplet mining) ma dość złożoną postać w postaci dwóch funkcji poniżej:

# Generowanie pojedynczej trójki wzorców do treningu:
def mine_triplet(trn_X, trn_Y):
    
    # Na początku sprawdzamy, jakie klasy mamy do dyspozycji i wybieramy klasę kotwicy (ang. anchor):
    classes_available = np.unique(trn_Y)
    anchor_class      = np.random.choice(classes_available)
    
    # Dla wygody podzielimy sobie zbiór na dane z klasy kotwicy i spoza tej klasy:
    anchor_class_examples    = trn_X[trn_Y==anchor_class]
    nonanchor_class_examples = trn_X[trn_Y!=anchor_class]
    
    # Następnie wybieramy indeks konkretnego przykładu kotwicy i przykładu pozytywnego
    # czyli takiego, który należy do tej samej klasy co kotwica:
    anchor_pattern_idx    = np.random.choice(len(anchor_class_examples),1)
    positive_pattern_idx  = np.random.choice(len(anchor_class_examples),1)
    
    # Jeśli wylosowaliśmy takie same indeksy, to losujemy jeszcze raz, aż uzyskamy różne indeksy:
    while positive_pattern_idx == anchor_pattern_idx:
        positive_pattern_idx  = np.random.choice(len(anchor_class_examples),1)
    
    # Na koniec losujemy przykład negatywny (spoza klasy kotwicy):
    negative_pattern_idx  = np.random.choice(len(nonanchor_class_examples),1)
    
    # Przygotowujemy dane do zwrócenia z funkcji, pobieramy wzorce spod wybranych indeksów
    # i przekazujemy je do dalszej części skryptu.
    anchor_pattern   = anchor_class_examples[anchor_pattern_idx]
    positive_pattern = anchor_class_examples[positive_pattern_idx]
    negative_pattern = nonanchor_class_examples[negative_pattern_idx]
    
    return (anchor_pattern, positive_pattern, negative_pattern)

# Proces treningu przeprowadzany jest w podgrupach danych (ang. batches), stąd nie wystarczy tylko generować trójek danych,
# konieczne jest także określenie sposobu generowania całych podgrup:
def mine_batch_of_triplets(encoder_model, trn_X, trn_Y, batch_size, input_dims, distance_norm='euclidean', margin=1, eps=0.001):
    
    # Na początku określamy jaki kształt ma macierz przechowująca grupę danych.
    # Rezerwujemy też miejsce na macierze przechowujące poszczególne składowe trójek
    # reprezentacji.
    batch_matrix_shape = tuple([batch_size] + list(input_dims))
    anchors_mtx        = np.zeros(batch_matrix_shape)
    positive_mtx       = np.zeros(batch_matrix_shape)
    negative_mtx       = np.zeros(batch_matrix_shape)
    
    # Główną trudnością w generowaniu trójek przykładów jest wybranie takich trójek, które
    # charakteryzują się wartością straty większą od 0 (bo jeśli wartość ta jest równa 0, to 
    # przykład taki jest bezużyteczny w treningu - nie generuje poprawek dla wag sieci).
    #
    # Dlatego w pierwszym przejściu poniższej pętli generujemy wstępne propozycje trójek w grupie danych,
    # a następnie jeszcze raz generujemy trójki, dla których strata trójkowa wynosi 0:
    first_pass = True
    while True:
        # Policzmy reprezentacje dla wybranych trójek przykładów:
        representations_anchor   = encoder_model(anchors_mtx)
        representations_positive = encoder_model(positive_mtx)
        representations_negative = encoder_model(negative_mtx)
        
        # Teraz obliczmy funkcję straty dla wybranego zestawu trójek wzorców:
        partial_losses = triplet_loss_function(representations_anchor, representations_positive, representations_negative, distance_norm, margin, return_partials = True)
        
        # Zobaczmy, gdzie wartość straty jest bliska zeru (mniejsza od eps):
        zero_loss_triplet_mask = partial_losses.numpy() <= eps
        
        # Jeśli którakolwiek wartość straty jest równa zeru to...
        if zero_loss_triplet_mask.any() or first_pass:
            for i in range(batch_size):
                # ... losujemy nową trójkę w miejsce tej, która dała zerową stratę i jeszcze raz 
                # sprawdzamy, czy w naszej grupie danych wszystkie trójki generują niezerową
                # wartość funkcji straty.
                if (zero_loss_triplet_mask[i] == False) and (not first_pass): continue
                anchor_pattern, positive_pattern, negative_pattern = mine_triplet(trn_X, trn_Y)
                anchors_mtx[i,::]  = anchor_pattern
                positive_mtx[i,::] = positive_pattern
                negative_mtx[i,::] = negative_pattern
        else:
            break
        
        first_pass = False
    
    loss_value = tf.math.reduce_mean(partial_losses)
    return anchors_mtx, positive_mtx, negative_mtx, representations_anchor, representations_positive, representations_negative, loss_value

# Przygotowanie środowiska pracy:
if not os.path.isdir(folder_repspace_preview): os.mkdir(folder_repspace_preview)
if not os.path.isdir(folder_model_weights):    os.mkdir(folder_model_weights)

# Opcjonalnie - można załadować wagi modelu:
if os.path.isfile(os.path.join(folder_model_weights, 'model.weights.h5')):
    encoder_model.load_weights(os.path.join(folder_model_weights, 'model.weights.h5'))

# Na koniec rozpoczynamy właściwy trening sieci:
optimizer     = tf.keras.optimizers.Adam(learning_rate)

print("trening sieci neuronowej algorytmem triplet loss został rozpoczęty")
print()

# trenujemy, aż nie otrzymamy sygnału wstrzymania z klawiatury (ctrl+c):
for epoch_number in range(1, num_training_epochs+1):

    # --------------------------------------------------------
    # właściwy algorytm treningu sieci:

    print(f"epoka: {epoch_number}")

    # Korzystamy z własnego zdefiniowania algorytmu treningu, stąd musimy posłuzyć się kontekstem GradientTape:
    with tf.GradientTape() as tape:
        accumulated_loss = 0

        # Dla każdej grupy danych:
        for batch_idx in tqdm(range(batches_per_epoch)):
            # Wygeneruj trójki przykładów, które generują niezerową wartość funkcji straty:
            anchors_mtx, positive_mtx, negative_mtx, representations_anchor, representations_positive, representations_negative, loss_value = mine_batch_of_triplets(encoder_model, trn_X, trn_Y, batch_size, input_dims, distance_norm, margin)

            # Dodajemy stratę grupy danych do zmiennej, która na koniec wykonania wewnętrznej pętli będzie zawierać
            # średnią wartość straty dla całej epoki:
            accumulated_loss += loss_value*tf.constant(1./batches_per_epoch)

    # Wylicz gradienty na podstawie funkcji straty:
    grads = tape.gradient(accumulated_loss, encoder_model.trainable_weights)

    # Zaaplikuj poprawki do wag sieci za pomocą algorytmu wstecznej propagacji błędów
    optimizer.apply_gradients(zip(grads, encoder_model.trainable_weights))

    # --------------------------------------------------------
    # wizualizacja, zapis wag i raportowanie postępów
    info_text = f"wartość straty trójkowej (triplet loss): {accumulated_loss}"
    print(info_text)
    print()

    # Zapis wag do późniejszego użycia:
    encoder_model.save_weights(os.path.join(folder_model_weights,'model.weights.h5'))

    # Wizualizacja przestrzeni reprezentacji za pomocą PCA, jest to konieczne,
    # bo wizualizacja ma tylko 2 wymiary, a reprezentacje typowo mają wymiarów
    # znacznie więcej.
    test_embeddings = encoder_model.predict(test_X, batch_size=batch_size, verbose=0)
    pca             = PCA(n_components=2, svd_solver='full')
    reduced         = pca.fit_transform(test_embeddings)

    # Rysujemy obliczoną wcześniej wizualizację i zapisujemy na dysk:
    plt.figure(figsize=(16,9))
    for class_name in np.unique(test_Y):
        mask = test_Y == class_name
        plt.scatter(reduced[mask,0],reduced[mask,1], label=class_name, s=10)
    plt.legend()
    plt.title(info_text)
    plt.grid()

    epoch_num_str = str(epoch_number).zfill(5)
    plt.savefig(os.path.join(folder_repspace_preview,f"preview_{epoch_num_str}.png"))
    plt.close()

    epoch_number += 1
