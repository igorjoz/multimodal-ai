# Converted from contrastive loss.ipynb


# %% [markdown]
# ##### AI TECH - Akademia Innowacyjnych Zastosowań Technologii Cyfrowych. Programu Operacyjnego Polska Cyfrowa na lata 2014-2020

# %% [markdown]
# AI-TECH
# LABORATORIUM 3
# z Przetwarzania multimediów w systemach decyzyjnych
# Przegląd algorytmów uczących się metryk dystansu dla
# danych wizyjnych
# Adam Kurowski
# algorytm straty kontrastowej (contrastive loss)

# %% [markdown]
# ![stopka.svg](stopka.svg)

# %% [markdown]
# Projekt współfinansowany ze środków Unii Europejskiej w ramach Europejskiego Funduszu Rozwoju Regionalnego
# Program Operacyjny Polska Cyfrowa na lata 2014-2020,
# Oś Priorytetowa nr 3 "Cyfrowe kompetencje społeczeństwa" Działanie  nr 3.2 "Innowacyjne rozwiązania na rzecz aktywizacji cyfrowej"
# Tytuł projektu:  „Akademia Innowacyjnych Zastosowań Technologii Cyfrowych (AI Tech)”

# %% [markdown]
# # Trening sieci algorytmem straty kontrastowej

# %% [markdown]
# Algorytm straty kontrastowej w porównaniu do wykorzystywanej implementacji algorytmu starty trójkowej jest nieco szybszy.

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
# # Definicje funkcji składowych algorytmu contrastive loss i trening

# %%
def contrastive_loss_function(pattern_pos_A, pattern_pos_B, pattern_neg_A, pattern_neg_B, distance_norm='euclidean', margin=1):
    distance_pos   = tf.norm(pattern_pos_B - pattern_pos_A, ord=distance_norm, axis=1)
    distance_neg   = tf.norm(pattern_neg_B - pattern_neg_A, ord=distance_norm, axis=1)
    partial_losses = distance_pos + tf.math.maximum(0, margin - distance_neg)
    loss_value     = tf.math.reduce_mean(partial_losses)
    return loss_value

def mine_pairs(trn_X, trn_Y):
    classes_available = np.unique(trn_Y)
    positive_class    = np.random.choice(classes_available)
    
    positive_class_examples = trn_X[trn_Y==positive_class]
    negative_class_examples = trn_X[trn_Y!=positive_class]
    
    positive_pattern_idx    = np.random.choice(len(positive_class_examples),2,replace=False)
    
    negative_pattern_idx  = np.random.choice(len(negative_class_examples),2,replace=False)
    
    pattern_pos_A = positive_class_examples[positive_pattern_idx[0]]
    pattern_pos_B = positive_class_examples[positive_pattern_idx[1]]
    pattern_neg_A = negative_class_examples[negative_pattern_idx[0]]
    pattern_neg_B = negative_class_examples[negative_pattern_idx[1]]
    
    return (pattern_pos_A, pattern_pos_B, pattern_neg_A, pattern_neg_B)

# Przygotowanie środowiska pracy:
if not os.path.isdir(folder_repspace_preview): os.mkdir(folder_repspace_preview)
if not os.path.isdir(folder_model_weights):    os.mkdir(folder_model_weights)

# Opcjonalnie - można załadować wagi modelu:
if os.path.isfile(os.path.join(folder_model_weights, 'model.weights.h5')):
    encoder_model.load_weights(os.path.join(folder_model_weights, 'model.weights.h5'))

# Na koniec rozpoczynamy właściwy trening sieci:
epoch_number  = 1
optimizer     = tf.keras.optimizers.Adam(learning_rate)

print("trening sieci neuronowej algorytmem contrastive loss został rozpoczęty")
print()

# trenujemy, aż nie otrzymamy sygnału wstrzymania z klawiatury (ctrl+c):
batch_matrix_shape = tuple([batch_size] + list(input_dims))

for epoch_number in range(1,num_training_epochs+1):

    # --------------------------------------------------------
    # właściwy algorytm treningu sieci:

    print(f"epoka: {epoch_number}")

    # Korzystamy z własnego zdefiniowania algorytmu treningu, stąd musimy posłuzyć się kontekstem GradientTape:
    with tf.GradientTape() as tape:
        accumulated_loss = 0

        # Dla każdej grupy danych:
        for batch_idx in tqdm(range(batches_per_epoch)):

            pattern_pos_A_mtx = np.zeros(batch_matrix_shape)
            pattern_pos_B_mtx = np.zeros(batch_matrix_shape)
            pattern_neg_A_mtx = np.zeros(batch_matrix_shape)
            pattern_neg_B_mtx = np.zeros(batch_matrix_shape)

            for i in range(batch_size):
                pattern_pos_A, pattern_pos_B, pattern_neg_A, pattern_neg_B = mine_pairs(trn_X, trn_Y)
                pattern_pos_A_mtx[i,::] = pattern_pos_A
                pattern_pos_B_mtx[i,::] = pattern_pos_B
                pattern_neg_A_mtx[i,::] = pattern_neg_A
                pattern_neg_B_mtx[i,::] = pattern_neg_B

            # Oblicz reprezentacje wyłonionych przykładów:
            representations_pos_A = encoder_model(pattern_pos_A_mtx, training=True)
            representations_pos_B = encoder_model(pattern_pos_B_mtx, training=True)
            representations_neg_A = encoder_model(pattern_neg_A_mtx, training=True)
            representations_neg_B = encoder_model(pattern_neg_B_mtx, training=True)

            # Oblicz wartość funkcji straty i dodaj ją do zmiennej, która na koniec wykonania wewnętrznej pętli będzie zawierać
            # średnią wartość straty dla całej epoki:
            loss_value = contrastive_loss_function(representations_pos_A, representations_pos_B, representations_neg_A, representations_neg_B, distance_norm=distance_norm, margin=margin)
            accumulated_loss += loss_value*tf.constant(1./batches_per_epoch)

    # Wylicz gradienty na podstawie funkcji straty:
    grads = tape.gradient(accumulated_loss, encoder_model.trainable_weights)

    # Zaaplikuj poprawki do wag sieci za pomocą algorytmu wstecznej propagacji błędów
    optimizer.apply_gradients(zip(grads, encoder_model.trainable_weights))

    # --------------------------------------------------------
    # wizualizacja, zapis wag i raportowanie postępów
    info_text = f"wartość straty kontrastowej (contrastive loss): {accumulated_loss}"
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
