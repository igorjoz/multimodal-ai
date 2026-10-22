# Converted from P2_przetwarzanie_GTZAN.ipynb

# AI TECH - Akademia Innowacyjnych Zastosowań Technologii Cyfrowych. Programu Operacyjnego Polska Cyfrowa na lata 2014-2020
# AI-TECH

# LABORATORIUM 4
# z Przetwarzania multimediów w systemach decyzyjnych
# Przegląd algorytmów uczących się metryk dystansu dla
# danych akustycznych
# Adam Kurowski
# przetwarzanie zbioru GTZAN sparametryzowanego za pomocą spektrogramów i MFCC-gramów
# stopka.svg
# Projekt współfinansowany ze środków Unii Europejskiej w ramach Europejskiego Funduszu Rozwoju Regionalnego
# Program Operacyjny Polska Cyfrowa na lata 2014-2020,
# Oś Priorytetowa nr 3 "Cyfrowe kompetencje społeczeństwa" Działanie  nr 3.2 "Innowacyjne rozwiązania na rzecz aktywizacji cyfrowej"
# Tytuł projektu:  „Akademia Innowacyjnych Zastosowań Technologii Cyfrowych (AI Tech)”
# Pobranie danych z Kaggle
# W ćwiczeniu zilustrujemy przetwarzanie sygnałów akustycznych za pomocą przetwarzania nagrań muzycznych. Jednym ze zbiorów danych często wybieranych do nauki przetwarzania sygnałów muzycznych jest zbiór GTZAN. Niestety, w bibliotece Tensorflow Datasets, która zwykle jest wygodnym wyborem do pozyskiwania danych, pobieranie tego zbioru danych następuje z bardzo wolnego serwera, dlatego w tym ćwiczeniu zalecane jest posłużenie się danymi pobranymi z serwisu Kaggle:

# https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification
#
# Pobrane archiwum (archive.zip) należy następnie załadować na własny dysk Google Drive (do głównego folderu), a ten podłączyć do środowiska Google Colab. W menu plików maszyny wirtualnej pojawi się możliwość skopiowania pliku do maszyny wirtualnej, co należy uczynić wykonując komórkę poniżej:
# %%
# Colab-only Drive setup; skipped because the dataset is provided locally.
# from google.colab import drive
# drive.mount('/content/drive')
# Następnie można wykonać poniższą komórkę w celu rozpakowania archiwum. Zakłada ona, że archiwum jest w głównym folderze dysku, skąd jest ono kopiowane do pamięci sesji i rozpakowywane.
# %%
# Colab shell commands for downloading/extracting the dataset; extraction is user-managed.
# !cp drive/MyDrive/archive.zip archive.zip
# !unzip archive.zip
# Import bibliotek koniecznych do dalszego wykonania ćwiczenia:
# %%

import numpy as np
import os
from   scipy.io.wavfile import read
import matplotlib.pyplot as plt
import copy as cp
from scipy.fftpack import fft
from librosa.feature import mfcc
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers     import Input, Dense, Conv2D, Flatten, PReLU,ReLU, MaxPooling2D, Lambda
from tensorflow.keras.models     import Model
from tensorflow.keras.utils      import to_categorical
from tensorflow.keras            import backend as K
from numpy.linalg                import norm
from sklearn.decomposition       import PCA
import tensorflow as tf
# Nastawy pracy algorytmu:
# %%
# ------------------------------------------------------------------
# Ścieżka do zbioru danych GTZAN:
gtzan_dataset_inpath = "Data/genres_original"

# Nastawy przetwarzania pliku za pomocą spektrogramów
frame_length   = 1024 # szerokość okna analizy (w próbkach)
overlap_factor = 0.0  # współczynnik nachodzenia się na siebie 
                      # ramek danych (okien analizy)

# Nastawy przetwarzania pliku za pomocą MFCC-gramów:
n_mfcc         = 40   # liczba współczynników MFCC w ramce
n_fft          = 1024 # długość FFT wykorzystana do obliczenia 
                      # współczynników MFCC
fft_window     = np.ones(frame_length) # funkcja okna
# Inne okna można wygenerować funkcjami z:
# https://numpy.org/doc/stable/reference/routines.window.html

# parameterization_method = 'spectrogram'
parameterization_method = 'mfcc'

# ------------------------------------------------------------------
training_ratio   = 0.9
embedding_length = 20

# Nastawy algorytmu treningu
learning_rate       = 0.01
batches_per_epoch   = 10
batch_size          = 16
num_training_epochs = 200

# Nastawy algorytmu uczącego się metryk odległości
margin            = 100  # ograniczenie na docelową odległość między klastrami punktów
distance_norm     = 'euclidean' # metryka odległości Ln, może być też 1,2, czy np.inf

# Nazwy folderów na dane:
folder_model_weights    = "_weights"
folder_repspace_preview = "_previews"
# Parametryzacja zbioru GTZAN
# Po pozyskaniu danych możliwe staje się przeanalizowanie danych. Najpierw wykorzystamy do tego analizę za pomocą algorytmu FFT, a dokładniej - spektrogramów. Wpierw zdefiniujmy kilka funkcji pomocnicznych:
# %%
# Funkcja wycinająca wybrany fragment sygnału o zadanej szybkości próbkowania fs
# (miejsca cięć podajemy w sekundach)
def obtain_signal_slice(insignal, fs, cut_start, cut_stop):
    cut_start_idx = int(fs*cut_start)
    cut_stop_idx  = int(fs*cut_stop)

    if cut_stop_idx>len(insignal):
        raise RuntimeError("błąd - czas zakończenia cięcia jest późniejszy niż długość nagrania")

    output_slice = insignal[cut_start_idx:cut_stop_idx]
    return output_slice
  
# Pomocnicza funkcja do podziału danych na ramki:
#    - możliwe jest zadanie długości ramki,
#    - oraz okreslenie, w jakim ułamku ramki te powinny na siebie nachodzić
#    (tzw. overlap factor).
def split_to_frames(insignal, frame_length, overlap_factor):
    shift_factor        = frame_length*(1-overlap_factor)
    target_frames_count = (len(insignal)-frame_length)/shift_factor
    shift_factor        = int(shift_factor)
    target_frames_count = int(target_frames_count)
    output_data         = np.zeros((frame_length, target_frames_count))

    for i in range(target_frames_count):
        frame_start      = i*shift_factor
        frame_end        = frame_start + frame_length
        output_data[:,i] = insignal[frame_start:frame_end]
    
    return output_data

# Funkcja licząca spektrogram na podstawie danych audio podzielonych na ramki
def frames_to_spectrogram(frames_mtx, fft_window):
    output_data = cp.copy(frames_mtx)
    for i in range(output_data.shape[1]):
        output_data[:,i] = np.abs(fft(output_data[:,i]*fft_window))
    
    output_data = output_data[0:output_data.shape[0]//2,:]
    return output_data

# Funkcja która łącznie dzieli sygnał na ramki i liczy spektrogram.
def signal_to_spectrogram(ex, frame_length, overlap_factor, fft_window):
    split_ex    = split_to_frames(ex, frame_length, overlap_factor)
    spectrogram = frames_to_spectrogram(split_ex, fft_window)
    return spectrogram


parameterized_audio = []
labels              = []
fs                  = None

# Sprawdzenie plików do odczytu
num_files = 0
for folder_name in os.listdir(gtzan_dataset_inpath):
    subfolder_path = os.path.join(gtzan_dataset_inpath,folder_name)
    for fname in os.listdir(subfolder_path):
        _, ext = os.path.splitext(fname)
        if ext == '.wav':
            num_files += 1

# Odczyt plików do pamięci
error_prompts = []

print()
print("Odczyt i parametryzacja plików z dysku:")
pbar = tqdm(total=num_files)
# Odczyt plików ze zbioru danych:
for folder_name in os.listdir(gtzan_dataset_inpath):
    subfolder_path = os.path.join(gtzan_dataset_inpath,folder_name)
    for fname in os.listdir(subfolder_path):
        # Upewniamy się, że odczytujemy plik .wav zawsze o takich samych nastawach jak np. 
        # szybkość próbkowania.
        _, ext = os.path.splitext(fname)
        if ext == '.wav':
            fpath = os.path.join(subfolder_path, fname)
            try:
                fs_infile, audio_data  = read(fpath)
                
                if fs is None:
                    fs = fs_infile
                else:
                    if fs != fs_infile:
                        print(f"warning: sampling frequency of file {fname} does not match sample rates of other files, stopping data reading")
                
                audio_data  = audio_data.astype(float)
                
                # Pobierane jest jednocześnie wiele wycinków by zwielokrotnić liczbę przykładów treningowich.
                # Bez tego przykładów jest na tyle mało, że grozi to zanikiem zbieżności algorytmu
                # (sieć zwróci wartości NaN i taką też wartość będą miały jej wagi)
                for cut_start, cut_stop in [(2,5),(5,8),(8,11),(11,14),(14,17),(17,20),(20,23),(23,26)]:
                    audio_slice = obtain_signal_slice(audio_data, fs, cut_start, cut_stop)
                    
                    # Normalizujemy wycięty z sygnału audio fragment
                    audio_slice = audio_slice/np.max(np.abs(audio_slice))
                    
                    # Parametryzujemy sygnał"
                    if parameterization_method == 'spectrogram':
                        parameterized_example = signal_to_spectrogram(audio_slice, frame_length, overlap_factor, fft_window)
                    else:
                        parameterized_example = mfcc(y=audio_slice, sr=fs, n_mfcc=n_mfcc, n_fft=n_fft)
                        
                    # Normalizujemy pozyskany zestaw parametrów
                    parameterized_example = parameterized_example/np.max(np.abs(parameterized_example))
                    
                    # Zwracamy wygenerowany przykład i etykietę do głównego zbioru przykładów/etykiet.
                    parameterized_audio.append(parameterized_example)
                    labels.append(folder_name)
            except:
                error_prompts.append(f"unreadable file: {fname}, the file was omitted")
            pbar.update(1)
pbar.close()

labels = np.array(labels)

# Przygotowanie wymiarów w strukturze przechowującej dane - konieczne są wymiary:
# (liczba przykładów, długość ramki, nr ramki, 1), 1 wynika z faktu, że mamy sygnał 
# jednokanałowy (mono).
parameterized_audio = np.dstack(parameterized_audio)
parameterized_audio = np.moveaxis(parameterized_audio,-1,0)
parameterized_audio = parameterized_audio[np.newaxis,:]
parameterized_audio = np.moveaxis(parameterized_audio,0,-1)

np.random.seed(42) # blokujemy generator, by mieć powtarzalne wyniki

# Losowo zmieniamy kolejność przykładów:
shuffling_pattern   = np.arange(parameterized_audio.shape[0])
np.random.shuffle(shuffling_pattern)
parameterized_audio[shuffling_pattern]
num_training = int(training_ratio*parameterized_audio.shape[0])
num_test     = parameterized_audio.shape[0] - num_training

# Dzielimy pulę przykładów na docelowe zbiory:
trn_X  = parameterized_audio[shuffling_pattern][0:num_training]
test_X = parameterized_audio[shuffling_pattern][num_training:len(parameterized_audio)-1]
trn_Y  = labels[shuffling_pattern][0:num_training]
test_Y = labels[shuffling_pattern][num_training:len(labels)-1]
# Uruchomienie algorytmu triplet loss
# %%
# --------------------------------------------------------------------
# Definicja modelu kodera:
# --------------------------------------------------------------------

input_dims = parameterized_audio.shape[1::]

network_input = Input(shape=input_dims, name="left_input")
x = Conv2D(8,(2,2),  activation=PReLU(), padding='same')(network_input)
x = MaxPooling2D((2,4))(x)
x = Conv2D(8,(2,2),  activation=PReLU(), padding='same')(x)
x = MaxPooling2D((2,4))(x)
x = Flatten()(x)
x = Dense(128)(x)
embedding_vector = Dense(embedding_length, activation=None)(x)
encoder_model    = Model(inputs=network_input, outputs=embedding_vector)

print("Struktura kodera:")
encoder_model.summary()
print()

# --------------------------------------------------------------------

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

# Jeśli w folderze z wizualizacjami pozostały pliki po poprzednim uruchomieniu skryptu, to
# te pliki usuwamy.
for fname in os.listdir(folder_repspace_preview):
    fpath = os.path.join(folder_repspace_preview, fname)
    os.remove(fpath)

# Opcjonalnie - można załadować wagi modelu:
# if os.path.isfile(os.path.join(folder_model_weights,'model_weights.h5')):
    # try:
        # encoder_model.load_weights(os.path.join(folder_model_weights,"model_weights.h5"))
    # except ValueError as e:
        # print()
        # print("uwaga: nie udało się wczytać poprzednich wag modelu (np. dlatego, że zmieniła się struktura sieci, lub z innego powodu)")
        # print("opis błędu odczytu:")
        # print(e)
        # print()

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

    # Wizualizacja przestrzeni reprezentacji za pomocą PCA, jest to konieczne,
    # bo wizualizacja ma tylko 2 wymiary, a reprezentacje typowo mają wymiarów
    # znacznie więcej.
    try:
        test_embeddings = encoder_model(test_X)
        pca             = PCA(n_components=2, svd_solver='full')
        reduced         = pca.fit_transform(test_embeddings)
    except:
        print("wykonanie PCA nie powiodło się - model najprawdopdobniej utracił zbieżność, ostatnie poprawne wagi są zachowane na dysku")
        break
    
    # Zapis wag do późniejszego użycia:
    encoder_model.save_weights(os.path.join(folder_model_weights,'model.weights.h5'))

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

