import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram


# konfiguracja całego skryptu
PLIK_EDF = 'Gruby4Hz_raw.edf'  # Tutaj wstawić nazwe pliku (musi być w tym samym folderze co skrypt)
CZESTOTLIWOSC_SSVEP = 4        # Badana częstotliwość dla Scenariusza 1 (w HZ)

# PARAMETRY DLA SCENARIUSZY ERP
TRIGGER_TARGET = 101                                        # Wartość triggera dla bodźca poszukiwanego
TRIGGER_NOTARGET = [110, 120, 130, 140, 150, 160]           # Wartość triggera dla bodźca neutralnego (jeśli ich są nie poprawne to znalezione pojawią się w konsoli)

# Okno czasowe wokół bodźca dla ERP (w sekundach)
T_MIN = -0.2                   
T_MAX = 0.8                    

# Wczytywanie i dopasowywanie kanałów
print(f"--- ROZPOCZYNAM ANALIZĘ DANYCH EEG: {PLIK_EDF} ---")
raw = mne.io.read_raw_edf(PLIK_EDF, preload=True, verbose=False)
Fs = raw.info['sfreq']                                      #odczytuje częstotliwość próbkowania
ch_names = raw.info['ch_names']                             #odczytuje nazwy kanałów

print(f"\n[INFO] Wykryte kanały w pliku EDF: {ch_names}")

# Funkcja automatycznie dopasowująca kanały (ignoruje wielkość liter, spacje i myślniki)
def auto_znajdz_kanal(lista_kanalow, klucz):
    for ch in lista_kanalow:
        znormalizowany = ch.lower().replace(' ', '').replace('_', '').replace('-', '')
        if klucz.lower() in znormalizowany:
            return ch
    return None

# Automatyczne mapowanie
ch_O1 = auto_znajdz_kanal(ch_names, 'O1')
ch_O2 = auto_znajdz_kanal(ch_names, 'O2')
ch_Fp1 = auto_znajdz_kanal(ch_names, 'Fp1')
ch_trig = auto_znajdz_kanal(ch_names, 'trig') or auto_znajdz_kanal(ch_names, 'event') or auto_znajdz_kanal(ch_names, 'trg')

if not ch_O1 or not ch_O2 or not ch_Fp1:
    raise ValueError(f"\n[BŁĄD] Nie udało się automatycznie dopasować elektrod bazowych!\n"
                     f"Znaleziono: O1->{ch_O1}, O2->{ch_O2}, Fp1->{ch_Fp1}.\n"
                     f"Sprawdź powyższą listę wykrytych kanałów i upewnij się, że plik zawiera te elektrody.")

print(f"\n=> Pomyślnie dopasowano kanały strukturalne:")
print(f"   * Kora potyliczna lewa  (O1)  -> '{ch_O1}'")
print(f"   * Kora potyliczna prawa (O2)  -> '{ch_O2}'")
print(f"   * Kora czołowa          (Fp1) -> '{ch_Fp1}'")
print(f"   * Kanał synchronizacji  (Trig)-> '{ch_trig}'\n")

# Filtrowanie cyfrowe sygnału
print("=> Filtrowanie cyfrowe sygnału (Bandpass 1.0 - 40.0 Hz)...")
raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=40.0, method='iir', verbose=False)
raw_filtered.notch_filter(freqs=50.0, verbose=False)

# Pobranie indeksów pozycji dla wyciętych nazw
idx_O1 = ch_names.index(ch_O1)
idx_O2 = ch_names.index(ch_O2)
idx_Fp1 = ch_names.index(ch_Fp1)

# Funkcja do analizy SSVEP
def analizuj_ssvep(raw_obj, raw_filt_obj, f_target):
    print("--- URUCHAMIAM ANALIZĘ WIDMOWĄ SSVEP ---")
    
    if not ch_trig:
        print("[UWAGA] Brak sprzętowego kanału Trigger. Stosuję automatyczne cięcie czasowe (gilotyna).")
        start_idx = int(3 * Fs)
        data = raw_filt_obj.get_data()
        s_O1 = data[idx_O1, start_idx:]
        s_O2 = data[idx_O2, start_idx:]
        s_Fp1 = data[idx_Fp1, start_idx:]
    else:
        trig_data = raw_obj.get_data(picks=ch_trig)[0]
        threshold = np.max(trig_data) * 0.5
        active_idxs = np.where(trig_data > threshold)[0]
        
        if len(active_idxs) == 0:
            print("[UWAGA] Kanał Trigger jest płaski lub nie wykryto impulsów. Stosuję gilotynę czasową.")
            start_idx = int(3 * Fs)
            stop_idx = raw_filt_obj.n_times
        else:
            start_idx = active_idxs[0] + int(2.0 * Fs)
            stop_idx = active_idxs[-1] - int(1.0 * Fs)
        
        data = raw_filt_obj.get_data()
        s_O1 = data[idx_O1, start_idx:stop_idx]
        s_O2 = data[idx_O2, start_idx:stop_idx]
        s_Fp1 = data[idx_Fp1, start_idx:stop_idx]

    # Filtr przestrzenny (Wirtualna elektroda Oz)
    s_Oz = (s_O1 + s_O2) / 2.0 * 1e6  # Konwersja do uV
    s_Fp1_uV = s_Fp1 * 1e6
    
    # Obliczanie FFT
    L = len(s_Oz)
    freqs = fftfreq(L, 1/Fs)[:L//2]
    
    Y_Oz = fft(s_Oz - np.mean(s_Oz))
    P_Oz = 2.0 / L * np.abs(Y_Oz[:L//2])
    
    Y_Fp1 = fft(s_Fp1_uV - np.mean(s_Fp1_uV))
    P_Fp1 = 2.0 / L * np.abs(Y_Fp1[:L//2])
    
    # Wykresy Widma
    lim_x = 25 if f_target < 20 else 60
    plt.figure(figsize=(12, 7))
    plt.suptitle(f"Analiza SSVEP - Stymulacja {f_target} Hz (Plik: {PLIK_EDF})", fontsize=12, fontweight='bold')
    
    plt.subplot(2, 1, 1)
    plt.plot(freqs, P_Oz, 'b-', linewidth=1.5, label='Kora Wzrokowa (Wirtualne Oz)')
    plt.axvline(x=f_target, color='r', linestyle='--', alpha=0.8, label=f'Częstotliwość f0 ({f_target} Hz)')
    if f_target * 2 < lim_x:
        plt.axvline(x=f_target*2, color='m', linestyle=':', alpha=0.8, label=f'Harmoniczna 2xf0 ({f_target*2} Hz)')
    plt.ylabel('Amplituda [$\mu$V]')
    plt.xlim(0, lim_x)
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(freqs, P_Fp1, 'r-', linewidth=1, label='Kora Czołowa (Fp1 - Artefakty/Szum)')
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('Amplituda [$\mu$V]')
    plt.xlim(0, lim_x)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Analiza ERP
def analizuj_erp(raw_filt_obj, t_target, t_notarget):
    print("--- URUCHAMIAM ANALIZĘ POTENCJAŁÓW WYWOŁANYCH ERP ---")
    if not ch_trig:
        print("[BŁĄD] Brak kanału Trigger. Nie można zsynchronizować epok dla ERP.")
        return
        
    try:
        events = mne.find_events(raw_filt_obj, stim_channel=ch_trig, min_duration=0.005, verbose=False)
        print("=> WYKRYTE MARKERY I ICH LICZBA:", {k: list(events[:, 2]).count(k) for k in np.unique(events[:, 2])})
    except:
        events = mne.find_events(raw_filt_obj, stim_channel=None, min_duration=0.005, verbose=False)
        print("=> WYKRYTE MARKERY I ICH LICZBA:", {k: list(events[:, 2]).count(k) for k in np.unique(events[:, 2])})

    # Grupowanie wielu triggerów NoTarget w jeden wspólny wirtualny znacznik (np. 999)
    if isinstance(t_notarget, list):
        print(f"=> Łączenie markerów tła {t_notarget} w jedną grupę do uśrednienia...")
        events = mne.merge_events(events, t_notarget, 999, replace_events=True)
        t_notarget_id = 999
    else:
        t_notarget_id = t_notarget

    event_id = {'Target': t_target, 'NoTarget': t_notarget_id}
    dostepne_ids = {k: v for k, v in event_id.items() if v in events[:, 2]}
    
    if not dostepne_ids:
        print(f"[BŁĄD] Triggery {event_id} nie pasują do pliku. Wykryte to: {np.unique(events[:, 2])}")
        return
        
    epochs = mne.Epochs(raw_filt_obj, events, event_id=dostepne_ids, tmin=T_MIN, tmax=T_MAX,
                        baseline=(None, 0), preload=True, verbose=False)
    
    evokeds = {k: epochs[k].average() for k in dostepne_ids.keys()}
    times = epochs.times * 1000 
    
    plt.figure(figsize=(10, 5))
    plt.title(f"Uśredniony Wzorzec ERP (Elektrody potyliczne O1/O2) - Plik: {PLIK_EDF}")
    
    if 'Target' in evokeds:
        data_target = (evokeds['Target'].get_data(picks=idx_O1)[0] + evokeds['Target'].get_data(picks=idx_O2)[0]) / 2.0 * 1e6
        plt.plot(times, data_target, 'r-', linewidth=2, label=f"Target (n={len(epochs['Target'])})")
    if 'NoTarget' in evokeds:
        data_notarget = (evokeds['NoTarget'].get_data(picks=idx_O1)[0] + evokeds['NoTarget'].get_data(picks=idx_O2)[0]) / 2.0 * 1e6
        plt.plot(times, data_notarget, 'b--', linewidth=1.5, label=f"NoTarget (n={len(epochs['NoTarget'])})")
        
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    plt.axvspan(300, 500, color='gray', alpha=0.15, label='Okno P300')
    plt.xlabel('Czas [ms]')
    plt.ylabel('Amplituda [$\mu$V]')
    plt.grid(True)
    plt.legend()
    plt.show()


#Tutaj wywołać funkcję którą chce
analizuj_ssvep(raw, raw_filtered, CZESTOTLIWOSC_SSVEP) #do SSVEP
# analizuj_erp(raw_filtered, TRIGGER_TARGET, TRIGGER_NOTARGET) #do ERP