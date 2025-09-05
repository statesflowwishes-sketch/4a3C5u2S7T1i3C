# Impulsantworten (IR/BRIR) Repository

**Lege hier deine Impulsantworten (IR/BRIR) ab. Beispiel: brir_cinema_wide.wav**

## Ordnerstruktur

```
irs/
├── cinema/                 # Kino-Räume
│   ├── brir_cinema_wide.wav
│   ├── brir_cinema_intimate.wav
│   └── brir_cinema_surround.wav
├── studios/               # Proberäume & Studios
│   ├── brir_studio_dry.wav
│   ├── brir_studio_live.wav
│   └── brir_studio_booth.wav
├── rooms/                 # Wohnräume & kleinere Räume
│   ├── brir_room_living.wav
│   ├── brir_room_bedroom.wav
│   └── brir_room_office.wav
├── halls/                 # Konzertsäle & große Räume
│   ├── brir_hall_classical.wav
│   ├── brir_hall_modern.wav
│   └── brir_hall_church.wav
└── outdoor/               # Außenräume
    ├── brir_outdoor_park.wav
    ├── brir_outdoor_street.wav
    └── brir_outdoor_forest.wav
```

## Namenskonvention

**Format:** `[typ]_[raum]_[charakteristik].[format]`

- **typ:** brir, ir, hrtf
- **raum:** cinema, studio, room, hall, outdoor
- **charakteristik:** wide, intimate, dry, live, warm, bright
- **format:** wav, aiff

## Beispiele

- `brir_cinema_wide.wav` - Breites Kino mit diffuser Akustik
- `brir_studio_dry.wav` - Trockener Proberaum
- `brir_room_intimate.wav` - Intimer Wohnraum
- `hrtf_standard_kemar.wav` - Standard KEMAR HRTF

## Technische Spezifikationen

- **Samplerate:** 48 kHz (empfohlen) oder 44.1 kHz
- **Bittiefe:** 24-bit (empfohlen) oder 16-bit
- **Kanäle:** 
  - BRIR: Stereo (L/R) oder Quadro (FL/FR/RL/RR)
  - HRTF: Stereo (L/R)
- **Länge:** 
  - Kleine Räume: 0.5-2 Sekunden
  - Große Räume: 2-8 Sekunden
  - Outdoor: bis zu 12 Sekunden

## MOTHERSHIP Integration

Diese Impulsantworten werden von MOTHERSHIP automatisch erkannt und können den vier Layer-Archetypen zugeordnet werden:

1. **Kino-Layer:** `cinema/` Ordner
2. **Proberaum-Layer:** `studios/` Ordner  
3. **Nebenraum-Layer:** `rooms/` Ordner
4. **Intim-Layer:** Kurze IRs aus `rooms/` oder spezielle `intimate/` IRs

## Qualitätskontrolle

Vor der Verwendung sollten alle IRs geprüft werden:

- [ ] Kein DC-Offset
- [ ] Normalisiert auf -6dB Peak
- [ ] Kein Clipping
- [ ] Sauberer Fade-Out
- [ ] Konsistente Lautstärke zwischen L/R

## Lizenzhinweise

Stelle sicher, dass alle verwendeten Impulsantworten:
- Frei verfügbar oder lizenziert sind
- Urheberrechtlich unbedenklich sind  
- Für kommerzielle Nutzung freigegeben sind (falls relevant)

## README-Datei pro IR

Jede Impulsantwort sollte mit einer begleitenden `.txt` Datei dokumentiert werden:

```
brir_cinema_wide.wav
brir_cinema_wide.txt
```

Inhalt der Textdatei:
```
Raum: Multiplexkino, Saal 3
Aufnahmeort: Reihe 8, Mitte
Mikrofon: Neumann KU100 Dummy Head
Datum: 15.08.2025
Besonderheiten: Textile Verkleidung, 180 Plätze
Nachhallzeit RT60: 1.2s
Lizenz: Creative Commons BY-SA 4.0
```