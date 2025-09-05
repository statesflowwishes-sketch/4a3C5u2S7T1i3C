Lege hier deine Impulsantworten (IR/BRIR) ab. Beispiel: brir_cinema_wide.wav
MOTHERSHIP – Technische Beschreibung (rein textlich)
1. Ziel und Grundprinzip

MOTHERSHIP ist eine Steuer- und Entwurfsumgebung für dreidimensionale Klanginszenierungen auf Basis psychoakustischer Signalverarbeitung. Reale Lautsprecher (auch kleine Boxen) oder Kopfhörer werden als „Fenster“ in einen virtuell erzeugten Raum genutzt. Alle Räumlichkeit entsteht vorab in der Engine: Position, Bewegung, Tiefe, und Mehrschicht-Mischungen verschiedener „Raum-Archetypen“.

2. Architekturübersicht

Kontrolleingabe
– Interaktive Bedienelemente liefern ausschließlich Steuerdaten: Pfad-Automation (Punkte A/B/C plus Zeit), XY-Pad (vier Eck-Archetypen), Dreiecksmischer (drei Archetypen), Morphing A↔B, Z-Achse (Tiefe).
– Externe Steuerung über OSC und MIDI ist vorgesehen.

Szenenabstraktion
– Eine Szene bündelt: bewegte Klangobjekte (Pfad), Layer-Anteile (XY/Dreieck), globale Tiefen-Parameter, Timing/Easing, Preset-Morphing.

DSP-Graph (Renderer)
– Bausteine: Upmixer/Decoder, binaurale/raumbezogene Faltung (HRTF/BRIR), Raum-Sends und Reverbs, optional Crosstalk-Cancellation, Raumkorrektur (FIR/EQ), Safety-Limiter.
– Ausgabe: Kopfhörer (binaural) oder Stereo-Lautsprecher (psychoakustisch getäuschte Räumlichkeit).

3. Psychoakustische Grundlage (knapp)

– HRTF: Richtung und Höhe werden über Kopf-/Ohrmodell nachgebildet.
– BRIR: reale Raumantworten als Faltung für „Glaubwürdigkeit“.
– Ambisonics/HOA: flexible Rotation des Schallfelds.
– Crosstalk-Cancellation: reduziert Übersprechen bei Lautsprecher-Wiedergabe, erfordert sorgfältige Dosis.
– Wellenfeld-Prinzipien: inspirieren die Verteilung auf vorhandene Treiber, ohne große Arrays zu benötigen.

4. Pfad-Automation (Bewegung A→B→C)

Geometrie
– Drei Keypoints definieren die Bahn. Der Verlauf wird geglättet, sodass keine Ecken hörbar sind.
– Konstante Bewegungsgeschwindigkeit entsteht durch Orientierung an der tatsächlichen Pfadlänge (nicht nur am Index der Punkte).

Timing
– Keyframe-Zeiten: A ist Startzeit 0, B ist Zwischenzeit, C ist Endzeit 1. Die Zeitachse wird passend auf die beiden Segmente verteilt.
– Optional Quantisierung der Zeit (gleichmäßige Schritte) für reproduzierbare Automation.

Easing
– Zeitkurven formen Start-/Zielverhalten (z. B. weich anfahren, weich ausrollen). Die Geometrie bleibt identisch, nur die Zeit „fließt“ anders.

Z-Achse (Tiefe)
– Z wird als 0…1 geführt und in der Engine sinnvoll abgebildet, z. B. auf Raum-Send, Pre-Delay, sanfte Tiefpass-/Tiefen-EQ-Anteile, Reverb-Größe.
– Ziel: Nähe klingt trockener und präsenter, Ferne klingt diffuser, weicher, leicht gefiltert.

Mehrspur-Objekte
– Bis zu vier gleichartige „Reisepunkte“ können den Pfad mit unterschiedlichen Phasenversätzen befahren.
– Solo/Mute pro Spur regelt Beteiligung. Phasenversatz ermöglicht Choruseffekte oder räumlich versetzte Doppellungen.

Laufzeit-Logik
– Zur Ausführungszeit wird aus aktueller Zeit, Keyframe-Verteilung und gewählter Easing-Kurve die normierte Position entlang der Bahn bestimmt; daraus ergibt sich die Raumpose des Objekts.

5. Layer-Mischung (XY-Pad, 4 Ecken)

Bedeutung der Ecken
– Beispielhafte Archetypen: Kino (breit, diffus), Proberaum (trocken, klein), Nebenraum (gedämpft, gefiltert, frühe Reflexionen), Intim (sehr nah, wenig Raum).
– Die Position im Quadrat bestimmt die vier Gewichte. Summe der Gewichte beträgt stets 100 %.

Best Practices
– Bewegungen über das Pad sollten geglättet werden (interne zeitliche Mittelung), damit keine abrupten Raumsprünge entstehen.
– Ecken definieren nur Charaktere; die konkreten DSP-Zuweisungen (welcher Reverb, welche Filter, wie viel Send) sind frei gestaltbar, aber konstant dokumentiert.

6. Dreiecksmischer (3 Layer)

Einsatz
– Wenn drei prägnante Raum-Archetypen genügen, liefert der Dreiecksmischer besonders robuste, musikalische Übergänge.
– Die Position innerhalb des Dreiecks wird auf drei Gewichte verteilt, wiederum mit Gesamtsumme 100 %.
– Typisch für dramaturgische Szenen: zwischen „nah“, „mittel“, „weit“ wandern.

7. Preset-Morphing A↔B

Zweck
– Zwei komplette Zustände (Pfadpunkte, XY-Position, Dreieckspunkt, ggf. globale Parameter) werden gespeichert. Ein Morph-Regler überblendet kontinuierlich.
– Sicherstellen, dass während des Morphens keine unzulässigen Sprünge in Gain, Delay oder Filter auftreten; Übergänge sollten auditiv glatt sein.

Anwendung
– Live-Performance (Morph am Controller), Übergänge zwischen Strophen/Refrains, Film-Cuts ohne Hard-Edges.

8. Externe Steuerung (Konzept, ohne Protokolldetails)

OSC
– Klare, kurze Adressen für kontinuierliche Steuergrößen (z. B. X, Y, Einzellayer-Gewichte, Morph-Anteil).
– Update-Raten so begrenzen, dass keine Paketflut entsteht; interne Glättung gegen Jitter.

MIDI
– Zuordnung einzelner Controller-Nummern für X, Y und Morph.
– Optional: weitere CCs für Z (Tiefe), Gesamtlautstärke einzelner Layer, oder Spur-Solos.

Smoothing
– Eingehende Werte werden zeitlich sanft gefiltert (sehr kurze Glättung), damit Bewegungen haptisch sind, aber nicht zittern.

9. Persistenz und Teilen

– Automatisches Zwischenspeichern des Bedienzustands lokal, damit Arbeitsschritte nicht verloren gehen.
– Geteilte Zustände als kompakte, lesbare Preset-Daten; beim Laden werden nur definierte, sichere Parameter überschrieben.

10. Kalibrierung und Raumkorrektur

Messung
– Raumimpuls an der Hörposition (und optional an mehreren Punkten) aufnehmen; daraus Korrekturfilter und Referenzpegel gewinnen.

Korrektur
– Vor dem kreativen Raumsynthetisieren erfolgt die lineare Entzerrung des Systems.
– Kleine Boxen profitieren stark von präzisem FIR/EQ und kontrollierter Bassentlastung (Sub- oder sanfte Hochpassführung).

Crosstalk-Cancellation (bei Lautsprechern)
– Sparsam einsetzen, da kopfpositionskritisch. Ideal mit kleinem Sweetspot oder Head-Tracking.

Limiter
– Am Ende der Kette; genügend Headroom vorhalten. Keine Verzerrungen bei hektischen Automationen.

11. Betriebshinweise je Plattform

– Windows: Routing über virtuelle Treiber; Korrektur und Faltung mit gängigen Tools; darauf aufbauend die MOTHERSHIP-Steuerdaten einspeisen.
– macOS: Virtuelle Devices für Routing; modulare Effektketten; präzise FIR-Stufen möglich.
– Linux: PipeWire-Routing; grafische Ketten oder explizite DSP-Konfiguration.
(Die konkrete Toolwahl bleibt frei; wichtig ist die saubere Reihenfolge und Stabilität.)

12. Gain-Staging, Latenz, Sicherheit

– Abmischung stets mit Reserve; Automation kann Peaks erzeugen.
– Latenzbudget so wählen, dass Interaktion „echtzeitnah“ bleibt; Convolver-Längen und Puffergrößen sinnvoll abstimmen.
– Notfall-Limiter mit True-Peak-Erkennung; vorsichtiger Einsatz, um Pumpen zu vermeiden.

13. Test- und Abnahmekriterien

– Bewegung entlang des Pfads wirkt zeitlich gleichmäßig; die Zwischenzeit B teilt die Gesamtbewegung nachvollziehbar.
– Easing beeinflusst nur die Tempokurve, nicht die Form der Bahn.
– XY- und Dreiecksmischungen bleiben in Summe konstant; keine plötzlichen Pegel- oder Spektralsprünge.
– Mehrspur-Reisepunkte respektieren Solo/Mute und Phasenversatz, ohne Kollisionen im Pegel.
– Preset-Morphing verläuft hörbar glatt; keine „Klicks“, kein Breitband-Rauschen, keine CPU-Spitzen.
– Externe Steuerung zeigt keine spürbaren Verzögerungen oder Sprünge; Glättung dämpft Jitter.

14. Roadmap (sachlich)

– Segmentweise Kurven-Editor für Zeitverlauf direkt an der Bahn.
– Mehr Keyframes pro Pfad und visuelles Zeitlineal mit Marker-Snapping.
– Head-Tracking zur Stabilisierung von Lautsprecher-Räumlichkeit.
– Export-Adapter in fertige DSP-Konfigurationen gängiger Renderer.
– Kuratierte Impulsantwort-Bibliothek für typische Räume und Lautsprecher.
