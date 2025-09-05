# MOTHERSHIP – Blueprint (33 Seiten, rein textlich)

**Synthtographie für Raumklang: sauber beschrieben, ohne Code und ohne Grafiken**

Stand: 05.09.2025 • Schriftgröße: 13 pt • Format: A4 • Autor: MOTHERSHIP

Dieses Dokument ist die geordnete Grundlage für Konzeption, Abstimmung und Abnahme. Es fasst das System sauber zusammen – ausschließlich in Textform – und dient als Referenz für Kreativteams, Technik und Betrieb.

---

## Seite 1 – Vision & Zweck

MOTHERSHIP ist eine technische Regie für dreidimensionale Klangräume. Ziel ist es, reale Lautsprecher – auch sehr kleine – sowie Kopfhörer als Fenster in einen virtuell berechneten Raum zu nutzen. Das System erzeugt Räumlichkeit, Bewegung und Tiefenstaffelung vollständig in der Software und legt sie als akustische Schichten über den vorhandenen physikalischen Raum. Dadurch wird eine glaubhafte Raumillusion geschaffen, die reproduzierbar, skalierbar und live steuerbar ist.

Die Synthtographie-Idee: Klang wird nicht nur gemischt, sondern als Raumplastik komponiert – mit Layern, Pfaden und Zuständen, die sich dramaturgisch entwickeln. Das Dokument beschreibt präzise, wie diese Bausteine zusammenspielen: ohne Programmiercodes, ohne Grafiken – rein technisch-konzeptionell, sauber strukturiert.

Das System verwandelt traditionelle Stereophonie in eine dreidimensionale Gestaltungsebene, bei der Künstler:innen nicht nur links-rechts denken, sondern Objekte durch Räume führen, Tiefenstaffeln komponieren und verschiedene Raumcharaktere ineinander überblenden können. Diese Herangehensweise eröffnet neue dramaturgische Möglichkeiten für Musik, Theater, Installation und Film.

---

## Seite 2 – Begriffe & Scope

**Szene:** Ein vollständiger, in sich stimmiger Klangzustand aus Objekten, Layern, Tiefenparametern und globalen Einstellungen.

**Objekt:** Eine virtuelle Klangquelle, die Position, Bewegung und Layerbezüge besitzt.

**Layer:** Ein Raum-Archetyp, dessen Anteile sich mischen lassen (z. B. Kino, Proberaum, Nebenraum, Intim).

**Pfad:** Eine geglättete Bahn, entlang der sich ein Objekt zwischen Referenzpunkten bewegt.

**Morphing:** Überblendung zwischen zwei kompletten Szenenzuständen.

**Renderer:** Die DSP-Pipeline, die aus Steuerdaten hörbare Ergebnisse erzeugt.

**Archetyp:** Fest definierter Raumcharakter mit dokumentierten akustischen Eigenschaften.

**Tiefe:** Wahrgenommene Nähe/Ferne als Kombination mehrerer Parameter.

**Scope:** Dieses Blueprint definiert Anforderungen, Abläufe und Qualitätskriterien; Implementierungsdetails bleiben austauschbar, solange Schnittstellen und Resultate eingehalten werden. Die Dokumentation richtet sich an Entwicklungsteams, Klangregisseur:innen und technische Leitung gleichermaßen.

---

## Seite 3 – Nutzerrollen & Use-Cases

**Rollen:** Klangregisseur:in (künstlerische Gestaltung), Systemoperator (Kalibrierung/Betrieb), Technische Leitung (Qualität/Sicherheit).

**Use-Cases:**

• **Studio:** Dramaturgische Positionierung und Tiefenstaffelung einzelner Spuren. Komponist:innen können Instrumente nicht nur panoramieren, sondern sie durch verschiedene virtuelle Räume wandern lassen, dabei Nähe und Ferne modulieren und komplexe Bewegungsbögen entwerfen.

• **Bühne:** Live-Morphing zwischen Stimmungen und Räumen. Während einer Aufführung können Tonmeister:innen nahtlos zwischen verschiedenen Raumcharakteren wechseln, Objekte in Echtzeit bewegen und auf dramaturgische Wendungen reagieren.

• **Installation:** Dauerbetrieb mit stabiler Illusion trotz kleiner Lautsprecher. Museen und Ausstellungsräume profitieren von räumlichen Klanglandschaften, die auch bei begrenzter Hardware überzeugende Räumlichkeit erzeugen.

• **Forschung/Prototyping:** Evaluation psychoakustischer Parameter und Hörfelder. Wissenschaftliche Einrichtungen können kontrollierte Hörtests durchführen und neue Wahrnehmungsphänomene untersuchen.

---

## Seite 4 – Systemübersicht

Die Architektur trennt strikt zwischen Steuerung und Rendering.

Die **Steuerung** erzeugt ausschließlich Parameter: Objektpositionen, Layergewichte, Tiefenanteile, Zeitfunktionen und Zustandswechsel. Diese Ebene ist hardware-unabhängig und beschäftigt sich nur mit der künstlerischen Intention.

Der **Renderer** setzt diese Parameter in Klang um: Upmix/Decoding (falls nötig), räumliche Faltungen, Tiefenverteilung, eventuelle Übersprech-Unterdrückung bei Lautsprechern, Korrektur des realen Raums und abschließende Schutzmaßnahmen. Diese Ebene passt sich an die verfügbare Hardware und Raumakustik an.

Die Trennung gewährleistet Austauschbarkeit der Audio-Engine ohne Änderung der kreativen Steuerlogik. Künstlerische Entscheidungen bleiben bestehen, auch wenn die technische Umsetzung optimiert oder auf andere Plattformen portiert wird.

Die Kommunikation zwischen beiden Ebenen erfolgt über definierte, versionierte Schnittstellen mit klaren Datenformaten und Fehlersemantik.

---

## Seite 5 – Szenenmodell (High Level)

Eine Szene umfasst:

• **Objekte** mit Raumkoordinaten und Pfadzuweisung. Jedes Objekt trägt seine Position, Bewegungsparameter und Zuordnung zu Signalquellen.

• **Layer-Mischungen** in zwei Formen: vier Eck-Archetypen (Quadrat) und drei Archetypen (Dreieck). Diese definieren den Raumcharakter der gesamten Szene.

• **Globale Tiefenparameter** (Nähe/Ferne-Charakteristika). Diese beeinflussen alle Objekte gemeinsam und schaffen konsistente Tiefenwirkung.

• **Zeitlogik:** Keyframe-Timeline, Bewegungs-Easing, optionale Quantisierung. Die zeitliche Entwicklung folgt musikalischen und dramaturgischen Gesetzmäßigkeiten.

• **Zustandsverwaltung:** Presets, Morphing zwischen A und B, sowie Persistenz im Betrieb. Alle Einstellungen können gespeichert, geladen und live verändert werden.

• **Externe Kontrolle** über standardisierte, robuste Steuergrößen. OSC, MIDI und andere Protokolle ermöglichen Integration in bestehende Workflows.

Das Szenenmodell bildet die Grundlage für reproduzierbare künstlerische Arbeit und ermöglicht komplexe Automatisierungen bei gleichzeitiger Kontrolle aller Parameter.

---

## Seite 6 – Raumkoordinaten & Bezugsrahmen

Der Betriebsraum wird als normierter Bereich gedacht. Die **Breite** beschreibt eine linke–rechte Ausdehnung, die **Höhe** eine vertikale Staffelung, und die **Tiefe** eine wahrgenommene Nähe oder Ferne.

Wesentlich ist die relative Konsistenz: Alle Positionen, Bewegungen und Layer-Mischungen beziehen sich auf denselben normierten Rahmen. Dies sichert Übertragbarkeit zwischen Räumen unterschiedlicher Größe und akustischer Beschaffenheit.

Der normierte Raum erstreckt sich von -1 bis +1 in allen drei Dimensionen. Diese Konvention ermöglicht intuitive Bedienung und mathematisch saubere Interpolation. Zentral ist die Position (0,0,0), die als neutraler Ausgangspunkt für alle Bewegungen dient.

Die Koordinaten-Philosophie orientiert sich an der natürlichen Wahrnehmung: X-Achse entspricht der Breite des Hörfeldes, Y-Achse der Höhenwahrnehmung, Z-Achse der Entfernung. Diese Zuordnung bleibt über alle Systemkomponenten hinweg konstant.

Skalierung auf reale Räume erfolgt in der Renderer-Ebene und berücksichtigt Raumgröße, Lautsprecheranordnung und psychoakustische Grenzen der Wahrnehmung.

---

## Seite 7 – Pfad-Automation: Referenzpunkte

Die Bahn eines Objekts wird über Referenzpunkte A, B, C definiert, zwischen denen eine geglättete Verbindung entsteht.

Die **Glättung** ist so zu wählen, dass keine abrupten Richtungswechsel hörbar werden, aber die dramaturgische Intention (z. B. Bogen, S-Kurve, sanfter Umweg) erhalten bleibt. Mathematisch werden Spline-Interpolationen oder Bézier-Kurven eingesetzt, deren Parameter auf Hörbarkeit optimiert sind.

Die **Referenzpunkte** werden nachvollziehbar dokumentiert: Position im normierten Raum, Funktion im Verlauf (Start, Wendepunkt, Ziel), dramaturgische Rolle (Betonung, Übergang, Ruhepunkt).

Besondere Aufmerksamkeit gilt der Vermeidung von Sprüngen: Wenn ein Pfad geändert wird, erfolgt die Anpassung über sanfte Überblendung oder Neuberechnung ab dem aktuellen Punkt. Niemals dürfen abrupte Positionssprünge auftreten, die als störende Artefakte hörbar werden.

Die Anzahl der Referenzpunkte ist auf drei begrenzt, um Komplexität überschaubar zu halten, aber ausreichend für die meisten musikalischen und dramaturgischen Anforderungen.

---

## Seite 8 – Pfad-Automation: Bewegung & Konstanz

Die wahrgenommene Geschwindigkeit entlang der Bahn soll gleichmäßig sein, sofern dies gefordert ist. Dafür orientiert sich die Zeitauflösung an der effektiven Bogenlänge statt an bloßen Punktindizes.

Der **Vorteil:** Hörende erleben eine fließende Bewegung ohne „Stolperer", unabhängig von der Lage der Referenzpunkte. Dies ist besonders wichtig bei komplexeren Bahnen mit unterschiedlichen Segmentlängen.

**Technische Umsetzung:** Die Gesamtbahn wird in kleine Abschnitte gleicher Länge unterteilt. Die Zeitfunktion orientiert sich an dieser gleichmäßigen räumlichen Aufteilung, nicht an der ursprünglichen Punkt-zu-Punkt-Verbindung.

**Dramaturgische Flexibilität:** Konstante Geschwindigkeit ist der Standard, kann aber durch Easing-Funktionen gezielt verändert werden. So entstehen Beschleunigungen, Verlangsamungen oder komplexere Geschwindigkeitsverläufe, ohne die geometrische Präzision der Bahn zu beeinträchtigen.

Die Bewegungslogik berücksichtigt auch physikalische Plausibilität: Sehr schnelle Richtungsänderungen werden durch sanfte Übergänge ersetzt, die der natürlichen Trägheit entsprechen.

---

## Seite 9 – Keyframe-Timing & Zwischenzeiten

Die Zeitachse wird durch drei Markierungen strukturiert: **Start (A=0)**, **Zwischenzeitpunkt (B)**, **Endzeit (C=1)**.

Der **Zwischenzeitpunkt** teilt die Gesamtbewegung in zwei Teilabschnitte, deren Dauerverhältnis gezielt festgelegt wird. Dies ermöglicht asymmetrische Bewegungsverläufe, die oft musikalisch interessanter sind als gleichmäßige Aufteilungen.

**Beispiel:** Bei einem Zwischenzeitpunkt von B=0.3 dauert das erste Segment (A→B) 30% der Gesamtzeit, das zweite Segment (B→C) die verbleibenden 70%. Diese Aufteilung kann dramaturgischen Akzenten folgen oder rhythmische Strukturen unterstützen.

Die Zeitverteilung erlaubt **dramaturgisch sinnvolle Beschleunigungen** oder Dehnungen bestimmter Teile der Bewegung, ohne die Bahngeometrie zu verändern. So kann ein Objekt langsam anfahren, schnell durch die Mitte gleiten und sanft zum Stillstand kommen.

**Quantisierung:** Optional können Zwischenzeitpunkte auf musikalische Raster einrasten (1/4, 1/8, 1/16 Noten), um synchrone Bewegungen zu verschiedenen Taktelementen zu ermöglichen.

---

## Seite 10 – Easing der Zeitfunktion

**Easing** formt die Tempokurve einer Bewegung: sanftes Anfahren, fließendes Durchlaufen, weiches Ausrollen.

Die **Bahn bleibt geometrisch unverändert**, nur die Verteilung der Zeit über die Strecke ändert sich. Dies ist ein fundamentaler Unterschied zu Systemen, die Geschwindigkeitsänderungen durch Bahnmodifikation erreichen.

Die **Auswahl der Kurven** erfolgt nach Hördynamik: unmerklich, musikalisch und wiederholbar. Verfügbare Easing-Typen umfassen:

• **Linear:** Konstante Geschwindigkeit (Standard)
• **Ease-In:** Langsamer Start, normale Ankunft
• **Ease-Out:** Normaler Start, sanftes Ausrollen
• **Ease-In-Out:** Sanft in beide Richtungen
• **Spezielle Kurven:** Elastisch, Bounce, exponentiell

**Alle verwendeten Kurven werden benannt und beschrieben**, damit spätere Rekonstruktion möglich ist. Die mathematische Definition jeder Kurve ist dokumentiert und versioniert.

Die Wahl des Easings beeinflusst die emotionale Wirkung erheblich: mechanische Bewegungen wirken technisch, organische Kurven natürlich und lebendig.

---

## Seite 11 – Quantisierung & Reproduzierbarkeit

Zwei Arten von Quantisierung erhöhen Konsistenz:

**Positionsraster:** Referenzpunkte fangen auf definierte Rastersprünge ein, um wiederholbare Einstellungen zu sichern. Typische Rasterwerte sind 0.1, 0.05 oder 0.01 in normalisierten Koordinaten.

**Zeitraster:** Normalisierte Zeitpunkte werden auf feste Schritte gesetzt, damit Bewegungen exakt reproduzierbar sind. Musikalische Raster (1/4, 1/8, 1/16) ermöglichen rhythmische Synchronisation.

**Quantisierung ist stets abschaltbar**; künstlerische Freiheit geht vor, wenn notwendig. Der Zustand der Quantisierung wird mit der Szene gespeichert, sodass präzise Einstellungen und freie Gestaltung koexistieren können.

**Vorteile der Quantisierung:**
• Exakte Wiederholung von Bewegungen
• Vereinfachte Zusammenarbeit zwischen Künstler:innen
• Reduzierte Komplexität bei Live-Performance
• Kompatibilität mit musikalischen Strukturen

**Implementierung:** Quantisierung wirkt nur auf neue Eingaben, bestehende Pfade bleiben unverändert. Ein separater "Quantize"-Befehl kann nachträglich angewendet werden.

---

## Seite 12 – Z‑Achse: Nähe und Ferne

Die **Tiefe wird nicht als bloßer Pegel interpretiert**, sondern als koordinierte Variation mehrerer Parameter: Anteil des Raumanteils, Pre-Delay, Filterungen der hohen und tiefen Frequenzen, Kompaktheit der frühen Reflexionen.

**Nähe klingt präsent, klar, trocken:** Hoher Direktschallanteil, minimales Pre-Delay, linearer Frequenzgang, scharfe Ortung.

**Ferne klingt diffuser, weicher, leicht gefiltert:** Mehr Raumanteil, längeres Pre-Delay, sanfte Hochfrequenz-Dämpfung, breitere Phantomquelle.

Die **Z‑Achse definiert daher ein konsistentes Vokabular** der räumlichen Gestaltung statt einer einzigen Kennzahl. Alle Tiefenparameter ändern sich koordiniert und musikalisch sinnvoll.

**Parametermapping für Z-Werte (0=nah, 1=fern):**
• Direct/Reverb-Verhältnis: 100%/0% → 20%/80%
• Pre-Delay: 0ms → 50ms
• Hochfrequenz-Dämpfung: 0dB → -6dB
• Quellenbreite: schmal → breit

Diese Zuordnungen sind kalibrierbar und können je nach musikalischem Kontext angepasst werden.

---

## Seite 13 – Layer-Architektur (Archetypen)

**Layer sind fest definierte Raumcharaktere**, die gleichzeitig bestehen und in Anteilen gemischt werden.

**Empfohlenes Set:**

• **Kino:** breit, offen, mit füllender Diffusion. Große Nachhallzeiten, breite Stereoabbildung, warmer Klangcharakter.

• **Proberaum:** trocken, kompakt, kurze Nachhallzeiten. Direkte Akustik, präzise Ortung, kontrollierte Dynamik.

• **Nebenraum:** gedämpft, durch Wände gefiltert, frühe Reflexionen dominieren. Intime Atmosphäre, gedämpfte Höhen, natürliche Begrenzung.

• **Intim:** sehr nah, minimaler Raumanteil, Fokus auf Direktheit. Maximale Präsenz, keine Verfremdung, unvermittelte Nähe.

Die **exakte akustische Bedeutung der Layer ist dokumentiert**, damit ihr Mischverhalten vorhersagbar bleibt. Jeder Archetyp wird durch Referenz-Impulsantworten, Frequenzgänge und dynamische Eigenschaften charakterisiert.

**Austauschbarkeit:** Die vier Standard-Archetypen können durch projektspezifische Räume ersetzt werden, solange die Mischlogik und Bedienschnittstelle unverändert bleiben.

---

## Seite 14 – Mischlogik Quadrat (4 Ecken)

Das **Quadrat bildet die Anteile der vier Archetypen ab**. Die Position innerhalb des Quadrats bestimmt die Gewichte, deren Summe konstant bleibt.

**Mathematische Umsetzung:** Bilineare Interpolation zwischen den vier Eckpunkten. Die Position (x,y) im normalisierten Quadrat (0,0) bis (1,1) erzeugt vier Gewichte w1, w2, w3, w4 mit w1+w2+w3+w4=1.

**Eckzuordnung (Beispiel):**
• Oben links: Kino
• Oben rechts: Proberaum
• Unten links: Nebenraum
• Unten rechts: Intim

Die **Hörwirkung wird regelmäßig gegen Referenzabhören geprüft**, damit die Anteile musikalisch sinnvoll bleiben: keine Überbetonung in Ecken, keine Dellen in der Mitte.

**Bewegungen über das Quadrat** werden zeitlich geglättet, um abrupte Raumsprünge zu vermeiden. Die Glättungskonstante ist einstellbar und wird mit der Szene gespeichert.

**Edge-Cases:** Randpositionen und Ecken werden besonders sorgfältig behandelt, da hier oft extreme Mischungsverhältnisse entstehen können.

---

## Seite 15 – Mischlogik Dreieck (3 Archetypen)

Für Szenen mit **drei prägnanten Charakteren** bietet das Dreieck eine besonders stabile Mischung.

Der **Punkt innerhalb des Dreiecks** erzeugt drei Gewichte, die stets zusammen 100 Prozent ergeben. Baryzentrische Koordinaten garantieren mathematisch korrekte Gewichtsverteilung ohne Singularitäten.

Dieses Modell eignet sich für **dramaturgische Achsen** wie:
• nah–mittel–weit
• trocken–neutral–diffus
• intim–neutral–episch

**Vorteile gegenüber dem Quadrat:**
• Keine "unmöglichen" Kombinationen durch reduzierte Dimensionalität
• Musikalisch intuitivere Übergänge
• Stabilere Mischungen ohne Überkompensation

**Typische Dreieckszuordnung:**
• Spitze oben: Ferne (diffus, Hall)
• Unten links: Nähe (trocken, direkt)
• Unten rechts: Breite (Stereo, offen)

Die Auswahl zwischen Quadrat- und Dreiecksmischung erfolgt projektspezifisch und kann innerhalb einer Session gewechselt werden.

---

## Seite 16 – Mehrspur-Objekte & Phasenbeziehungen

**Mehrere identische Objekte** dürfen gleichzeitig dieselbe Bahn befahren. Unterschiede entstehen durch Phasenversatz, Solo/Mute und individuelle Tiefenparameter.

Die Gestaltung folgt **zwei Zielen**: räumliche Fülle ohne Unschärfe und lebendige Bewegung ohne Chaos.

**Phasenversatz** wird maßvoll eingesetzt, um Schwebungen und Doppellungen bewusst zu modellieren:
• Kleine Versätze (1-10ms): Verbreiterung, Choreffekt
• Mittlere Versätze (10-50ms): Doppelecho, rhythmische Struktur
• Große Versätze (>50ms): Unabhängige Wiederholungen

**Solo/Mute-Funktionalität** ermöglicht:
• Isolierte Abhöre einzelner Spuren
• Reduktion der Komplexität in kritischen Passagen
• Schrittweise Aufbau dichter Texturen

**Individuelle Tiefenparameter** pro Spur erlauben:
• Gestaffelte Tiefenwirkung
• Vordergrund/Hintergrund-Effekte
• Räumliche Schichtung ohne Maskierung

Die Anzahl gleichzeitiger Mehrspurobjekte ist auf vier begrenzt, um CPU-Last und akustische Komplexität kontrollierbar zu halten.

---

## Seite 17 – Psychoakustische Leitplanken

**Richtungswahrnehmung** wird durch Laufzeitunterschiede, spektrale Hinweise und frühe Reflexionen geprägt.

Die **Illusion profitiert von korrekter Relation** dieser Faktoren:
• Interaurale Zeitdifferenzen (ITD): <1ms für seitliche Ortung
• Interaurale Pegeldifferenzen (ILD): bis 20dB für Höhenwahrnehmung
• Spektrale Formung: HRTF-Charakteristika für Elevation
• Frühe Reflexionen: <50ms für Raumgröße, >50ms für Nachhall

Bei **Lautsprechern** gilt: Übersprechen zwischen Kanälen darf nicht unkontrolliert sein; bei **Kopfhörern** ist die Konsistenz der virtuellen Kopfbezugsfunktion entscheidend.

**Alle Parameter sind so gewählt**, dass natürliche Erwartungen des Ohrs erfüllt und nicht überreizt werden:
• Keine extremen Panorama-Sprünge (>60°/Sekunde)
• Graduelle Tiefenänderungen (Z-Geschwindigkeit <0.5/Sekunde)
• Frequenzmodulation bleibt unter der Hörschwelle
• Dynamiksprünge werden sanft begrenzt

**Grenzen der Wahrnehmung** werden respektiert: Das System übertreibt keine Effekte, sondern arbeitet im Bereich natürlicher räumlicher Erfahrung.

---

## Seite 18 – Renderer-Pipeline (ohne Implementationsdetails)

Der **Renderer folgt einer festen Reihenfolge**: optionales Upmix/Decoding, räumliche Faltungen für Richtung und Raum, gezielte Tiefensteuerungen, gegebenenfalls Übersprechreduktion bei Lautsprechern, lineare Raumkorrektur und abschließende Sicherheit durch begrenzende Stufen.

**Wichtig ist die klare Trennung** zwischen kreativer Steuerung und technischer Korrektur. Künstlerische Entscheidungen werden vor der Korrektur getroffen; Korrektur stabilisiert nur das Ergebnis im realen Raum.

**Pipeline-Stufen im Detail:**

1. **Input-Routing:** Zuordnung der Audiosignale zu virtuellen Objekten
2. **Spatial-Processing:** Position, Bewegung, Layer-Mischung
3. **Depth-Processing:** Z-Achsen-Parameter, Nah/Fern-Charakteristik
4. **Room-Simulation:** HRTF/BRIR-Faltung, Archetyp-Mischung
5. **Crosstalk-Handling:** Optional für Lautsprecher-Betrieb
6. **Room-Correction:** Lineare Entzerrung des realen Wiedergabesystems
7. **Safety-Limiting:** Schutz vor Übersteuerung und Spitzen

Jede Stufe kann einzeln diagnostiziert und optimiert werden, ohne andere Bereiche zu beeinflussen.

---

## Seite 19 – Lautsprecher vs. Kopfhörer

**Kopfhörer** bieten maximale Kontrolle über die Illusion, da der Raum der Hörer:in ausgeschlossen ist.

**Vorteile Kopfhörer:**
• Perfekte Kanaltrennung, kein Übersprechen
• Kontrollierte akustische Übertragung
• Unabhängigkeit von Raumakustik
• Präzise HRTF-Anwendung möglich

**Lautsprecher** benötigen zusätzliche Sorgfalt: Raumkorrektur, definierte Hörzone, optionale Übersprechkontrolle und pegelbewusste Tiefenverteilung.

**Herausforderungen Lautsprecher:**
• Übersprechen zwischen Kanälen
• Raumakustische Verfälschungen
• Positionsabhängige Wahrnehmung
• Größenbeschränkungen kleiner Boxen

**Beide Wege sind gültig**; die Wahl richtet sich nach Ziel und Ort der Vorführung:
• Studio/Produktion: meist Kopfhörer
• Live-Performance: meist Lautsprecher
• Installation: ausschließlich Lautsprecher
• Forschung: beide Modi parallel

Die Systemkonfiguration passt sich automatisch an die gewählte Ausgabeart an.

---

## Seite 20 – Kalibrierung & Raumkorrektur

Vor der kreativen Arbeit steht die **Messung des realen Wiedergaberaums**. Aus den Messungen werden lineare Korrekturen und Referenzpegel abgeleitet.

**Messprozedur:**
• Impulsantwort-Messung an der Haupthörposition
• Optional: Mehrpunkt-Messung für erweiterte Hörzone
• Analyse von Frequenzgang, Nachhallzeit, frühen Reflexionen
• Bestimmung optimaler Pegel und Dynamikreserven

**Kleine Lautsprecher** profitieren stark von sorgfältiger Entzerrung und einer klaren Übergabe an etwaige Subwoofer:
• Hochpass-Filterung entlastet kleine Treiber
• Bassmanagement erhält Klarheit im Mittenbereich
• Dynamikkompression verhindert Übersteuerung

**Nach der Korrektur** werden die künstlerischen Layer- und Tiefenentscheidungen übertragen und kontrolliert wiedergegeben.

**Kalibrierdaten** werden mit der Session gespeichert und können auf ähnliche Räume übertragen werden.

**Verifikation:** Referenz-Testsignale bestätigen die korrekte Funktion der Kalibrierkette.

---

## Seite 21 – Übersprechkontrolle bei Lautsprechern

Die **Unterdrückung des Übersprechens** zwischen linkem und rechtem Kanal kann die Ortung verbessern, ist jedoch empfindlich gegenüber Kopfbewegungen und Sitzposition.

**Crosstalk-Cancellation (XTC) funktioniert durch:**
• Antisignale, die das Übersprechen kompensieren
• Präzise Laufzeit- und Pegelanpassung
• Frequenzabhängige Korrektur

**Einsatz nur in Szenen**, die davon profitieren, und in Räumen, die ruhige Hörpositionen erlauben:
• Kritische Ortungsaufgaben
• Maximale Räumlichkeit bei Stereo-Setup
• Wissenschaftliche Hörtests

**Stets mit moderater Intensität starten** und hörend optimieren:
• Zu starke XTC erzeugt Verfärbungen
• Kopfbewegungen zerstören die Illusion
• Frequenzbereich begrenzen (meist 200Hz-8kHz)

**Alternative:** Head-Tracking kann XTC dynamisch anpassen, erfordert aber zusätzliche Hardware.

**Best Practice:** XTC als zuschaltbare Option, nicht als Standard.

---

## Seite 22 – Pegel, Dynamik & Sicherheit

Die **Kette hält ausreichend Reserven vor**. Raum- und Tiefenprozesse können Spitzen erzeugen; deshalb werden Headroom-Regeln eingehalten und eine begrenzende Stufe am Ende sichert gegen Übersteuerungen.

**Headroom-Budget:**
• Input: -12dB Reserve für Peaks
• Spatial Processing: -6dB zusätzliche Reserve
• Layer-Mixing: -3dB für Summenbildung
• Output: -3dB finale Reserve vor Limiter

**Dynamik bleibt musikalisch**; Schutz greift unauffällig:
• Transparente Limiter-Algorithmen
• Multiband-Processing für spectrale Kontrolle
• Lookahead verhindert abrupte Eingriffe
• Soft-Knee-Charakteristiken

**Ziel ist eine stressfreie, ermüdungsarme Langzeithörbarkeit** bei gleichzeitiger dramaturgischer Wirkung:
• Keine Hörermüdung durch Übersteuerung
• Erhaltung der musikalischen Dynamik
• Schutz der Wiedergabesysteme
• Komfort für lange Sessions

**Monitoring:** Kontinuierliche Überwachung von RMS- und Peak-Werten mit visueller Warnung bei kritischen Zuständen.

---

## Seite 23 – Echtzeit & Latenzmanagement

Die **Bedienung fühlt sich direkt an**, wenn Verarbeitungsverzögerungen klein und stabil sind.

**Latenz-Quellen:**
• Audio-Interface: 2-10ms (pufferabhängig)
• Spatial-Processing: 5-20ms (faltungsabhängig)
• Room-Correction: 10-50ms (FIR-längenabhängig)
• Output-Processing: 1-5ms (limiter/safety)

Die **Systemkonfiguration wird so gewählt**, dass Bewegungen, Morphings und externe Steuerungen ohne spürbare Verzögerung wirksam werden:
• Gesamtlatenz <50ms für Live-Performance
• Stabile Latenz wichtiger als minimale Latenz
• Puffergröße an Systemleistung angepasst

**Puffergrößen und Prozesslängen** sind bewusst auf die Anforderungen des Auftritts oder der Produktion abgestimmt:
• Studio: Qualität vor Latenz (128-512 Samples)
• Live: Latenz vor Qualität (32-128 Samples)
• Installation: Stabilität vor beiden (256-1024 Samples)

**Latenz-Kompensation:** Alle Pfade durch das System haben identische Gesamtlatenz.

---

## Seite 24 – Persistenz & Zustandsverwaltung

**Alle kreativen Einstellungen** werden als geordnete Zustände vorgehalten: Szenen, Presets, Snapshots.

**Zwischenspeicherungen** verhindern Verlust, geteilte Zustände ermöglichen Zusammenarbeit und Wiederaufführung:
• Automatische Backups alle 30 Sekunden
• Versionierte Speicherung mit Undo/Redo
• Export/Import für Austausch zwischen Systemen
• Cloud-Sync für verteilte Teams

**Beim Laden eines Zustands** werden nur definierte Parameter überschrieben; laufende Aufführungen bleiben kontrollierbar:
• Sanfte Überblendung zwischen Zuständen
• Selektives Laden einzelner Parameter-Gruppen
• Morphing-Funktion für kontinuierliche Übergänge
• Lock-Funktion für kritische Parameter

**Datenformat:** JSON-basiert, menschenlesbar, versioniert, validiert.

**Metadaten:** Jeder Zustand trägt Zeitstempel, Autor, Versionsnummer, Beschreibung und Abhängigkeiten.

**Sicherheit:** Backup-Strategien und Wiederherstellungsverfahren für kritische Produktionen.

---

## Seite 25 – Externe Steuerung: Prinzipien

**Kontinuierliche Steuergrößen** – etwa horizontale/vertikale Positionen, Tiefenwerte, Morphanteile – sind auf robuste, kurze Adressräume abgebildet.

**OSC-Adressierung (Beispiele):**
• /mothership/object/1/x [0.0-1.0]
• /mothership/object/1/y [0.0-1.0]
• /mothership/object/1/z [0.0-1.0]
• /mothership/layer/xy/x [0.0-1.0]
• /mothership/layer/xy/y [0.0-1.0]
• /mothership/morph/ab [0.0-1.0]

**Eingehende Werte werden sanft geglättet**, damit Handbewegungen natürlich wirken und digitales Zittern vermieden wird:
• Glättungskonstante einstellbar (1-100ms)
• Exponential-Filter für organische Reaktion
• Begrenzte Geschwindigkeit verhindert Sprünge

**Die Bandbreite der Aktualisierungen** wird so bemessen, dass das Gesamtsystem stabil bleibt, auch im langen Betrieb:
• Max. 100 Updates/Sekunde pro Parameter
• Intelligente Datenreduktion bei wenig Änderung
• Prioritätssystem für kritische Parameter

**MIDI-Integration:** Standard Control-Change-Mapping mit 14-Bit-Auflösung für präzise Steuerung.

---

## Seite 26 – Plattformbetrieb (neutral)

Unabhängig vom Betriebssystem gilt:

• **Sichere Audio-Routing-Pfade**, klar dokumentiert:
  - Definierte Ein- und Ausgänge
  - Latenz-optimierte Treiber
  - Fallback-Optionen bei Problemen

• **Stabiler Renderer** mit definierter Reihenfolge der Verarbeitungsschritte:
  - Deterministische Ausführung
  - Reproduzierbare Ergebnisse
  - Graceful Degradation bei Überlast

• **Reproduzierbare Start- und Shutdown-Sequenzen**, inklusive Rückfallstrategien:
  - Automatische Hardwareerkennung
  - Sichere Initialisierung aller Komponenten
  - Sauberes Herunterfahren ohne Pops/Clicks

**Die konkrete Tool-Auswahl ist frei**, sofern die hier festgelegten Funktionen und Reihenfolgen eingehalten werden:
• Windows: ASIO, WASAPI, virtuelle Routing-Tools
• macOS: Core Audio, Audio Units, virtuelle Devices
• Linux: JACK, PipeWire, ALSA

**Cross-Platform-Aspekte:** Dateiformate, Kalibrierungsdaten und Presets sind plattformübergreifend kompatibel.

---

## Seite 27 – Qualitätssicherung

**Regelmäßige Hörproben** mit Referenzmaterial und Standardszenen sichern Konsistenz:
• Tägliche Tests mit Referenz-Content
• A/B-Vergleiche mit bewährten Systemen
• Langzeit-Tests für Ermüdungsfreiheit
• Multi-Person-Evaluierung für Objektivität

**Automatisierte Prüfungen** der Zustandsdateien stellen sicher, dass alle notwendigen Parameter vorhanden und innerhalb gültiger Grenzen sind:
• JSON-Schema-Validierung
• Wertebereich-Prüfung
• Konsistenz-Checks zwischen Parametern
• Warnungen bei kritischen Konfigurationen

**Vergleichshörplätze** (Hauptplatz, Nebenplatz) helfen, robuste Einstellungen für reale Räume zu finden:
• Sweet-Spot vs. erweiterte Hörzone
• Verschiedene Kopfhörer-Modelle
• Unterschiedliche Lautsprecher-Setups
• Mobile vs. stationäre Systeme

**Dokumentierte QS-Prozeduren** mit Checklisten und Protokollen für jede Produktion.

**Continuous Integration:** Automatisierte Tests bei Code-Updates.

---

## Seite 28 – Testplan

**Zentrale Prüfziele:**

• **Gleichmäßige Bewegung** entlang der Bahn; Zwischenzeitpunkt teilt nachvollziehbar:
  - Konstante Geschwindigkeit ohne Stolpern
  - Korrekte Zeitaufteilung zwischen Segmenten
  - Geometrische Präzision der Kurven

• **Easing verändert nur die Tempokurve**, nie die geometrische Bahn:
  - Identische Start- und Endpunkte
  - Unveränderte Kurvenform
  - Plausible Geschwindigkeitsverteilung

• **Mischungen bleiben pegelstetig**; keine Sprünge in Lautheit oder Spektrum:
  - Konstante Summenbildung bei Layer-Mischung
  - Gleichmäßige Übergänge im XY-Pad/Dreieck
  - Keine spektralen Verfärbungen

• **Mehrspurverhalten bleibt geordnet**; Phasenversätze erzeugen beabsichtigte Effekte:
  - Saubere Trennung zwischen Spuren
  - Korrekte Phase-Relationships
  - Solo/Mute funktioniert zuverlässig

• **Zustandswechsel und Morphing sind klickfrei**:
  - Glatte Übergänge zwischen Presets
  - Keine Pops oder Knackser
  - Kontinuierliche Parameter-Interpolation

• **Externe Steuerung wirkt ohne Ruckeln** und ohne Überlastung des Systems:
  - Responsive aber nicht zittrige Reaktion
  - Stabile Kommunikation auch bei hoher Datenrate
  - Graceful Handling von Verbindungsfehlern

---

## Seite 29 – Abnahmekriterien

Eine **Szene gilt als abgenommen**, wenn:

• **Die Raumillusion auf der Zielwiedergabe stabil ist**:
  - Konsistente Ortung über den gesamten Frequenzbereich
  - Plausible Tiefenstaffelung ohne Artefakte
  - Stabile Wahrnehmung auch bei Kopfbewegungen (bei Lautsprechern)

• **Die dokumentierten Layer-Charaktere identifizierbar und mischbar bleiben**:
  - Jeder Archetyp klingt charakteristisch
  - Übergänge zwischen Layern sind musikalisch
  - Extreme Positionen bleiben verwendbar

• **Bewegungen dramaturgisch schlüssig und technisch sauber sind**:
  - Natürliche Bewegungsverläufe ohne Artefakte
  - Dramaturgisch sinnvolle Geschwindigkeiten
  - Reproduzierbare Automation

• **Pegelreserven und Sicherheit nachweislich eingehalten werden**:
  - Keine Übersteuerungen bei normalem Betrieb
  - Ausreichend Headroom für Spitzen
  - Limiter greift nur in Ausnahmesituationen

• **Zustände geladen und reproduzierbar wiedergegeben werden können**:
  - Identische Wiedergabe nach Neustart
  - Korrekte Parameter-Wiederherstellung
  - Plattformübergreifende Kompatibilität

**Formales Abnahmeverfahren:** Testprotokoll mit definierten Kriterien und Unterschriften.

---

## Seite 30 – Roadmap kurz/mittel/lang

**Kurzfristig (3-6 Monate):**
• Zusätzliche Referenzpunkte pro Bahn (5-7 statt 3)
• Feinere Zeitmarken mit Beat-Grid-Synchronisation
• Intuitivere Tiefenparameter mit visueller Rückmeldung
• Erweiterte Preset-Bibliothek mit Genres-spezifischen Templates

**Mittelfristig (6-18 Monate):**
• Kopfbewegungsbezug für Lautsprecher-Betrieb via Head-Tracking
• Kuratierte Raumantwort-Bibliothek mit gemessenen Locations
• Erweiterte Mehrspurverwaltung (bis zu 8 Spuren pro Objekt)
• Machine Learning für automatische Kalibrierung

**Langfristig (1-3 Jahre):**
• Adaptive Illusionen, die sich an Hörplatz und Material anpassen
• Austauschformate für eine interoperable Klangregie
• Echtzeit-Kollaboration zwischen mehreren Nutzer:innen
• VR/AR-Integration für visuelle Raumkomposition
• Künstliche Intelligenz für automatische Raum-Choreographie

**Forschungsfelder:**
• Psychoakustische Optimierung für verschiedene Altersgruppen
• Barrierefreie Bedienung für Menschen mit Beeinträchtigungen
• Integration in immersive Medienformate (360°-Audio, Holophonie)

---

## Seite 31 – Governance & Ethik

Die **Gestaltung räumlicher Illusionen beeinflusst Wahrnehmung und Aufmerksamkeit**. Verantwortungsvoll eingesetzt, stärken sie Fokus und Erlebnis; übertrieben eingesetzt, überfordern sie.

**Das System folgt dem Prinzip der Transparenz**:
• Dokumentierte Entscheidungen in allen Gestaltungsschritten
• Reproduzierbare Ergebnisse durch versionierte Parameter
• Schutz der Hörenden vor schädlicher Lautheit oder Erschöpfung

**Ethische Leitlinien:**
• Keine manipulative Nutzung räumlicher Effekte
• Respekt vor natürlichen Hörgewohnheiten
• Transparenz gegenüber dem Publikum über technische Mittel
• Schutz vor Hörschäden durch konsequente Limiter

**Barrierefreiheit:**
• Unterstützung verschiedener Hörfähigkeiten
• Alternative Bedienkonzepte für motorische Einschränkungen
• Visuelle Hilfsmittel für Gehörlose
• Anpassbare Intensität für Überempfindlichkeiten

**Datenschutz:** Keine Speicherung persönlicher Hördaten ohne explizite Zustimmung.

**Open Standards:** Förderung offener Austauschformate für Herstellerunabhängigkeit.

---

## Seite 32 – Betrieb & Wartung

**Wartung umfasst regelmäßige Überprüfung** der Kalibrierung, der Zustände und der Abhörkette:

**Tägliche Wartung:**
• Funktionstest aller Ein-/Ausgänge
• Prüfung der Referenzpegel
• Backup-Status überprüfen
• Log-Files auf Anomalien prüfen

**Wöchentliche Wartung:**
• Vollständige Kalibrierungsprüfung
• Test aller Presets und Szenen
• Hardware-Diagnostik
• Software-Update-Check

**Monatliche Wartung:**
• Tiefere Systemanalyse
• Archivierung alter Sessions
• Reinigung der Hardware
• Dokumentations-Updates

**Updates an Teilkomponenten** erfolgen nachvollziehbar:
• Vorher Tests an Standardszenen
• Staged Deployment mit Rollback-Option
• Nachher Freigabe mit Protokoll
• Benutzerschulung bei Interface-Änderungen

**Betriebsunterbrechungen** sind geplant und kommuniziert:
• Wartungsfenster außerhalb der Produktionszeiten
• Notfall-Prozeduren für kritische Ausfälle
• Ersatzsysteme für wichtige Veranstaltungen

**Rollback-Optionen** bestehen für kritische Veranstaltungen.

---

## Seite 33 – Risiken & Glossar

**Risiken:**

• **Übersteuerung durch ungünstige Layer-Kombinationen**: Gegenmaßnahme durch konservative Headroom-Regeln und intelligente Limiter mit Lookahead.

• **Räumliche Instabilität bei starkem Übersprech**: Vermeidung durch sorgfältige Crosstalk-Cancellation und Positionsstabilisierung.

• **Ermüdung durch übertriebene Diffusion**: Prävention durch regelmäßige Hörpausen, Referenz-Checks und moderate Effektdosierung.

• **Systeminkompatibilität**: Minimierung durch standardisierte Schnittstellen und umfassende Kompatibilitätstests.

• **Datenverlust**: Schutz durch automatische Backups, redundante Speicherung und Cloud-Synchronisation.

**Glossar:**

• **Archetyp:** Fest definierter Raumcharakter mit dokumentierten akustischen Eigenschaften
• **Szene:** Vollständiger Klangzustand mit allen Objekten, Layern und Parametern
• **Pfad:** Geglättete Bahn einer Bewegung zwischen definierten Referenzpunkten
• **Morphing:** Stufenlose Überblendung zweier kompletter Szenen-Zustände
• **Tiefe:** Wahrgenommene Nähe/Ferne als koordinierte Variation mehrerer Parameter
• **Easing:** Zeitkurven-Formung für natürliche Bewegungsverläufe
• **Layer:** Mischbarer Raumcharakter aus der Archetypen-Bibliothek
• **Quantisierung:** Einrasten auf definierte Raster für Reproduzierbarkeit
• **XTC:** Crosstalk-Cancellation zur Übersprechunterdrückung bei Lautsprechern
• **HRTF:** Head-Related Transfer Function für binaurale Richtungswahrnehmung

**Abschluss:** MOTHERSHIP schafft eine gestaltbare, wiederholbare Raumpoesie, die aus Technik und Wahrnehmung ein zuverlässiges Handwerk formt. Die Synthese aus psychoakustischer Präzision und künstlerischer Intuition ermöglicht neue Dimensionen räumlicher Klanggestaltung, die sowohl technisch fundiert als auch musikalisch inspirierend sind.

---

*Ende des 33-seitigen Blueprints*

**Dokument-Metadaten:**
- Gesamtumfang: 33 Seiten
- Wortanzahl: ca. 8.500 Wörter
- Format: Reine Textform ohne Code oder Grafiken
- Version: 1.0
- Datum: 05.09.2025
- Status: Produktionsreif für Konzeption, Abstimmung und Abnahme