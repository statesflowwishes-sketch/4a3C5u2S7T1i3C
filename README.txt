Lege hier deine Impulsantworten (IR/BRIR) ab. Beispiel: brir_cinema_wide.wav
MOTHERSHIP – Technische Beschreibung (rein textlich)


MOTHERSHIP – Blueprint (33 Seiten, rein textlich)
Synthtographie für Raumklang: sauber beschrieben, ohne Code und ohne Grafiken.
Stand: 05.09.2025 • Schriftgröße: 13 pt • Format: A4 • Autor: MOTHERSHIP

Dieses Dokument ist die geordnete Grundlage für Konzeption, Abstimmung und Abnahme. Es fasst das System sauber zusammen – ausschließlich in Textform – und dient als Referenz für Kreativteams, Technik und Betrieb.
Seite 1 – Vision & Zweck

MOTHERSHIP ist eine technische Regie für dreidimensionale Klangräume. Ziel ist es, reale Lautsprecher – auch sehr kleine – sowie Kopfhörer als Fenster in einen virtuell berechneten Raum zu nutzen. Das System erzeugt Räumlichkeit, Bewegung und Tiefenstaffelung vollständig in der Software und legt sie als akustische Schichten über den vorhandenen physikalischen Raum. Dadurch wird eine glaubhafte Raumillusion geschaffen, die reproduzierbar, skalierbar und live steuerbar ist.

Die Synthtographie-Idee: Klang wird nicht nur gemischt, sondern als Raumplastik komponiert – mit Layern, Pfaden und Zuständen, die sich dramaturgisch entwicklen. Das Dokument beschreibt präzise, wie diese Bausteine zusammenspielen: ohne Programmiercodes, ohne Grafiken – rein technisch-konzeptionell, sauber strukturiert.
Seite 2 – Begriffe & Scope

Szene: Ein vollständiger, in sich stimmiger Klangzustand aus Objekten, Layern, Tiefenparametern und globalen Einstellungen.

Objekt: Eine virtuelle Klangquelle, die Position, Bewegung und Layerbezüge besitzt.

Layer: Ein Raum-Archetyp, dessen Anteile sich mischen lassen (z. B. Kino, Proberaum, Nebenraum, Intim).

Pfad: Eine geglättete Bahn, entlang der sich ein Objekt zwischen Referenzpunkten bewegt.

Morphing: Überblendung zwischen zwei kompletten Szenenzuständen.

Renderer: Die DSP-Pipeline, die aus Steuerdaten hörbare Ergebnisse erzeugt.

Scope: Dieses Blueprint definiert Anforderungen, Abläufe und Qualitätskriterien; Implementierungsdetails bleiben austauschbar, solange Schnittstellen und Resultate eingehalten werden.
Seite 3 – Nutzerrollen & Use-Cases

Rollen: Klangregisseur:in (künstlerische Gestaltung), Systemoperator (Kalibrierung/Betrieb), Technische Leitung (Qualität/Sicherheit).

Use-Cases:

• Studio: Dramaturgische Positionierung und Tiefenstaffelung einzelner Spuren.

• Bühne: Live-Morphing zwischen Stimmungen und Räumen.

• Installation: Dauerbetrieb mit stabiler Illusion trotz kleiner Lautsprecher.

• Forschung/Prototyping: Evaluation psychoakustischer Parameter und Hörfelder.
Seite 4 – Systemübersicht

Die Architektur trennt strikt zwischen Steuerung und Rendering.

Die Steuerung erzeugt ausschließlich Parameter: Objektpositionen, Layergewichte, Tiefenanteile, Zeitfunktionen und Zustandswechsel.

Der Renderer setzt diese Parameter in Klang um: Upmix/Decoding (falls nötig), räumliche Faltungen, Tiefenverteilung, eventuelle Übersprech-Unterdrückung bei Lautsprechern, Korrektur des realen Raums und abschließende Schutzmaßnahmen.

Die Trennung gewährleistet Austauschbarkeit der Audio-Engine ohne Änderung der kreativen Steuerlogik.
Seite 5 – Szenenmodell (High Level)

Eine Szene umfasst:

• Objekte mit Raumkoordinaten und Pfadzuweisung.

• Layer-Mischungen in zwei Formen: vier Eck-Archetypen (Quadrat) und drei Archetypen (Dreieck).

• Globale Tiefenparameter (Nähe/Ferne-Charakteristika).

• Zeitlogik: Keyframe-Timeline, Bewegungs-Easing, optionale Quantisierung.

• Zustandsverwaltung: Presets, Morphing zwischen A und B, sowie Persistenz im Betrieb.

• Externe Kontrolle über standardisierte, robuste Steuergrößen.
Seite 6 – Raumkoordinaten & Bezugsrahmen

Der Betriebsraum wird als normierter Bereich gedacht. Die Breite beschreibt eine linke–rechte Ausdehnung, die Höhe eine vertikale Staffelung, und die Tiefe eine wahrgenommene Nähe oder Ferne.

Wesentlich ist die relative Konsistenz: Alle Positionen, Bewegungen und Layer-Mischungen beziehen sich auf denselben normierten Rahmen. Dies sichert Übertragbarkeit zwischen Räumen unterschiedlicher Größe und akustischer Beschaffenheit.
Seite 7 – Pfad-Automation: Referenzpunkte

Die Bahn eines Objekts wird über Referenzpunkte A, B, C definiert, zwischen denen eine geglättete Verbindung entsteht.

Die Glättung ist so zu wählen, dass keine abrupten Richtungswechsel hörbar werden, aber die dramaturgische Intention (z. B. Bogen, S-Kurve, sanfter Umweg) erhalten bleibt.

Die Referenzpunkte werden nachvollziehbar dokumentiert (Position, Funktion im Verlauf, dramaturgische Rolle).
Seite 8 – Pfad-Automation: Bewegung & Konstanz

Die wahrgenommene Geschwindigkeit entlang der Bahn soll gleichmäßig sein, sofern dies gefordert ist. Dafür orientiert sich die Zeitauflösung an der effektiven Bogenlänge statt an bloßen Punktindizes.

Der Vorteil: Hörende erleben eine fließende Bewegung ohne „Stolperer“, unabhängig von der Lage der Referenzpunkte.
Seite 9 – Keyframe-Timing & Zwischenzeiten

Die Zeitachse wird durch drei Markierungen strukturiert: Start (A=0), Zwischenzeitpunkt (B), Endzeit (C=1).

Der Zwischenzeitpunkt teilt die Gesamtbewegung in zwei Teilabschnitte, deren Dauerverhältnis gezielt festgelegt wird.

Dies erlaubt dramaturgisch sinnvolle Beschleunigungen oder Dehnungen bestimmter Teile der Bewegung, ohne die Bahngeometrie zu verändern.
Seite 10 – Easing der Zeitfunktion

Easing formt die Tempokurve einer Bewegung: sanftes Anfahren, fließendes Durchlaufen, weiches Ausrollen.

Die Bahn bleibt geometrisch unverändert, nur die Verteilung der Zeit über die Strecke ändert sich.

Die Auswahl der Kurven erfolgt nach Hördynamik: unmerklich, musikalisch und wiederholbar.

Alle verwendeten Kurven werden benannt und beschrieben, damit spätere Rekonstruktion möglich ist.
Seite 11 – Quantisierung & Reproduzierbarkeit

Zwei Arten von Quantisierung erhöhen Konsistenz:

• Positionsraster: Referenzpunkte fangen auf definierte Rastersprünge ein, um wiederholbare Einstellungen zu sichern.

• Zeitraster: Normalisierte Zeitpunkte werden auf feste Schritte gesetzt, damit Bewegungen exakt reproduzierbar sind.

Quantisierung ist stets abschaltbar; künstlerische Freiheit geht vor, wenn notwendig.
Seite 12 – Z‑Achse: Nähe und Ferne

Die Tiefe wird nicht als bloßer Pegel interpretiert, sondern als koordinierte Variation mehrerer Parameter: Anteil des Raumanteils, Pre-Delay, Filterungen der hohen und tiefen Frequenzen, Kompaktheit der frühen Reflexionen.

Nähe klingt präsent, klar, trocken. Ferne klingt diffuser, weicher, leicht gefiltert.

Die Z‑Achse definiert daher ein konsistentes Vokabular der räumlichen Gestaltung statt einer einzigen Kennzahl.
Seite 13 – Layer-Architektur (Archetypen)

Layer sind fest definierte Raumcharaktere, die gleichzeitig bestehen und in Anteilen gemischt werden.

Empfohlenes Set:

• Kino: breit, offen, mit füllender Diffusion.

• Proberaum: trocken, kompakt, kurze Nachhallzeiten.

• Nebenraum: gedämpft, durch Wände gefiltert, frühe Reflexionen dominieren.

• Intim: sehr nah, minimaler Raumanteil, Fokus auf Direktheit.

Die exakte akustische Bedeutung der Layer ist dokumentiert, damit ihr Mischverhalten vorhersagbar bleibt.
Seite 14 – Mischlogik Quadrat (4 Ecken)

Das Quadrat bildet die Anteile der vier Archetypen ab. Die Position innerhalb des Quadrats bestimmt die Gewichte, deren Summe konstant bleibt.

Die Hörwirkung wird regelmäßig gegen Referenzabhören geprüft, damit die Anteile musikalisch sinnvoll bleiben: keine Überbetonung in Ecken, keine Dellen in der Mitte.
Seite 15 – Mischlogik Dreieck (3 Archetypen)

Für Szenen mit drei prägnanten Charakteren bietet das Dreieck eine besonders stabile Mischung.

Der Punkt innerhalb des Dreiecks erzeugt drei Gewichte, die stets zusammen 100 Prozent ergeben.

Dieses Modell eignet sich für dramaturgische Achsen wie nah–mittel–weit oder trocken–neutral–diffus.
Seite 16 – Mehrspur-Objekte & Phasenbeziehungen

Mehrere identische Objekte dürfen gleichzeitig dieselbe Bahn befahren. Unterschiede entstehen durch Phasenversatz, Solo/Mute und individuelle Tiefenparameter.

Die Gestaltung folgt zwei Zielen: räumliche Fülle ohne Unschärfe und lebendige Bewegung ohne Chaos.

Phasenversatz wird maßvoll eingesetzt, um Schwebungen und Doppellungen bewusst zu modellieren.
Seite 17 – Psychoakustische Leitplanken

Richtungswahrnehmung wird durch Laufzeitunterschiede, spektrale Hinweise und frühe Reflexionen geprägt.

Die Illusion profitiert von korrekter Relation dieser Faktoren.

Bei Lautsprechern gilt: Übersprechen zwischen Kanälen darf nicht unkontrolliert sein; bei Kopfhörern ist die Konsistenz der virtuellen Kopfbezugsfunktion entscheidend.

Alle Parameter sind so gewählt, dass natürliche Erwartungen des Ohrs erfüllt und nicht überreizt werden.
Seite 18 – Renderer-Pipeline (ohne Implementationsdetails)

Der Renderer folgt einer festen Reihenfolge: optionales Upmix/Decoding, räumliche Faltungen für Richtung und Raum, gezielte Tiefensteuerungen, gegebenenfalls Übersprechreduktion bei Lautsprechern, lineare Raumkorrektur und abschließende Sicherheit durch begrenzende Stufen.

Wichtig ist die klare Trennung zwischen kreativer Steuerung und technischer Korrektur. Künstlerische Entscheidungen werden vor der Korrektur getroffen; Korrektur stabilisiert nur das Ergebnis im realen Raum.
Seite 19 – Lautsprecher vs. Kopfhörer

Kopfhörer bieten maximale Kontrolle über die Illusion, da der Raum der Hörer:in ausgeschlossen ist.

Lautsprecher benötigen zusätzliche Sorgfalt: Raumkorrektur, definierte Hörzone, optionale Übersprechkontrolle und pegelbewusste Tiefenverteilung.

Beide Wege sind gültig; die Wahl richtet sich nach Ziel und Ort der Vorführung.
Seite 20 – Kalibrierung & Raumkorrektur

Vor der kreativen Arbeit steht die Messung des realen Wiedergaberaums. Aus den Messungen werden lineare Korrekturen und Referenzpegel abgeleitet.

Kleine Lautsprecher profitieren stark von sorgfältiger Entzerrung und einer klaren Übergabe an etwaige Subwoofer.

Nach der Korrektur werden die künstlerischen Layer- und Tiefenentscheidungen übertragen und kontrolliert wiedergegeben.
Seite 21 – Übersprechkontrolle bei Lautsprechern

Die Unterdrückung des Übersprechens zwischen linkem und rechtem Kanal kann die Ortung verbessern, ist jedoch empfindlich gegenüber Kopfbewegungen und Sitzposition.

Einsatz nur in Szenen, die davon profitieren, und in Räumen, die ruhige Hörpositionen erlauben.

Stets mit moderater Intensität starten und hörend optimieren.
Seite 22 – Pegel, Dynamik & Sicherheit

Die Kette hält ausreichend Reserven vor. Raum- und Tiefenprozesse können Spitzen erzeugen; deshalb werden Headroom-Regeln eingehalten und eine begrenzende Stufe am Ende sichert gegen Übersteuerungen.

Dynamik bleibt musikalisch; Schutz greift unauffällig.

Ziel ist eine stressfreie, ermüdungsarme Langzeithörbarkeit bei gleichzeitiger dramaturgischer Wirkung.
Seite 23 – Echtzeit & Latenzmanagement

Die Bedienung fühlt sich direkt an, wenn Verarbeitungsverzögerungen klein und stabil sind.

Die Systemkonfiguration wird so gewählt, dass Bewegungen, Morphings und externe Steuerungen ohne spürbare Verzögerung wirksam werden.

Puffergrößen und Prozesslängen sind bewusst auf die Anforderungen des Auftritts oder der Produktion abgestimmt.
Seite 24 – Persistenz & Zustandsverwaltung

Alle kreativen Einstellungen werden als geordnete Zustände vorgehalten: Szenen, Presets, Snapshots.

Zwischenspeicherungen verhindern Verlust, geteilte Zustände ermöglichen Zusammenarbeit und Wiederaufführung.

Beim Laden eines Zustands werden nur definierte Parameter überschrieben; laufende Aufführungen bleiben kontrollierbar.
Seite 25 – Externe Steuerung: Prinzipien

Kontinuierliche Steuergrößen – etwa horizontale/vertikale Positionen, Tiefenwerte, Morphanteile – sind auf robuste, kurze Adressräume abgebildet.

Eingehende Werte werden sanft geglättet, damit Handbewegungen natürlich wirken und digitales Zittern vermieden wird.

Die Bandbreite der Aktualisierungen wird so bemessen, dass das Gesamtsystem stabil bleibt, auch im langen Betrieb.
Seite 26 – Plattformbetrieb (neutral)

Unabhängig vom Betriebssystem gilt:

• Sichere Audio-Routing-Pfade, klar dokumentiert.

• Stabiler Renderer mit definierter Reihenfolge der Verarbeitungsschritte.

• Reproduzierbare Start- und Shutdown-Sequenzen, inklusive Rückfallstrategien.

Die konkrete Tool-Auswahl ist frei, sofern die hier festgelegten Funktionen und Reihenfolgen eingehalten werden.
Seite 27 – Qualitätssicherung

Regelmäßige Hörproben mit Referenzmaterial und Standardszenen sichern Konsistenz.

Automatisierte Prüfungen der Zustandsdateien stellen sicher, dass alle notwendigen Parameter vorhanden und innerhalb gültiger Grenzen sind.

Vergleichshörplätze (Hauptplatz, Nebenplatz) helfen, robuste Einstellungen für reale Räume zu finden.
Seite 28 – Testplan

Zentrale Prüfziele:

• Gleichmäßige Bewegung entlang der Bahn; Zwischenzeitpunkt teilt nachvollziehbar.

• Easing verändert nur die Tempokurve, nie die geometrische Bahn.

• Mischungen bleiben pegelstetig; keine Sprünge in Lautheit oder Spektrum.

• Mehrspurverhalten bleibt geordnet; Phasenversätze erzeugen beabsichtigte Effekte.

• Zustandswechsel und Morphing sind klickfrei.

• Externe Steuerung wirkt ohne Ruckeln und ohne Überlastung des Systems.
Seite 29 – Abnahmekriterien

Eine Szene gilt als abgenommen, wenn:

• Die Raumillusion auf der Zielwiedergabe stabil ist.

• Die dokumentierten Layer-Charaktere identifizierbar und mischbar bleiben.

• Bewegungen dramaturgisch schlüssig und technisch sauber sind.

• Pegelreserven und Sicherheit nachweislich eingehalten werden.

• Zustände geladen und reproduzierbar wiedergegeben werden können.
Seite 30 – Roadmap kurz/mittel/lang

Kurzfristig: zusätzliche Referenzpunkte pro Bahn, feinere Zeitmarken, intuitivere Tiefenparameter.

Mittelfristig: Kopfbewegungsbezug für Lautsprecher-Betrieb, kuratierte Raumantwort-Bibliothek, erweiterte Mehrspurverwaltung.

Langfristig: adaptive Illusionen, die sich an Hörplatz und Material anpassen, sowie Austauschformate für eine interoperable Klangregie.
Seite 31 – Governance & Ethik

Die Gestaltung räumlicher Illusionen beeinflusst Wahrnehmung und Aufmerksamkeit. Verantwortungsvoll eingesetzt, stärken sie Fokus und Erlebnis; übertrieben eingesetzt, überfordern sie.

Das System folgt dem Prinzip der Transparenz: dokumentierte Entscheidungen, reproduzierbare Ergebnisse und Schutz der Hörenden vor schädlicher Lautheit oder Erschöpfung.
Seite 32 – Betrieb & Wartung

Wartung umfasst regelmäßige Überprüfung der Kalibrierung, der Zustände und der Abhörkette.

Updates an Teilkomponenten erfolgen nachvollziehbar: Vorher Tests an Standardszenen, danach Freigabe mit Protokoll.

Betriebsunterbrechungen sind geplant und kommuniziert; Rollback-Optionen bestehen für kritische Veranstaltungen.
Seite 33 – Risiken & Glossar

Risiken: Übersteuerung durch ungünstige Layer-Kombinationen; räumliche Instabilität bei starkem Übersprech; Ermüdung durch übertriebene Diffusion. Gegenmaßnahmen: Headroom-Regeln, moderater Einsatz von Übersprechkontrolle, regelmäßige Hörpausen und Referenzchecks.

Glossar:

• Archetyp: fest definierter Raumcharakter.

• Szene: vollständiger Klangzustand.

• Pfad: geglättete Bahn einer Bewegung.

• Morphing: stufenlose Überblendung zweier Zustände.

• Tiefe: wahrgenommene Nähe/Ferne als Kombination mehrerer Parameter.

Abschluss: MOTHERSHIP schafft eine gestaltbare, wiederholbare Raumpoesie, die aus Technik und Wahrnehmung ein zuverlässiges Handwerk formt.

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
