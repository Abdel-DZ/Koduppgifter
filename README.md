Kunskapskontroll – Kapitel 07, 08 och 10

1- Innehåll
Detta repo innehåller tre separata uppgifter:

Kapitel 07	ANN	Tränar ett enkelt artificiellt neuralt nätverk.
Kapitel 08	CNN	Bildklassificering med Convolutional Neural Network och en enkel app.
Kapitel 10	Chattbot (RAG)	Chattbot som svarar på frågor baserat på två PDF‑dokument.

2- Kapitel 07 – ANN
Syfte:
Träna ett artificiellt neuralt nätverk för klassificering.

Innehåll:

Databehandling

Modelluppbyggnad

Träning och utvärdering

Körning:
Öppna och kör notebooken Kapitel07.ipynb.

3- Kapitel 08 – CNN
Syfte:
Bygga och träna en CNN för bildklassificering.

Innehåll:

Modell för bildigenkänning

Träning på bilddata

Testning med egna bilder

Enkel applikation (app.py) för att klassificera bilder

Körning:
Notebook: Kapitel_08.ipynb  
App:
python app.py

4- Kapitel 10 – Chattbot (RAG)
Syfte:
Skapa en chattbot som använder dokument för att svara på frågor.

Innehåll:

PDF‑läsning

Chunking

Embeddings med SentenceTransformer

Semantisk sökning

Groq‑modell för att generera svar

Enkel evaluering

Körning:
python Kapitel_10.py

5- Installation
Installera beroenden:

pip install -r
numpy
polars
pypdf
sentence-transformers
groq
torch
flask

6- Krav
Python 3.10 eller senare

Nödvändiga bibliotek (NumPy, Polars, SentenceTransformer, Groq m.fl.)

PDF‑filerna ligger i mappen för Kapitel 10

7- Övrigt
API‑nycklar ska inte finnas i koden.
Projektet är rensat från hemligheter.
