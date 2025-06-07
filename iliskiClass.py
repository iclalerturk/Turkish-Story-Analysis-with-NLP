from collections import Counter, defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import networkx as nx
import matplotlib.pyplot as plt
import re
import spacy
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PyQt5.QtGui import QImage, QPixmap
import io
from sklearn.metrics import pairwise_distances
import numpy as np

class KarakterIliski:
    def __init__(self, text_path, character_path="karakterler.txt"):
        self.model_name = "savasy/bert-base-turkish-sentiment-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.classifier = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        self.nlp = spacy.load("tr_core_news_trf")
        
        self.text_path = text_path
        self.character_path = character_path
        self.variation_2_character = {}
        self.text = ""
        self.sentences = []
        self.all_relationships = []
        self.previous_sentences = []
        self.pattern = re.compile(r'“[^“”]+”|"[^"]+"|[^“”".]+[.]')
        self.sentiment_colors = {"positive": "green", "negative": "red", "neutral": "gray"}  # duygu renkleri
        self.is_dialog = False

    def load_text(self):
        with open(self.text_path, "r", encoding="utf-8") as file:
            self.text = file.read()
            
        with open(self.character_path, "r", encoding="utf-8") as file:
            for line in file:
                parts = [p.strip() for p in line.strip().split(",") if p.strip()]
                if parts:
                    main_character = parts[0]
                    for variation in parts:
                        self.variation_2_character[variation.lower()] = main_character
                        
    def split_sentences_and_dialogues(self):
        for match in re.finditer(self.pattern, self.text):
            sentence = match.group().strip()
            if not sentence:
                continue
            self.is_dialog = (sentence.startswith("“") and sentence.endswith("”")) or (sentence.startswith('"') and sentence.endswith('"'))
            self.sentences.append((sentence.strip("“”\""), self.is_dialog))

    def find_characters(self,sentence):  # yardımcı fonksiyon
        sentence = sentence.lower()
        bulunanlar = set()
        for variation, karakter in self.variation_2_character.items():
            pattern = r'\b' + re.escape(variation.lower()) + r'(?:[ınunıneoaüie]{0,4})?\b'
            if re.search(pattern, sentence):
                bulunanlar.add(karakter)
        return list(bulunanlar)

    def has_pronoun(self,doc):
        return any(token.pos_ == "PRON" for token in doc)

    def extract_unique_characters(self, candidates):
        seen = set()
        unique_char = []
        for a in candidates:
            if a not in seen:
                unique_char.append(a)
                seen.add(a)
        return unique_char

    def find_subject_character(self, sentence):
        doc = self.nlp(sentence)
        for token in doc:
            if token.dep_ == "nsubj":
                word = token.text.lower()
                for variation, character in self.variation_2_character.items():
                    if variation.lower() == word:
                        return character

    def find_previous_char(self, max_lookback=3):
        candidates = []
        for sentence in reversed(self.previous_sentences[-max_lookback:]):
            mentioned_characters = self.find_characters(sentence)
            if mentioned_characters:
                candidates.extend(mentioned_characters)

        seen = set()
        unique_candidate = []
        for a in candidates:
            if a not in seen:
                unique_candidate.append(a)
                seen.add(a)
                
        if len(unique_candidate) == 1:
            return unique_candidate[0]
        
        for sentence in reversed(self.previous_sentences[-max_lookback:]):
            return self.find_subject_character(sentence)
                        
        if unique_candidate:
            return unique_candidate[0]
        return None
    
    def build_dialog_relationships(self, characters, relationships, sentiment, previous_character=None):
        if previous_character:  # diyalog içindeki karakterlerle önceki karakter arasında ilişki
            for target in characters:
                if target != previous_character:
                    relationships.append({"kim": previous_character, "kime": target, "duygu": sentiment})
        if len(characters) >= 2:  # diyalog içindeki karakterler kendi aralarında da ilişki kurabilir
            for i in range(len(characters)):
                for j in range(i + 1, len(characters)):
                    if characters[i] != characters[j]:
                        relationships.append({"kim": characters[i], "kime": characters[j], "duygu": sentiment})
        return relationships

    def build_narrative_relationships(self, characters, relationships, sentiment, previous_character=None):
        if len(characters) >= 2:  # diyalog değilse, karakterler arasında normal ilişki kur
            for i in range(len(characters)):
                for j in range(i + 1, len(characters)):
                    if characters[i] != characters[j]:
                        relationships.append({"kim": characters[i], "kime": characters[j], "duygu": sentiment})
        elif len(characters) == 1 and previous_character and characters[0] != previous_character:  # tek karakter varsa ve öncesinde karakter varsa, onlarla ilişki kur
            relationships.append({"kim": previous_character, "kime": characters[0], "duygu": sentiment})
        return relationships

    def build_sentiment_relationship(self, cumle, characters, sentiment, previous_character=None):
        relationships = []
        
        if self.is_dialog:
            # diyalog durumunda:
            # 1) önceki karakter varsa, diyalog içindeki tüm karakterlerle ilişki kur
            # 2. diyalog içindeki karakterler kendi aralarında da ilişki kurabilir
            self.build_dialog_relationships(characters, relationships, sentiment, previous_character=None)
        
        else:
            # diyalog değilse, karakterler arasında normal ilişki kur
            self.build_narrative_relationships(characters, relationships, sentiment, previous_character=None)
        
        return relationships

    def moving_average(self,data, window_size=5):
        return [sum(data[i:i+window_size])/window_size for i in range(len(data)-window_size+1)]

    def process(self):
        for i, (sentence, self.is_dialog) in enumerate(self.sentences):
            doc = self.nlp(sentence)
            mentioned = self.find_characters(sentence)
            if not mentioned and self.has_pronoun(doc):
                previous = self.find_previous_char()
                candidate_characters = []
                if self.is_dialog:
                    next_character = None
                    for j in range(i + 1, len(self.sentences)):
                        sonraki_cumle, _ = self.sentences[j]
                        next_characters = self.find_characters(sonraki_cumle)
                        if next_characters:
                            next_character = next_characters[0]
                            break
                    if previous:
                        candidate_characters.append(previous)
                    if next_character:
                        candidate_characters.append(next_character)
                else:
                    if previous:
                        candidate_characters.append(previous)
                mentioned = candidate_characters
            if not mentioned:
                self.previous_sentences.append(sentence)
                continue
            try:
                sentiment = self.classifier(sentence)[0]["label"]
            except:
                self.previous_sentences.append(sentence)
                continue
            previous = self.find_previous_char()
            relationships = self.build_sentiment_relationship(sentence, mentioned, sentiment, previous)
            self.all_relationships.extend(relationships)
            self.previous_sentences.append(sentence)
            

    def draw_sentiment_curve(self) -> QPixmap:
        sentiment_scores = []
        label_2_score = {"positive": 1, "neutral": 0, "negative": -1}
        for relationship in self.all_relationships:
            sentiment = relationship["duygu"]
            skor = label_2_score.get(sentiment, 0)
            sentiment_scores.append(skor)

        smoothed_sentiment_flow = self.moving_average(sentiment_scores, window_size=5)  # hareketli ortalama ile yumuşatma

        # duyguların değişim şiddetini hesapla
        sentiment_change_scores = [abs(smoothed_sentiment_flow[i] - smoothed_sentiment_flow[i - 1]) for i in range(1, len(smoothed_sentiment_flow))]
        if len(sentiment_change_scores) < 2:
            raise ValueError("Yeterli duygu verisi yok.")

        # en büyük iki kırılma noktasını bul
        scores_copy = sentiment_change_scores.copy()
        first_change = scores_copy.index(max(scores_copy)) + 1
        scores_copy[first_change - 1] = -1
        second_change = scores_copy.index(max(scores_copy)) + 1
        text_section_points = sorted([first_change, second_change])

        # Grafik çizimi
        fig = plt.figure(figsize=(10, 4))
        plt.plot(smoothed_sentiment_flow, color='purple', linewidth=2)
        plt.axvline(x=text_section_points[0], color='gray', linestyle='--', label="Bölüm Geçişi")
        plt.axvline(x=text_section_points[1], color='gray', linestyle='--')
        plt.title("Duygu Eğrisi (Hareketli Ortalama ile)")
        plt.xlabel("Zaman (Cümle/İlişki Sırası)")
        plt.ylabel("Duygu Skoru")
        plt.yticks([-1, 0, 1], ["Negatif", "Nötr", "Pozitif"])
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        # Belleğe çiz
        canvas = FigureCanvasAgg(fig)
        buf = io.BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        plt.close(fig)

        image = QImage.fromData(buf.getvalue())
        pixmap = QPixmap.fromImage(image)

        return pixmap

    def overall_relationship_graph(self) -> QPixmap:
        relationships_sentiments = defaultdict(list)
        for relationship in self.all_relationships:
            c1, c2 = sorted([relationship["kim"], relationship["kime"]])
            relationships_sentiments[(c1, c2)].append(relationship["duygu"])

        dominant_relationships = {}
        for character_pair, sentiments in relationships_sentiments.items():
            count = Counter(sentiments)
            dominant_sentiment = count.most_common(1)[0][0]
            dominant_relationships[character_pair] = dominant_sentiment

        G = nx.Graph()
        for (kim, kime), duygu in dominant_relationships.items():
            G.add_node(kim)
            G.add_node(kime)
            G.add_edge(kim, kime, duygu=duygu)

        edge_colors = [self.sentiment_colors.get(data['duygu'], 'black') for _, _, data in G.edges(data=True)]

        fig = plt.figure(figsize=(10, 7))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, edge_color=edge_colors, node_color='lightblue',
                node_size=2000, font_size=10, font_weight='bold')
        edge_labels = nx.get_edge_attributes(G, 'duygu')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black')
        plt.title("Genel Masal Karakterleri Duygusal İlişki Ağı")
        plt.axis('off')
        plt.tight_layout()

        canvas = FigureCanvasAgg(fig)
        buf = io.BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        plt.close(fig)

        image = QImage.fromData(buf.getvalue())
        pixmap = QPixmap.fromImage(image)

        return pixmap

    def plot_graph_by_section(self, title=None) -> QPixmap:
        sentiment_sequence = [iliski["duygu"] for iliski in self.all_relationships]  # duyguları vektörleştirme (sadece sıralı olarak işlenebilir hale getirme)
        sentiment_count = len(sentiment_sequence)
        
        if sentiment_count < 3:
            raise ValueError("Yeterli ilişki verisi yok (en az 3 ilişki gerekli)")

        # Duyguları bir indeks listesine çevir
        unique_sentiments = list(set(sentiment_sequence))
        sentiment_2_idx = {sentiment: i for i, sentiment in enumerate(unique_sentiments)}
        vectors = [sentiment_2_idx[d] for d in sentiment_sequence]

        # Değişim skorlarını hesapla (birbirini takip eden duygular arasındaki fark)
        score_change = [abs(vectors[i] - vectors[i - 1]) for i in range(1, len(vectors))]

        # En büyük iki kırılma noktası bulunur
        score_copy = score_change.copy()
        firs_sentiment_change = score_copy.index(max(score_copy)) + 1
        score_copy[firs_sentiment_change - 1] = -1  # ilkini eledik
        second_sentiment_change = score_copy.index(max(score_copy)) + 1

        # Sıraya göre ayarlama (garanti)
        section_points = sorted([firs_sentiment_change, second_sentiment_change])

        introduction = self.all_relationships[:section_points[0]]
        development = self.all_relationships[section_points[0]:section_points[1]]
        conclusion = self.all_relationships[section_points[1]:]

        sections = {
            "Giriş Bölümü": introduction,
            "Gelişme Bölümü": development,
            "Sonuç Bölümü": conclusion
        }

        if title not in sections:
            raise ValueError("Geçersiz bölüm başlığı. 'Giriş Bölümü', 'Gelişme Bölümü' veya 'Sonuç Bölümü' olmalıdır.")

        section = sections[title]

        relationships_sentiments = defaultdict(list)
        for relationship in section:
            c1, c2 = sorted([relationship["kim"], relationship["kime"]])
            relationships_sentiments[(c1, c2)].append(relationship["duygu"])

        dominant_relationships = {}
        for character_pair, sentiments in relationships_sentiments.items():
            sayim = Counter(sentiments)
            dominant_sentiment = sayim.most_common(1)[0][0]
            dominant_relationships[character_pair] = dominant_sentiment

        G = nx.Graph()
        for (kim, kime), duygu in dominant_relationships.items():
            G.add_node(kim)
            G.add_node(kime)
            G.add_edge(kim, kime, duygu=duygu)

        edge_colors = [self.sentiment_colors.get(data['duygu'], 'black') for _, _, data in G.edges(data=True)]

        fig = plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, edge_color=edge_colors, node_color='lightblue',
                node_size=2000, font_size=10, font_weight='bold')
        edge_labels = nx.get_edge_attributes(G, 'duygu')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black')
        plt.title(f"{title} Duygusal İlişki Ağı")
        plt.axis('off')
        plt.tight_layout()

        canvas = FigureCanvasAgg(fig)
        buf = io.BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        plt.close(fig)

        image = QImage.fromData(buf.getvalue())
        pixmap = QPixmap.fromImage(image)

        return pixmap



# if __name__ == "__main__":
#     iliski = KarakterIliski("NLP-Hikayeler/hikayeler/zehrakisahikaye.txt")
#     iliski.metin_yukle()
#     iliski.cumle_diyalog_ayirma()
#     iliski.islemler()
#     iliski.duygu_egrisi_cizme()
#     iliski.genel_iliski_grafigi()
#     iliski.bolume_gore_grafik("Giriş Bölümü")