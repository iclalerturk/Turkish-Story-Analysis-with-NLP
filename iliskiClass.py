from collections import Counter, defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import networkx as nx
import matplotlib.pyplot as plt
import re
import spacy
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PyQt5.QtGui import QImage, QPixmap
import io

class KarakterIliski:
    def __init__(self, hikaye_yolu, karakter_yolu="karakterler.txt"):
        # 🔧 NLP ve Model ayarları
        self.model_name = "savasy/bert-base-turkish-sentiment-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.classifier = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        self.nlp = spacy.load("tr_core_news_trf")
        
        # 📄 Metin ve karakter-varyasyon yükleme
        self.hikaye_yolu = hikaye_yolu
        self.karakter_yolu = karakter_yolu
        self.varyasyon2karakter = {}
        self.text = ""
        self.parcalar = []
        self.tum_iliskiler = []
        self.gecmis_cumleler = []
        self.pattern = re.compile(r'“[^“”]+”|"[^"]+"|[^“”".]+[.]')
        # Duygu renkleri
        self.duygu_renk = {"positive": "green", "negative": "red", "neutral": "gray"}
        self.is_diyalog = False
    def metin_yukle(self):
        with open(self.hikaye_yolu, "r", encoding="utf-8") as file:
            self.text = file.read()
            
        with open(self.karakter_yolu, "r", encoding="utf-8") as file:
            for line in file:
                parts = [p.strip() for p in line.strip().split(",") if p.strip()]
                if parts:
                    ana_karakter = parts[0]
                    for varyasyon in parts:
                        self.varyasyon2karakter[varyasyon.lower()] = ana_karakter
                        
    def cumle_diyalog_ayirma(self):
        for eslesme in re.finditer(self.pattern, self.text):
            cumle = eslesme.group().strip()
            if not cumle:
                continue
            self.is_diyalog = (cumle.startswith("“") and cumle.endswith("”")) or (cumle.startswith('"') and cumle.endswith('"'))
            self.parcalar.append((cumle.strip("“”\""), self.is_diyalog))

    # 🔍 Yardımcı fonksiyonlar
    def karakterleri_bul(self,cumle):
        cumle = cumle.lower()
        bulunanlar = set()
        for varyasyon, karakter in self.varyasyon2karakter.items():
            pattern = r'\b' + re.escape(varyasyon.lower()) + r'(?:[ınunıneoaüie]{0,4})?\b'
            if re.search(pattern, cumle):
                bulunanlar.add(karakter)
        return list(bulunanlar)

    def cumlede_zamir_var_mi(self,doc):
        return any(token.pos_ == "PRON" for token in doc)

    def onceki_karakteri_bul(self, max_geri_bakis=3):
        adaylar = []
        for cumle in reversed(self.gecmis_cumleler[-max_geri_bakis:]):
            gecen_karakterler = self.karakterleri_bul(cumle)
            if gecen_karakterler:
                adaylar.extend(gecen_karakterler)
        seen = set()
        benzersiz_adaylar = []
        for a in adaylar:
            if a not in seen:
                benzersiz_adaylar.append(a)
                seen.add(a)
        if len(benzersiz_adaylar) == 1:
            return benzersiz_adaylar[0]
        for cumle in reversed(self.gecmis_cumleler[-max_geri_bakis:]):
            doc = self.nlp(cumle)
            for token in doc:
                if token.dep_ == "nsubj":
                    kelime = token.text.lower()
                    for varyasyon, karakter in self.varyasyon2karakter.items():
                        if varyasyon.lower() == kelime:
                            return karakter
        if benzersiz_adaylar:
            return benzersiz_adaylar[0]
        return None

    def duygusal_iliski_kur(self, cumle, karakterler, duygu, onceki_karakter=None):
        iliskiler = []
        
        if self.is_diyalog:
            # Diyalog durumunda:
            # 1. Önceki karakter varsa, diyalog içindeki tüm karakterlerle ilişki kur
            if onceki_karakter:
                # Diyalog içindeki karakterlerle önceki karakter arasında ilişki
                for hedef in karakterler:
                    if hedef != onceki_karakter:
                        iliskiler.append({"kim": onceki_karakter, "kime": hedef, "duygu": duygu})
            
            # 2. Diyalog içindeki karakterler kendi aralarında da ilişki kurabilir
            if len(karakterler) >= 2:
                for i in range(len(karakterler)):
                    for j in range(i + 1, len(karakterler)):
                        if karakterler[i] != karakterler[j]:
                            iliskiler.append({"kim": karakterler[i], "kime": karakterler[j], "duygu": duygu})
        
        else:
            # Diyalog değilse, karakterler arasında normal ilişki kur
            if len(karakterler) >= 2:
                for i in range(len(karakterler)):
                    for j in range(i + 1, len(karakterler)):
                        if karakterler[i] != karakterler[j]:
                            iliskiler.append({"kim": karakterler[i], "kime": karakterler[j], "duygu": duygu})
            
            # Tek karakter varsa ve önceki karakter varsa, onlarla ilişki kur
            elif len(karakterler) == 1 and onceki_karakter and karakterler[0] != onceki_karakter:
                iliskiler.append({"kim": onceki_karakter, "kime": karakterler[0], "duygu": duygu})
        
        return iliskiler

    def moving_average(self,data, window_size=5):
        return [sum(data[i:i+window_size])/window_size for i in range(len(data)-window_size+1)]

    def islemler(self):
        for i, (cumle, self.is_diyalog) in enumerate(self.parcalar):
            doc = self.nlp(cumle)
            gecenler = self.karakterleri_bul(cumle)
            if not gecenler and self.cumlede_zamir_var_mi(doc):
                onceki = self.onceki_karakteri_bul()
                tahmini_karakterler = []
                if self.is_diyalog:
                    sonraki = None
                    for j in range(i + 1, len(self.parcalar)):
                        sonraki_cumle, _ = self.parcalar[j]
                        sonraki_karakterler = self.karakterleri_bul(sonraki_cumle)
                        if sonraki_karakterler:
                            sonraki = sonraki_karakterler[0]
                            break
                    if onceki:
                        tahmini_karakterler.append(onceki)
                    if sonraki:
                        tahmini_karakterler.append(sonraki)
                else:
                    if onceki:
                        tahmini_karakterler.append(onceki)
                gecenler = tahmini_karakterler
            if not gecenler:
                self.gecmis_cumleler.append(cumle)
                continue
            try:
                duygu = self.classifier(cumle)[0]["label"]
            except:
                self.gecmis_cumleler.append(cumle)
                continue
            onceki = self.onceki_karakteri_bul()
            iliskiler = self.duygusal_iliski_kur(cumle, gecenler, duygu, onceki)
            self.tum_iliskiler.extend(iliskiler)
            self.gecmis_cumleler.append(cumle)
            

    def duygu_egrisi_cizme(self) -> QPixmap:
        # 📈 Duygu eğrisi çizimi
        duygu_puanlari = []
        etiket2skor = {"positive": 1, "neutral": 0, "negative": -1}
        for iliski in self.tum_iliskiler:
            duygu = iliski["duygu"]
            skor = etiket2skor.get(duygu, 0)
            duygu_puanlari.append(skor)

        # Hareketli ortalama ile yumuşatma
        duygu_akisi_smooth = self.moving_average(duygu_puanlari, window_size=5)

        # Duyguların değişim şiddetini hesapla
        degisim_skorlari = [abs(duygu_akisi_smooth[i] - duygu_akisi_smooth[i - 1]) for i in range(1, len(duygu_akisi_smooth))]
        if len(degisim_skorlari) < 2:
            raise ValueError("Yeterli duygu verisi yok.")

        # En büyük iki kırılma noktasını bul
        skorlar_kopya = degisim_skorlari.copy()
        ilk_kirilma = skorlar_kopya.index(max(skorlar_kopya)) + 1
        skorlar_kopya[ilk_kirilma - 1] = -1
        ikinci_kirilma = skorlar_kopya.index(max(skorlar_kopya)) + 1
        bolum_noktalar = sorted([ilk_kirilma, ikinci_kirilma])

        # Grafik çizimi
        fig = plt.figure(figsize=(10, 4))
        plt.plot(duygu_akisi_smooth, color='purple', linewidth=2)
        plt.axvline(x=bolum_noktalar[0], color='gray', linestyle='--', label="Bölüm Geçişi")
        plt.axvline(x=bolum_noktalar[1], color='gray', linestyle='--')
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

    def genel_iliski_grafigi(self) -> QPixmap:
        iliskiler_duygular = defaultdict(list)
        for iliski in self.tum_iliskiler:
            c1, c2 = sorted([iliski["kim"], iliski["kime"]])
            iliskiler_duygular[(c1, c2)].append(iliski["duygu"])

        baskin_iliskiler = {}
        for cift, duygular in iliskiler_duygular.items():
            sayim = Counter(duygular)
            baskin_duygu = sayim.most_common(1)[0][0]
            baskin_iliskiler[cift] = baskin_duygu

        G = nx.Graph()
        for (kim, kime), duygu in baskin_iliskiler.items():
            G.add_node(kim)
            G.add_node(kime)
            G.add_edge(kim, kime, duygu=duygu)

        edge_colors = [self.duygu_renk.get(data['duygu'], 'black') for _, _, data in G.edges(data=True)]

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

    def bolume_gore_grafik(self, baslik=None) -> QPixmap:
        from sklearn.metrics import pairwise_distances
        import numpy as np

        # Duyguları vektörleştirme (sadece sıralı olarak işlenebilir hale getirme)
        duygu_sirasi = [iliski["duygu"] for iliski in self.tum_iliskiler]
        duygu_sayisi = len(duygu_sirasi)
        
        if duygu_sayisi < 3:
            raise ValueError("Yeterli ilişki verisi yok (en az 3 ilişki gerekli)")

        # Duyguları bir indeks listesine çevir
        benzersiz_duygular = list(set(duygu_sirasi))
        duygu2idx = {duygu: i for i, duygu in enumerate(benzersiz_duygular)}
        vektorler = [duygu2idx[d] for d in duygu_sirasi]

        # Değişim skorlarını hesapla (birbirini takip eden duygular arasındaki fark)
        degisim_skorlari = [abs(vektorler[i] - vektorler[i - 1]) for i in range(1, len(vektorler))]

        # En büyük iki kırılma noktası bulunur
        skorlar_kopya = degisim_skorlari.copy()
        ilk_kirilma = skorlar_kopya.index(max(skorlar_kopya)) + 1
        skorlar_kopya[ilk_kirilma - 1] = -1  # ilkini eledik
        ikinci_kirilma = skorlar_kopya.index(max(skorlar_kopya)) + 1

        # Sıraya göre ayarlama (garanti)
        bolum_noktalar = sorted([ilk_kirilma, ikinci_kirilma])

        bolum1 = self.tum_iliskiler[:bolum_noktalar[0]]
        bolum2 = self.tum_iliskiler[bolum_noktalar[0]:bolum_noktalar[1]]
        bolum3 = self.tum_iliskiler[bolum_noktalar[1]:]

        bolumler = {
            "Giriş Bölümü": bolum1,
            "Gelişme Bölümü": bolum2,
            "Sonuç Bölümü": bolum3
        }

        if baslik not in bolumler:
            raise ValueError("Geçersiz bölüm başlığı. 'Giriş Bölümü', 'Gelişme Bölümü' veya 'Sonuç Bölümü' olmalıdır.")

        bolum = bolumler[baslik]

        iliskiler_duygular = defaultdict(list)
        for iliski in bolum:
            c1, c2 = sorted([iliski["kim"], iliski["kime"]])
            iliskiler_duygular[(c1, c2)].append(iliski["duygu"])

        baskin_iliskiler = {}
        for cift, duygular in iliskiler_duygular.items():
            sayim = Counter(duygular)
            baskin_duygu = sayim.most_common(1)[0][0]
            baskin_iliskiler[cift] = baskin_duygu

        G = nx.Graph()
        for (kim, kime), duygu in baskin_iliskiler.items():
            G.add_node(kim)
            G.add_node(kime)
            G.add_edge(kim, kime, duygu=duygu)

        edge_colors = [self.duygu_renk.get(data['duygu'], 'black') for _, _, data in G.edges(data=True)]

        fig = plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, edge_color=edge_colors, node_color='lightblue',
                node_size=2000, font_size=10, font_weight='bold')
        edge_labels = nx.get_edge_attributes(G, 'duygu')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black')
        plt.title(f"{baslik} Duygusal İlişki Ağı")
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
