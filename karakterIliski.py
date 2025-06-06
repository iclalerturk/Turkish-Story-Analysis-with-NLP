from collections import Counter, defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import networkx as nx
import matplotlib.pyplot as plt
import re
import spacy

# 🔧 NLP ve Model ayarları
model_name = "savasy/bert-base-turkish-sentiment-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
nlp = spacy.load("tr_core_news_trf")

# 📄 Metin ve karakter-varyasyon yükleme
with open("NLP-Hikayeler/hikayeler/zehrakisahikaye.txt", "r", encoding="utf-8") as file:
    text = file.read()

varyasyon2karakter = {}
with open("karakterler.txt", "r", encoding="utf-8") as file:
    for line in file:
        parts = [p.strip() for p in line.strip().split(",") if p.strip()]
        if parts:
            ana_karakter = parts[0]
            for varyasyon in parts:
                varyasyon2karakter[varyasyon.lower()] = ana_karakter

# ✂ Cümle ve diyalog ayırma
parcalar = []
pattern = re.compile(r'“[^“”]+”|"[^"]+"|[^“”".]+[.]')
for eslesme in re.finditer(pattern, text):
    cumle = eslesme.group().strip()
    if not cumle:
        continue
    is_diyalog = (cumle.startswith("“") and cumle.endswith("”")) or (cumle.startswith('"') and cumle.endswith('"'))
    parcalar.append((cumle.strip("“”\""), is_diyalog))

# 🔍 Yardımcı fonksiyonlar
def karakterleri_bul(cumle, varyasyon2karakter):
    cumle = cumle.lower()
    bulunanlar = set()
    for varyasyon, karakter in varyasyon2karakter.items():
        pattern = r'\b' + re.escape(varyasyon.lower()) + r'(?:[ınunıneoaüie]{0,4})?\b'
        if re.search(pattern, cumle):
            bulunanlar.add(karakter)
    return list(bulunanlar)

def cumlede_zamir_var_mi(doc):
    return any(token.pos_ == "PRON" for token in doc)

def onceki_karakteri_bul(gecmis_cumleler, varyasyon2karakter, nlp, max_geri_bakis=3):
    adaylar = []
    for cumle in reversed(gecmis_cumleler[-max_geri_bakis:]):
        gecen_karakterler = karakterleri_bul(cumle, varyasyon2karakter)
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
    for cumle in reversed(gecmis_cumleler[-max_geri_bakis:]):
        doc = nlp(cumle)
        for token in doc:
            if token.dep_ == "nsubj":
                kelime = token.text.lower()
                for varyasyon, karakter in varyasyon2karakter.items():
                    if varyasyon.lower() == kelime:
                        return karakter
    if benzersiz_adaylar:
        return benzersiz_adaylar[0]
    return None

def duygusal_iliski_kur(cumle, karakterler, duygu, is_diyalog, onceki_karakter=None):
    iliskiler = []
    
    if is_diyalog:
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

# 🔁 Ana analiz döngüsü
tum_iliskiler = []
gecmis_cumleler = []
for i, (cumle, is_diyalog) in enumerate(parcalar):
    doc = nlp(cumle)
    gecenler = karakterleri_bul(cumle, varyasyon2karakter)
    if not gecenler and cumlede_zamir_var_mi(doc):
        onceki = onceki_karakteri_bul(gecmis_cumleler, varyasyon2karakter, nlp)
        tahmini_karakterler = []
        if is_diyalog:
            sonraki = None
            for j in range(i + 1, len(parcalar)):
                sonraki_cumle, _ = parcalar[j]
                sonraki_karakterler = karakterleri_bul(sonraki_cumle, varyasyon2karakter)
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
        gecmis_cumleler.append(cumle)
        continue
    try:
        duygu = classifier(cumle)[0]["label"]
    except:
        gecmis_cumleler.append(cumle)
        continue
    onceki = onceki_karakteri_bul(gecmis_cumleler, varyasyon2karakter, nlp)
    iliskiler = duygusal_iliski_kur(cumle, gecenler, duygu, is_diyalog, onceki)
    tum_iliskiler.extend(iliskiler)
    gecmis_cumleler.append(cumle)

# 📈 Duygu eğrisi çizimi
duygu_puanlari = []
etiket2skor = {"positive": 1, "neutral": 0, "negative": -1}
for iliski in tum_iliskiler:
    duygu = iliski["duygu"]
    skor = etiket2skor.get(duygu, 0)
    duygu_puanlari.append(skor)

def moving_average(data, window_size=5):
    return [sum(data[i:i+window_size])/window_size for i in range(len(data)-window_size+1)]

duygu_akisi_smooth = moving_average(duygu_puanlari, window_size=5)

n = len(tum_iliskiler)
bolum1_bitis = n // 3
bolum2_bitis = 2 * n // 3
bolumler = [
    tum_iliskiler[:bolum1_bitis],
    tum_iliskiler[bolum1_bitis:bolum2_bitis],
    tum_iliskiler[bolum2_bitis:]
]
basliklar = ["Giriş Bölümü", "Gelişme Bölümü", "Sonuç Bölümü"]

plt.figure(figsize=(10, 4))
plt.plot(duygu_akisi_smooth, color='purple', linewidth=2)
plt.axvline(x=bolum1_bitis, color='gray', linestyle='--')
plt.axvline(x=bolum2_bitis, color='gray', linestyle='--')
plt.title("Duygu Eğrisi (Hareketli Ortalama ile)")
plt.xlabel("Zaman (Cümle/İlişki Sırası)")
plt.ylabel("Duygu Skoru")
plt.yticks([-1, 0, 1], ["Negatif", "Nötr", "Pozitif"])
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

duygu_renk = {"positive": "green", "negative": "red", "neutral": "gray"}

for bolum, baslik in zip(bolumler, basliklar):
    if not bolum:
        continue
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
    edge_colors = [duygu_renk.get(data['duygu'], 'black') for _, _, data in G.edges(data=True)]
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, edge_color=edge_colors, node_color='lightblue',
            node_size=2000, font_size=10, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'duygu')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black')
    plt.title(f"{baslik} Duygusal İlişki Ağı")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

print("Genel İlişki Grafiği (Toplam):")
iliskiler_duygular = defaultdict(list)
for iliski in tum_iliskiler:
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

edge_colors = [duygu_renk.get(data['duygu'], 'black') for _, _, data in G.edges(data=True)]

plt.figure(figsize=(10, 7))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, edge_color=edge_colors, node_color='lightblue',
        node_size=2000, font_size=10, font_weight='bold')
edge_labels = nx.get_edge_attributes(G, 'duygu')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black')
plt.title("Genel Masal Karakterleri Duygusal İlişki Ağı")
plt.axis('off')
plt.tight_layout()
plt.show()