from collections import Counter, defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import networkx as nx
import matplotlib.pyplot as plt

# Model tanımları
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Metin ve karakter listesi
with open("NLP-Hikayeler\hikayeler\ceviz.txt", "r", encoding="utf-8") as file:
    text = file.read()
sentences = [s.strip() for s in text.split(".") if s.strip()]

karakterler = []
with open("karakterler.txt", "r", encoding="utf-8") as file:
    karakterler = [line.strip() for line in file]

# Karakter ve ilişki bulucu
def karakterleri_bul(cumle, karakterler):
    cumle = cumle.lower()
    return [k for k in karakterler if k.lower() in cumle]

def duygusal_iliski_kur(cumle, karakterler, duygu):
    gecenler = karakterleri_bul(cumle, karakterler)
    if len(gecenler) >= 2:
        iliskiler = []
        for i in range(len(gecenler)):
            for j in range(i + 1, len(gecenler)):
                iliskiler.append({
                    "kim": gecenler[i],
                    "kime": gecenler[j],
                    "duygu": duygu
                })
        return iliskiler
    return []




#  Zamir çözümleme entegrasyonu
zamirler = {"o", "ona", "onu", "onun", "ondan", "kendi", "kendisi", "kendisini"}
son_karakter = None
tum_iliskiler = []
from collections import deque, Counter

son_karakterler = deque(maxlen=5)  # son geçen karakterleri tutar

for cumle in sentences:
    gecenler = karakterleri_bul(cumle, karakterler)
    kelimeler = set(cumle.lower().split())

    if gecenler:
        son_karakterler.extend(gecenler)
        aktif_karakter = gecenler[0]
    elif zamirler & kelimeler and son_karakterler:
        # Son karakterlerden en sık geçenini ata
        sayim = Counter(son_karakterler)
        aktif_karakter = sayim.most_common(1)[0][0]
        cumle = f"{aktif_karakter} {cumle}"
    else:
        aktif_karakter = None

    # Duygu ve ilişki çıkarımı
    duygu = classifier(cumle)[0]["label"]
    iliskiler = duygusal_iliski_kur(cumle, karakterler, duygu)
    tum_iliskiler.extend(iliskiler)

# for cumle in sentences:
#     gecenler = karakterleri_bul(cumle, karakterler)

#     if gecenler:
#         son_karakter = gecenler[0]
#     else:
#         kelimeler = set(cumle.lower().split())
#         if zamirler & kelimeler and son_karakter:
#             cumle = f"{son_karakter} {cumle}"

#     duygu = classifier(cumle)[0]["label"]
#     iliskiler = duygusal_iliski_kur(cumle, karakterler, duygu)
#     tum_iliskiler.extend(iliskiler)

# Duygusal ilişki özetleme
iliskiler_duygular = defaultdict(list)
for iliski in tum_iliskiler:
    c1, c2 = sorted([iliski["kim"], iliski["kime"]])
    iliskiler_duygular[(c1, c2)].append(iliski["duygu"])

baskin_iliskiler = {}
for cift, duygular in iliskiler_duygular.items():
    sayim = Counter(duygular)
    baskin_duygu = sayim.most_common(1)[0][0]
    baskin_iliskiler[cift] = baskin_duygu

# Görselleştirme
print("Baskın Duygusal İlişkiler:")
for (kim, kime), duygu in baskin_iliskiler.items():
    print(f"{kim} ile {kime} arasında baskın olarak {duygu} ilişkisi var.")

G = nx.Graph()
for (kim, kime), duygu in baskin_iliskiler.items():
    G.add_node(kim)
    G.add_node(kime)
    G.add_edge(kim, kime, duygu=duygu)

# Küçük harfli renk eşlemesi
duygu_renk = {
    "positive": "green",
    "negative": "red",
    "neutral": "gray"
}

# Kenar renklerini duygu etiketine göre ayarla
edge_colors = [duygu_renk.get(data['duygu'], 'black') for _, _, data in G.edges(data=True)]


plt.figure(figsize=(10, 7))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, edge_color=edge_colors, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold')
edge_labels = nx.get_edge_attributes(G, 'duygu')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black')
plt.title("Masal Karakterleri Arası Duygusal İlişki Ağı")
plt.axis('off')
plt.tight_layout()
plt.show()
