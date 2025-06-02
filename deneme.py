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
with open("NLP-Hikayeler/hikayeler/guzelcirkin.txt", "r", encoding="utf-8") as file:
    text = file.read()
sentences = [s.strip() for s in text.split(".") if s.strip()]

karakterler = []
with open("karakterler.txt", "r", encoding="utf-8") as file:
    karakterler = [line.strip() for line in file]

# Soyut karakter listesi
soyut_karakterler = {"anne", "baba", "kardeş", "adam", "kadın", "çocuk", "genç", "ihtiyar"}
zamirler = {"o", "ona", "onu", "onun", "ondan", "kendi", "kendisi", "kendisini"}

# Karakter bulucu
def karakterleri_bul(cumle, karakterler):
    cumle = cumle.lower()
    return [k for k in karakterler if k.lower() in cumle]

# Duygusal ilişki çıkarımı
def duygusal_iliski_kur(cumle, karakterler, duygu):
    gecenler = karakterleri_bul(cumle, karakterler)
    soyutlar = [k for k in gecenler if k.lower() in soyut_karakterler]
    somutlar = [k for k in gecenler if k.lower() not in soyut_karakterler]

    iliskiler = []

    # Normal çoklu karakter ilişkisi
    if len(gecenler) >= 2:
        for i in range(len(gecenler)):
            for j in range(i + 1, len(gecenler)):
                iliskiler.append({
                    "kim": gecenler[i],
                    "kime": gecenler[j],
                    "duygu": duygu
                })

    # Ek olarak soyutları somutlara bağla
    for soyut in soyutlar:
        for somut in somutlar:
            iliskiler.append({
                "kim": soyut,
                "kime": somut,
                "duygu": duygu
            })

    return iliskiler

# Duygusal ilişki çıkarımı döngüsü
son_karakter = None
tum_iliskiler = []

for idx, cumle in enumerate(sentences):
    gecenler = karakterleri_bul(cumle, karakterler)
    soyutlar = [k for k in gecenler if k.lower() in soyut_karakterler]
    somutlar = [k for k in gecenler if k.lower() not in soyut_karakterler]

    # Önceki ve sonraki cümlelerdeki karakterler
    onceki_karakter = karakterleri_bul(sentences[idx - 1], karakterler) if idx > 0 else []
    sonraki_karakter = karakterleri_bul(sentences[idx + 1], karakterler) if idx < len(sentences) - 1 else []

    if soyutlar and not somutlar:
        referanslar = [k for k in onceki_karakter + sonraki_karakter if k.lower() not in soyut_karakterler]
        if referanslar:
            cumle = f"{referanslar[0]} {cumle}"
            somutlar = [referanslar[0]]

    if somutlar:
        son_karakter = somutlar[0]
    else:
        kelimeler = set(cumle.lower().split())
        if zamirler & kelimeler and son_karakter:
            cumle = f"{son_karakter} {cumle}"

    duygu = classifier(cumle)[0]["label"]
    iliskiler = duygusal_iliski_kur(cumle, karakterler, duygu)
    tum_iliskiler.extend(iliskiler)

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

duygu_renk = {
    "positive": "green",
    "negative": "red",
    "neutral": "gray"
}
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

# Karakterlerin ilişki kümelerini oluştur
karakter_iliski_kumesi = defaultdict(set)
for (kim, kime), duygu in baskin_iliskiler.items():
    karakter_iliski_kumesi[kim].add(kime)
    karakter_iliski_kumesi[kime].add(kim)

# Jaccard benzerliği tanımı
def jaccard_benzerlik(kume1, kume2):
    kesisim = kume1 & kume2
    birlesim = kume1 | kume2
    if not birlesim:
        return 0.0
    return len(kesisim) / len(birlesim)

# Benzer karakterleri eşle
benzer_karakterler = []
karakterler_listesi = list(karakter_iliski_kumesi.keys())
benzerlik_esigi = 0.8

for i in range(len(karakterler_listesi)):
    for j in range(i + 1, len(karakterler_listesi)):
        k1, k2 = karakterler_listesi[i], karakterler_listesi[j]
        benzerlik = jaccard_benzerlik(karakter_iliski_kumesi[k1], karakter_iliski_kumesi[k2])
        if benzerlik > benzerlik_esigi:
            benzer_karakterler.append((k1, k2, benzerlik))

# Karakter birleştirme eşlemesi
birlesme_haritasi = {}
gruplar = []

for k1, k2, _ in benzer_karakterler:
    bulundu = False
    for grup in gruplar:
        if k1 in grup or k2 in grup:
            grup.update([k1, k2])
            bulundu = True
            break
    if not bulundu:
        gruplar.append(set([k1, k2]))

for grup in gruplar:
    isim = "/".join(sorted(grup))
    for karakter in grup:
        birlesme_haritasi[karakter] = isim

# Yeni karakter adlarıyla ilişki grafiği
G2 = nx.Graph()
for (kim, kime), duygu in baskin_iliskiler.items():
    yeni_kim = birlesme_haritasi.get(kim, kim)
    yeni_kime = birlesme_haritasi.get(kime, kime)
    if yeni_kim != yeni_kime:
        G2.add_node(yeni_kim)
        G2.add_node(yeni_kime)
        G2.add_edge(yeni_kim, yeni_kime, duygu=duygu)

edge_colors = [duygu_renk.get(data['duygu'], 'black') for _, _, data in G2.edges(data=True)]

plt.figure(figsize=(10, 7))
pos = nx.spring_layout(G2, seed=42)
nx.draw(G2, pos, with_labels=True, edge_color=edge_colors, node_color='lightgreen', node_size=2000, font_size=10, font_weight='bold')
edge_labels = nx.get_edge_attributes(G2, 'duygu')
nx.draw_networkx_edge_labels(G2, pos, edge_labels=edge_labels, font_color='black')
plt.title("Birleştirilmiş Karakterlerle Duygusal İlişki Ağı")
plt.axis('off')
plt.tight_layout()
plt.show()
