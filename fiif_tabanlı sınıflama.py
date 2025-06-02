from collections import Counter, defaultdict
from difflib import get_close_matches
import re
import spacy
from nltk.tokenize import word_tokenize

nlp = spacy.load("tr_core_news_trf")

insansi_eylemler = {
    "duy", "sor", "sus", "git", "oyna", "hazırla", "paylaş", "tut", "ör", "söyle", "de", "konuş", 
    "düşün", "hisset", "hissed", "koyul", "kazan", "dokun", "tat", "yürü", "koş", "uyu", "uyan", 
    "gülümse", "ağla", "gül", "sev", "kork", "özlem", "iste", "hatırla", "unut", "kız", "öfkelen", 
    "öfke", "eğit", "uyar", "heyecanlan", "hüzünlen", "endişelen", "sevin", "üzül", "merak", "meraklan", 
    "kıskan", "ilgil", "sabırsızlan", "sakinleş", "kandır", "inandır", "cesaretlendir", "et", "incele", 
    "anla", "farket", "duygulan", "güven", "şüphelen", "şüphe", "destekle", "yargıla", "suçla", 
    "affet", "umutla", "eğlen", "rahatla", "çabala", "düşle", "şaşır", "şaş", "şevklen", "iste", 
    "koru", "gül", "ver", "yap", "bak", "öğren", "özle", "dinle", "planla", "arzula", "hedefle", "başla","sor",
    "gözlemle", "çöz", "kararlaştır", "reddet", "dinlen", "korkut", "izle", "oku", "anlat", "çalış","katıl",
    "yaz", "tartış", "hissizleş", "yürekglen", "yüreklendir", "gizle", "dile", "anlamlandır", "öv", 
    "döv", "eleştir", "sorgula", "haket", "diren", "vazgeç", "anlaş", "bağışla", "bağır", "mırıldan","ye", 
    "çalış", "konuş", "al", "dile", "kız", "kır", "kırıl", "hastalan"
}

turkce_stopwords = {
    "bir", "bu", "şu", "çok", "daha", "mi", "mu", "mı", "mü", "şey", "tane",
    "ve","ile", "de", "da", "ki", "ne", "ni", "için", "ama", "fakat", "ancak",
    "gibi", "ise", "en", "ya", "iç", "herkes","kimse","kendi"," ","\"","“","“",
    "hem", "veya", "ya da", "çünkü", "lakin", "diye", "kadar", "sonra", "kişi",
    "önce", "her", "hiç", "hep", "bazı", "tüm", "hangi", "neden", "nasıl", "nerede", "ne zaman",
    "a", "na", "dan", "den", "e", "ye", "deki", "daki", "nin", "nın", "nun", "nün",
    "ben", "sen", "o", "ora", "bura", "şura", "orada", "burada", "şurada", "biri", "biz", "siz"
}

def metin_onisleme(text):
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"'(\s+)", "'", text)
    return text

def temizle_kelime(kelime):
    return re.split(r"[’']", kelime)[0].lower()

def tamlayici_zincir(token):
    tamlayicilar = [token]
    stack = [token]
    while stack:
        current = stack.pop()
        for child in current.children:
            if child.dep_ in ["amod", "compound", "nmod:poss", "nmod"]:
                tamlayicilar.append(child)
                stack.append(child)
    if len(tamlayicilar) == 1:
        return None
    tamlayicilar = sorted(tamlayicilar, key=lambda x: x.i)
    return " ".join(t.text.lower() for t in tamlayicilar)

# ---------- Dosya oku ve işleme ----------
path = "NLP-Hikayeler/hikayeler/habilkabil.txt"
with open(path, "r", encoding="utf-8") as file:
    sentence = file.read()

sentence = metin_onisleme(sentence)
doc = nlp(sentence)

characters = set()

# NER tabanlı karakter çıkarımı
for ent in doc.ents:
    temiz_karakter = temizle_kelime(ent.text)
    if ent.label_ == "PERSON" and temiz_karakter not in characters and len(ent) > 1:
        if temiz_karakter not in turkce_stopwords:
            characters.add(temiz_karakter)

# Fiil tabanlı karakter çıkarımı
for token in doc:
    if token.dep_ in ["nsubj", "nsubjpass"] and token.pos_ in ["NOUN", "PROPN", "ADJ"]:
        birlesik = tamlayici_zincir(token)
        if birlesik:
            if token.head.pos_ == "VERB" and token.head.lemma_ in insansi_eylemler and all(k not in turkce_stopwords for k in birlesik.split()):
                characters.add(birlesik)
        else:
            temiz_karakter = temizle_kelime(token.lemma_)
            if token.head.pos_ == "VERB" and token.head.lemma_ in insansi_eylemler and temiz_karakter not in turkce_stopwords and len(temiz_karakter) > 1:
                characters.add(temiz_karakter)

# Karakter sayma
karakter_sayaci = Counter()
for character in characters:
    pattern = rf"\b{re.escape(character)}[a-zçğıöşü]*\b"
    matches = re.findall(pattern, sentence.lower())
    karakter_sayaci[character] += len(matches)

# Benzerleri grupla
karakter_birlesmis = set()
karakter_final = {}
filtered_characters = [k for k, v in karakter_sayaci.items() if v >= 5]

for karakter in filtered_characters:
    if karakter in karakter_birlesmis:
        continue
    benzerler = get_close_matches(karakter, filtered_characters, n=100, cutoff=0.8)
    toplam = sum(karakter_sayaci[b] for b in benzerler)
    en_fazla = max(benzerler, key=lambda x: karakter_sayaci[x])
    karakter_final[en_fazla] = toplam
    karakter_birlesmis.update(benzerler)

karakter_final_saf = {}
karakterler_sorted = sorted(karakter_final.items(), key=lambda x: -len(x[0]))
for i, (karakter, sayi) in enumerate(karakterler_sorted):
    alt_parca_mi = False
    for j in range(i):
        if karakter in karakterler_sorted[j][0]:
            alt_parca_mi = True
            break
    if not alt_parca_mi:
        karakter_final_saf[karakter] = sayi

print("\nMasalda Geçen Karakterler ve Geçiş Sayıları:")
for karakter, sayi in sorted(karakter_final_saf.items(), key=lambda x: x[1], reverse=True):
    print(f"{karakter}: {sayi} kez geçti")

with open("karakterler.txt", "w", encoding="utf-8") as file:
    for karakter in karakter_final_saf.keys():
        file.write(karakter + "\n")

# -------- ÇÖZÜM 1: Fiil tabanlı karakter gruplama --------
karakter_eylemleri = defaultdict(set)

for sent in doc.sents:
    sent_text = sent.text.lower()
    for karakter in karakter_final_saf.keys():
        # Karakter cümlede geçiyor mu (kök veya köke yakın biçimiyle)
        if re.search(rf"\b{re.escape(karakter)}[a-zçğıöşü]*\b", sent_text):
            for token in sent:
                if token.pos_ == "VERB" and token.lemma_ in insansi_eylemler:
                    karakter_eylemleri[karakter].add(token.lemma_)

# Sonuçları yazdır
print("\nKarakterlerin geçtiği cümlelerdeki fiiller:")
for karakter, fiiller in karakter_eylemleri.items():
    print(f"\n🧍 {karakter}: {', '.join(sorted(fiiller))}")

# Benzer fiil içeren karakter grupları (isteğe bağlı)
benzer_gruplar = []
ziyaret_edilen = set()
karakter_listesi = list(karakter_eylemleri.items())

for i in range(len(karakter_listesi)):
    ad1, eylemler1 = karakter_listesi[i]
    if ad1 in ziyaret_edilen:
        continue
    grup = {ad1}
    for j in range(i + 1, len(karakter_listesi)):
        ad2, eylemler2 = karakter_listesi[j]
        if len(eylemler1 & eylemler2) >= 5:
            grup.add(ad2)
            ziyaret_edilen.add(ad2)
    if len(grup) > 1:
        benzer_gruplar.append(grup)
        ziyaret_edilen.update(grup)

print("\n--- Fiil tabanlı benzer karakter grupları ---")
for i, grup in enumerate(benzer_gruplar, 1):
    print(f"Grup {i}: {', '.join(grup)}")

with open("benzer_karakter_gruplari.txt", "w", encoding="utf-8") as f:
    for i, grup in enumerate(benzer_gruplar, 1):
        f.write(f"Grup {i}: {', '.join(grup)}\n")