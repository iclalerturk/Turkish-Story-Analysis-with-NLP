from collections import Counter
from collections import defaultdict
from difflib import get_close_matches
import re
import string
import spacy
from spacy import displacy
from nltk.tokenize import word_tokenize

nlp = spacy.load("tr_core_news_trf")  # GPU kullanımı için device=0

path = "NLP-Hikayeler\hikayeler\\zehrakisahikaye.txt"
with open(path, "r", encoding="utf-8") as file:
    sentence = file.read()

# İnsansı eylemler (fiiller) listesi
insansi_eylemler = {
    "duy", "sor", "sus", "git", "oyna", "hazırla", "paylaş", "tut", "ör", "söyle", "de", "konuş", 
    "düşün", "hisset", "hissed", "koyul", "kazan", "dokun", "tat", "yürü", "koş", "uyu", "uyan", 
    "gülümse", "ağla", "gül", "sev", "kork", "özlem", "iste", "hatırla", "unut", "kız", "öfkelen", 
    "öfke", "eğit", "uyar", "heyecanlan", "hüzünlen", "endişelen", "sevin", "üzül", "merak", "meraklan", 
    "kıskan", "ilgil", "sabırsızlan", "sakinleş", "kandır", "inandır", "cesaretlendir", "et", "incele", 
    "anla", "farket", "duygulan", "güven", "şüphelen", "şüphe", "destekle", "yargıla", "suçla", 
    "affet", "umutla", "eğlen", "rahatla", "çabala", "düşle", "şaşır", "şaş", "şevklen", "iste", 
    "koru", "gül", "ver", "yap", "bak", "öğren", "özle", "dinle", "planla", "arzula", "hedefle", "sor",
    "gözlemle", "çöz", "kararlaştır", "reddet", "dinlen", "korkut", "izle", "oku", "anlat", "çalış","katıl",
    "yaz", "tartış", "hissizleş", "yürekglen", "yüreklendir", "gizle", "dile", "anlamlandır", "öv", 
    "döv", "eleştir", "sorgula", "haket", "diren", "vazgeç", "anlaş", "bağışla", "bağır", "mırıldan","ye", 
    "çalış", "konuş", "al", "dile", "kız", "kır", "kırıl", "hastalan"
}

turkce_stopwords = {
        "bir", "bu", "şu", "çok", "daha", "mi", "mu", "mı", "mü", "şey", "tane",
        "ve","ile", "de", "da", "ki", "ne", "ni", "için", "ama", "fakat", "ancak",
        "gibi", "ise", "en", "ya", "iç", "herkes","kimse","kendi","\"","“"," ","",
        "hem", "veya", "ya da", "çünkü", "lakin", "diye", "kadar", "sonra", "kişi",
        "önce", "her", "hiç", "hep", "bazı", "tüm", "hangi", "neden", "nasıl", "nerede", "ne zaman",
        "a", "na", "dan", "den", "e", "ye", "deki", "daki", "deki", "deki", "deki", "deki",
        "nin", "nin", "nın", "nun", "nün", "ki", "mı", "mi", "mu", "mü","içi","ses",
        "ya", "ya da", "ya da", "ya da", "ya da", "ya da", "ya da", "ya da","ben", "sen", "o","ora","bura","şura","orada","burada","şurada","biri","biz","siz"
    }

def temizle(metin):
    # Temel temizleme
    metin = metin.lower()
    metin = metin.translate(str.maketrans('', '', string.punctuation.replace("'", "")))  # Noktalama kaldır, ama kesme işaretini bırak
    metin = re.sub(r'\d+', '', metin)
    kelimeler = word_tokenize(metin, language='turkish')

    # Kesme işaretinden sonrasını at
    kelimeler = [kelime.split("'")[0] for kelime in kelimeler]

    return ' '.join(kelimeler)

def metin_onisleme(text):
    # ’ işaretini ' dönüştür
    text = text.replace("’", "'")
    #  “ işaretini " dönüştür
    text = text.replace("“", '"')
    #  ” işaretini " dönüştür
    text = text.replace("”", '"')
    # kesme işaretii sonrası boşluk varsa at
    text = re.sub(r"'(\s+)", "'", text)
    return text

sentence = metin_onisleme(sentence)
# sentence = sentence.lower()
doc = nlp(sentence)

# Karakterleri bulmak için bir liste oluştur
characters = set()


def temizle_kelime(kelime):
    return re.split(r"'", kelime)[0].lower()


# Spacy NER (kişiler)
for ent in doc.ents:
    temiz_karakter = temizle_kelime(ent.text)
    # print("1", temiz_karakter)
    if ent.label_ == "PERSON" and temiz_karakter not in characters and len(ent.text) > 1:
        # print("2",temiz_karakter)
        if temiz_karakter not in turkce_stopwords:
            print("3",temiz_karakter)
            characters.add(temiz_karakter)

# print("Karakterler:")
# for character in characters:
#     print(character)
# print("--------------------")

def tamlayici_zincir(token):
    tamlayicilar = []
    stack = [token]

    while stack:
        current = stack.pop()
        if current not in tamlayicilar:
            tamlayicilar.append(current)
        for child in current.children:
            if child.dep_ in ["amod", "compound", "nmod:poss", "nmod", "flat"]:
                stack.append(child)

    if len(tamlayicilar) >= 1:
        tamlayicilar = sorted(tamlayicilar, key=lambda x: x.i)
        son_kelime = tamlayicilar[-1].text.lower()
        oncekiler = [t.text.lower() for t in tamlayicilar[:-1]]
        return " ".join(oncekiler + [son_kelime])
    else:
        return None

# Fiil özneleri ve birleşik isimler
for token in doc:
    if token.dep_ in ["nsubj", "nsubjpass"] and token.pos_ in ["NOUN", "PROPN", "ADJ"]:
        birlesik = tamlayici_zincir(token)
        if birlesik:
            if token.head.pos_ == "VERB" and token.head.lemma_ in insansi_eylemler and all(kelime not in turkce_stopwords for kelime in birlesik.split()):
                characters.add(birlesik)
                # print("!!!!!!!!!!!!!!!!!!!!!!", birlesik, token.text)
                # print("Cümle:", token.sent.text.strip())  # ← burada cümle yazdırılır
        # Doğrudan token'ı alıyoruz
        else:
            # Doğrudan token'ı alıyoruz
            temiz_karakter = temizle_kelime(token.lemma_)
            # print("----------", temiz_karakter)
            if token.head.pos_ == "VERB" and token.head.lemma_ in insansi_eylemler and temiz_karakter not in characters:
                if temiz_karakter not in turkce_stopwords and len(temiz_karakter) > 1:
                    characters.add(temiz_karakter)
                    # print("++++++", temiz_karakter, token.lemma_, token.head.lemma_)

print("Karakterler:")
for character in characters:
    print(character)
print("--------------------")


###########################################################
# Spacy vb sonrası karakter seti hazırlandı
# Senin karakter çıkarma kısmın buraya gelecek (önce karakterleri topladın)

karakter_listesi = list(characters)
def is_sublist_or_subset(small, big):
    for i in range(len(big) - len(small) + 1):
        if big[i:i+len(small)] == small:
            return True
    return set(small).issubset(set(big))

def get_head_word(tamlama, nlp):
    doc = nlp(tamlama)
    if len(doc) == 0:
        return None
    return doc[-1].lemma_.lower()

def grup_ekle_mi(modifier, grup_modifier):
    return is_sublist_or_subset(modifier, grup_modifier) or is_sublist_or_subset(grup_modifier, modifier)

def gruplari_olustur(tamlamalar_listesi):
    alt_gruplar = []
    for tamlama in tamlamalar_listesi:
        kelimeler = tamlama.split()
        modifier = kelimeler[:-1]

        eklendi = False
        for grup in alt_gruplar:
            grup_modifier = grup[0].split()[:-1]
            if grup_ekle_mi(modifier, grup_modifier):
                grup.append(tamlama)
                eklendi = True
                break

        if not eklendi:
            alt_gruplar.append([tamlama])
    return alt_gruplar

def benzerleri_grupla_cekirdek_kelime(tamlamalar, nlp):
    cekirdek_grup_dict = defaultdict(list)
    for tamlama in tamlamalar:
        cekirdek = get_head_word(tamlama, nlp)
        cekirdek_grup_dict[cekirdek].append(tamlama)

    tum_gruplar = []
    for cekirdek_lemma, tamlama_listesi in cekirdek_grup_dict.items():
        tamlama_listesi = sorted(tamlama_listesi, key=lambda x: (len(x.split()), x))
        alt_gruplar = gruplari_olustur(tamlama_listesi)

        for grup in alt_gruplar:
            karakter = min([x for x in grup if len(x.split()) == max(len(y.split()) for y in grup)], key=lambda x: len(x))  #kelime sayısı en fazla olanlar arasından en kısası
            tum_gruplar.append({
                "cekirdek": cekirdek_lemma,
                "karakter": karakter,
                "gruplar": [grup]
            })

    return tum_gruplar
import re
from difflib import get_close_matches
def grupla_benzerlik_birleştir(gruplar, karakter_sayilari, threshold=0.8):
    yeni_gruplar = []
    gruplar_kopya = gruplar.copy()
    ziyaret_edilen = set()

    # Önce tek kelimelik karakterleri kontrol et
    for i, grup1 in enumerate(gruplar_kopya):
        if i in ziyaret_edilen:
            continue

        karakter1 = grup1["karakter"]
        # Eğer karakter tek kelimeden oluşuyorsa
        if len(karakter1.split()) == 1:
            eslesen_karakterler = []
            for j, grup2 in enumerate(gruplar_kopya):
                if j in ziyaret_edilen or i == j:
                    continue

                karakter2 = grup2["karakter"]
                # Eğer karakter1, karakter2'nin içinde geçiyorsa
                if karakter1 in karakter2.split():
                    eslesen_karakterler.append((j, karakter2, karakter_sayilari.get(karakter2, 0)))

            # Eğer eşleşen karakter varsa, en çok geçen karaktere ekle
            if eslesen_karakterler:
                # En çok geçiş sayısına sahip karakteri bul
                en_cok_gecen = max(eslesen_karakterler, key=lambda x: x[2])
                j, karakter2, _ = en_cok_gecen
                
                # Karakter2'nin grubuna karakter1'i ekle
                gruplar_kopya[j]["gruplar"][0].extend(grup1["gruplar"][0])
                # Karakter2'nin geçiş sayısına karakter1'in geçiş sayısını ekle
                karakter_sayilari[karakter2] = karakter_sayilari.get(karakter2, 0) + karakter_sayilari.get(karakter1, 0)
                ziyaret_edilen.add(i)

    # Normal benzerlik kontrolü
    for i, grup1 in enumerate(gruplar_kopya):
        if i in ziyaret_edilen:
            continue

        karakter1 = grup1["karakter"]
        grup_birlesik = grup1["gruplar"][0].copy()
        tum_benzerler = [(karakter1, karakter_sayilari.get(karakter1, 0))]

        for j in range(i + 1, len(gruplar_kopya)):
            if j in ziyaret_edilen:
                continue

            grup2 = gruplar_kopya[j]
            karakter2 = grup2["karakter"]

            if karakter1 != karakter2 and get_close_matches(karakter1, [karakter2], cutoff=threshold):
                grup_birlesik.extend(grup2["gruplar"][0])
                tum_benzerler.append((karakter2, karakter_sayilari.get(karakter2, 0)))
                ziyaret_edilen.add(j)

        # En çok geçen karakteri seç
        en_fazla_gecen_karakter = max(tum_benzerler, key=lambda x: x[1])[0]
        toplam_geçiş = sum([sayi for _, sayi in tum_benzerler])

        yeni_gruplar.append({
            "cekirdek": get_head_word(en_fazla_gecen_karakter, nlp),
            "karakter": en_fazla_gecen_karakter,
            "gruplar": [list(set(grup_birlesik))],
            "geçiş_sayısı": toplam_geçiş
        })

    return yeni_gruplar




gruplar = benzerleri_grupla_cekirdek_kelime(karakter_listesi, nlp)

print("\nGruplandırılmış Karakter Tamlamaları (Çekirdek Kelimeye Göre):")
for i, grup in enumerate(gruplar, 1):
    print(f"Grup {i}: {grup}")



from collections import Counter, defaultdict
import re

karakter_geçiş_sayacı = Counter()
karakter_cümleleri = defaultdict(list)
duzenlenmis_metin = sentence
cümleler = list(doc.sents)

for grup in gruplar:
    ana_karakter = grup["karakter"]
    
    # Tüm alternatif ifadeleri topla
    alternatifler = set()
    for alt_grup in grup["gruplar"]:
        alternatifler.update(alt_grup)
    
    # En uzun eşleşmeler öncelikli
    alternatifler = sorted(alternatifler, key=lambda x: -len(x))
    patternler = [r'\b' + re.escape(ifade) + r'([a-zçğıöşü]{0,6})?\b' for ifade in alternatifler]

    for sent in cümleler:
        sent_text = sent.text
        sent_lower = sent_text.lower()

        # Aday karakter bul
        adaylar = []
        for token in sent:
            if token.pos_ in ["NOUN", "PROPN", "ADJ"]:
                aday_raw = tamlayici_zincir(token) or token.text
                temiz_aday = temizle_kelime(aday_raw)
                adaylar.append(temiz_aday)

        eşleşti_mi = False
        for pattern in patternler:
            for aday in adaylar:
                if re.search(pattern, aday):
                    karakter_cümleleri[ana_karakter].append(sent_text.strip())
                    karakter_geçiş_sayacı[ana_karakter] += len(re.findall(pattern, sent_lower))
                    eşleşti_mi = True
                    break
            if eşleşti_mi:
                break

# Yazdırma kısmı
print("\nKarakter Geçiş Sayıları:")
for karakter, sayı in karakter_geçiş_sayacı.items():
    print(f"{karakter}: {sayı}")

gruplar= grupla_benzerlik_birleştir(gruplar, karakter_geçiş_sayacı, threshold=0.7500001)
print("\nbirleştirme:")
for i, grup in enumerate(gruplar, 1):
    print(f"Grup {i}: {grup}")

print("\nBirleştirilmiş Gruplar (Geçiş Sayılarıyla):")
for i, grup in enumerate(gruplar, 1):
    print(f"Grup {i}: Karakter = {grup['karakter']} | Geçiş Sayısı = {grup['geçiş_sayısı']}")
    print(f"  → Gruplar: {grup['gruplar'][0]}")

# Geçiş sayısına göre azalan şekilde sırala ve yazdır
print("\nBirleştirilmiş Gruplar (Geçiş Sayılarıyla, Sıralı):")
sirali_gruplar = sorted([g for g in gruplar if g["geçiş_sayısı"] >= 5], key=lambda x: x["geçiş_sayısı"], reverse=True)
for i, grup in enumerate(sirali_gruplar, 1):
    print(f"Karakter = {grup['karakter']} | Geçiş Sayısı = {grup['geçiş_sayısı']}")


with open("karakterler.txt", "w", encoding="utf-8") as f:
    for grup in gruplar:
        if grup["geçiş_sayısı"] >= 5:
            ana_karakter = grup["karakter"]
            alternatifler = list(set(grup["gruplar"][0]) - {ana_karakter})
            satir = ",".join([ana_karakter] + alternatifler)
            f.write(satir + "\n")
