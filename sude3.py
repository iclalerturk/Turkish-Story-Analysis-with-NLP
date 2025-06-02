from collections import Counter
from difflib import get_close_matches
import re
import string
import spacy
from spacy import displacy
from nltk.tokenize import word_tokenize

nlp = spacy.load("tr_core_news_trf")  # GPU kullanımı için device=0

# İnsansı eylemler (fiiller) listesi
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

path = "NLP-Hikayeler\hikayeler\Hansel GretelV.txt"
with open(path, "r", encoding="utf-8") as file:
    sentence = file.read()

turkce_stopwords = {
        "bir", "bu", "şu", "çok", "daha", "mi", "mu", "mı", "mü", "şey", "tane",
        "ve","ile", "de", "da", "ki", "ne", "ni", "için", "ama", "fakat", "ancak",
        "gibi", "ise", "en", "ya", "iç", "herkes","kimse","kendi"," ","\"","“","“",
        "hem", "veya", "ya da", "çünkü", "lakin", "diye", "kadar", "sonra", "kişi",
        "önce", "her", "hiç", "hep", "bazı", "tüm", "hangi", "neden", "nasıl", "nerede", "ne zaman",
        "a", "na", "dan", "den", "e", "ye", "deki", "daki", "deki", "deki", "deki", "deki",
        "nin", "nin", "nın", "nun", "nün", "ki", "mı", "mi", "mu", "mü","içi",
        "ya", "ya da", "ya da", "ya da", "ya da", "ya da", "ya da", "ya da","ben", "sen", "o","ora","bura","şura","orada","burada","şurada","biri","biz","siz"
    }

# def temizle(metin):
   

#     # # Kesme işaretinden sonrasını at
#     # kelimeler = [kelime.split("'")[0] for kelime in kelimeler]
#     metin = metin.replace("’", "'")  
#     metin = metin.replace('“', '"').replace('”', '"')  # Çift tırnakları da düzelt

#     return ' '.join(metin.split()) # Fazla boşlukları temizle
#sentence = temizle(sentence)
# # Cümleyi Spacy ile işleyin
# with torch.amp.autocast(device_type="cuda"):
#sentence = temizle(sentence)

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
doc = nlp(sentence)

###### 
# Bütün tokenleri yazalım
# for token in doc:
#     print(f"{token.text:<15} --> HEAD: {token.head.text} | POS:({token.pos_}) --- DEP: {token.dep_}")
# # ########


# Karakterleri bulmak için bir liste oluştur
characters = set()
karakter_sayaci = Counter()

def temizle_kelime(kelime):
    return re.split(r"[’']", kelime)[0].lower()

# Spacy NER (kişiler)
for ent in doc.ents:
    temiz_karakter = temizle_kelime(ent.text)
    if ent.label_ == "PERSON" and temiz_karakter not in characters and len(ent) > 1:
        if temiz_karakter not in turkce_stopwords:
            print("PERSON:", temiz_karakter)
            characters.add(temiz_karakter)

def tamlayici_zincir(token):
    tamlayicilar = [token]
    stack = [token]

    while stack:
        current = stack.pop()
        for child in current.children:
            if child.dep_ in ["amod", "compound", "nmod:poss", "nmod"]:
                tamlayicilar.append(child)
                stack.append(child)  # onun çocuklarına da bak

    if len(tamlayicilar) == 1:
        return None
    
    # Cümledeki sıraya göre sıralıyoruz
    tamlayicilar = sorted(tamlayicilar, key=lambda x: x.i)
    return " ".join(t.text.lower() for t in tamlayicilar)


# Fiil özneleri ve birleşik isimler
for token in doc:
    if token.dep_ in ["nsubj", "nsubjpass"] and token.pos_ in ["NOUN", "PROPN", "ADJ"]:
        birlesik = tamlayici_zincir(token)
        if birlesik:
            if token.head.pos_ == "VERB" and token.head.lemma_ in insansi_eylemler and all(kelime not in turkce_stopwords for kelime in birlesik.split()):
                characters.add(birlesik)
        # Doğrudan token'ı alıyoruz
        else:
            # Doğrudan token'ı alıyoruz
            temiz_karakter = temizle_kelime(token.lemma_)
            # print("----------", temiz_karakter)
            if token.head.pos_ == "VERB" and token.head.lemma_ in insansi_eylemler and temiz_karakter not in characters:
                if temiz_karakter not in turkce_stopwords and len(temiz_karakter) > 1:
                    characters.add(temiz_karakter)
                    # print("++++++", temiz_karakter, token.head.lemma_)

print("Karakterler:")
for character in characters:
    print(character)


karakter_sayaci = Counter()

for character in characters:
    pattern = rf"\b{re.escape(character)}[a-zçğıöşü]*\b"
    matches = re.findall(pattern, sentence.lower())
    karakter_sayaci[character] += len(matches)

for karakter, sayi in sorted(karakter_sayaci.items(), key=lambda x: x[1], reverse=True):
    print(f"{karakter}: {sayi} kez geçti")

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

# Benzerlik sonrası alt parçaları eleyelim
karakter_final_saf = {}
karakterler_sorted = sorted(karakter_final.items(), key=lambda x: -len(x[0]))  # uzun karakterler önce

for i, (karakter, sayi) in enumerate(karakterler_sorted):
    alt_parca_mi = False
    for j in range(i):
        daha_uzun_karakter = karakterler_sorted[j][0]
        if karakter in daha_uzun_karakter:
            alt_parca_mi = True
            break
    if not alt_parca_mi:
        karakter_final_saf[karakter] = sayi

print("\nMasalda Geçen Karakterler ve Geçiş Sayıları (Alt parçalar elendi):")
for karakter, sayi in sorted(karakter_final_saf.items(), key=lambda x: x[1], reverse=True):
    print(f"{karakter}: {sayi} kez geçti")


# Karakterleri bir dosyaya yaz
with open("karakterler.txt", "w", encoding="utf-8") as file:
    for karakter in karakter_final_saf.keys():
        file.write(karakter + "\n")


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

