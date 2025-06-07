from collections import Counter, defaultdict
from difflib import get_close_matches
import re
import string
import spacy
from spacy import displacy
from nltk.tokenize import word_tokenize

class MetinAnaliz:
    def __init__(self,path_hikaye, model_name="tr_core_news_trf"):
        self.path=path_hikaye
        self.nlp = spacy.load(model_name)
        self.insansi_eylemler = {
            "duy", "sor", "sus", "git", "oyna", "hazırla", "paylaş", "tut", "ör", "söyle", "de", "konuş", 
            "düşün", "hisset", "hissed", "koyul", "kazan", "dokun", "tat", "yürü", "koş", "uyu", "uyan", 
            "gülümse", "ağla", "gül", "sev", "kork", "özlem", "iste", "hatırla", "unut", "kız", "öfkelen", 
            "öfke", "eğit", "uyar", "heyecanlan", "hüzünlen", "endişelen", "sevin", "üzül", "merak", "meraklan", 
            "kıskan", "ilgil", "sabırsızlan", "sakinleş", "kandır", "inandır", "cesaretlendir", "et", "incele", 
            "anla", "farket", "duygulan", "güven", "şüphelen", "şüphe", "destekle", "yargıla", "suçla", 
            "affet", "umutla", "eğlen", "rahatla", "çabala", "düşle", "şaşır", "şaş", "şevklen", "iste", 
            "koru", "gül", "ver", "yap", "bak", "öğren", "özle", "dinle", "planla", "arzula", "hedefle", "sor",
            "gözlemle", "çöz", "kararlaştır", "reddet", "dinlen", "korkut", "izle", "oku", "anlat", "çalış", "katıl",
            "yaz", "tartış", "hissizleş", "yürekglen", "yüreklendir", "gizle", "dile", "anlamlandır", "öv", 
            "döv", "eleştir", "sorgula", "haket", "diren", "vazgeç", "anlaş", "bağışla", "bağır", "mırıldan", "ye", 
            "çalış", "konuş", "al", "dile", "kız", "kır", "kırıl", "hastalan"
        }
        self.turkce_stopwords = {
        "bir", "bu", "şu", "çok", "daha", "mi", "mu", "mı", "mü", "şey", "tane",
        "ve","ile", "de", "da", "ki", "ne", "ni", "için", "ama", "fakat", "ancak",
        "gibi", "ise", "en", "ya", "iç", "herkes","kimse","kendi","\"","“"," ","",
        "hem", "veya", "ya da", "çünkü", "lakin", "diye", "kadar", "sonra", "kişi",
        "önce", "her", "hiç", "hep", "bazı", "tüm", "hangi", "neden", "nasıl", "nerede", "ne zaman",
        "a", "na", "dan", "den", "e", "ye", "deki", "daki", "deki", "deki", "deki", "deki",
        "nin", "nin", "nın", "nun", "nün", "ki", "mı", "mi", "mu", "mü","içi","ses",
        "ya", "ya da", "ya da", "ya da", "ya da", "ya da", "ya da", "ya da","ben", "sen", "o","ora","bura","şura","orada","burada","şurada","biri","biz","siz"
        }
        self.characters = set()
        self.character_counter = Counter()
        self.sentences_by_character = defaultdict(list)
        self.groups = []

    # def clean_text(self, metin):
    #     metin = metin.lower()
    #     metin = metin.translate(str.maketrans('', '', string.punctuation.replace("'", "")))
    #     metin = re.sub(r'\d+', '', metin)
    #     kelimeler = word_tokenize(metin, language='turkish')
    #     kelimeler = [kelime.split("'")[0] for kelime in kelimeler]
    #     return ' '.join(kelimeler)

    def text_preprocess(self,text):
        text = text.replace("’", "'")  # ’ işaretini ' dönüştür
        text = text.replace("“", '"')  #  “ işaretini " dönüştür
        text = text.replace("”", '"')  #  ” işaretini " dönüştür
        text = re.sub(r"'(\s+)", "'", text)  # kesme işaretii sonrası boşluk varsa at
        return text

    def clean_word(self, word):
        return re.split(r"'", word)[0].lower()
    
    
    def get_modifier_chain(self,token):
        modifiers = []
        stack = [token]

        while stack:
            current = stack.pop()
            if current not in modifiers:
                modifiers.append(current)
            for child in current.children:
                if child.dep_ in ["amod", "compound", "nmod:poss", "nmod", "flat"]:
                    stack.append(child)

        if len(modifiers) >= 1:
            modifiers = sorted(modifiers, key=lambda x: x.i)
            # last_word = modifiers[-1].text.lower()
            # oncekiler = [t.text.lower() for t in modifiers[:-1]]
            # return " ".join(oncekiler + [last_word])
            return " ".join([t.text.lower() for t in modifiers])
        else:
            return None
        
    def is_sublist_or_subset(self, small, big):
        for i in range(len(big) - len(small) + 1):
            if big[i:i+len(small)] == small:
                return True
        return set(small).issubset(set(big))

    def get_head_word(self, phrase):
        doc = self.nlp(phrase)
        if len(doc) == 0:
            return None
        return doc[-1].lemma_.lower()

    def should_add_to_group(self, modifier, group_modifier):
        return self.is_sublist_or_subset(modifier, group_modifier) or self.is_sublist_or_subset(group_modifier, modifier)

    def create_groups(self,phrase_list):
        subgroups = []
        for phrase in phrase_list:
            words = phrase.split()
            modifier = words[:-1]

            added = False
            for grup in subgroups:
                grup_modifier = grup[0].split()[:-1]
                if self.should_add_to_group(modifier, grup_modifier):
                    grup.append(phrase)
                    added = True
                    break

            if not added:
                subgroups.append([phrase])
        return subgroups

    def group_similars_by_head_word(self,phrases, nlp):
        head_group_dict = defaultdict(list)
        for phrase in phrases:
            head = self.get_head_word(phrase)
            head_group_dict[head].append(phrase)

        groups = []
        for head_lemma, phrase_list in head_group_dict.items():
            phrase_list = sorted(phrase_list, key=lambda x: (len(x.split()), x))
            subgroups = self.create_groups(phrase_list)

            for group in subgroups:
                character = min([x for x in group if len(x.split()) == max(len(y.split()) for y in group)], key=lambda x: len(x))  #kelime sayısı en fazla olanlar arasından en kısası
                groups.append({
                    "cekirdek": head_lemma,
                    "karakter": character,
                    "gruplar": [group]
                })

        return groups    

    def merge_single_word_character(self, group1_index, character1, matched_characters, copy, visited_groups, character_count, group1):
        for j, group2 in enumerate(copy):
            if j in visited_groups or group1_index == j:
                continue

            character2 = group2["karakter"]
            if character1 in character2.split():  # karakter1, karakter2'nin içinde geçiyorsa
                matched_characters.append((j, character2, character_count.get(character2, 0)))

        if matched_characters:   # eşleşen karakter varsa, en çok geçen karaktere ekle
            most_frequent = max(matched_characters, key=lambda x: x[2])  # en çok geçiş sayısına sahip karakteri bul
            j, character2, _ = most_frequent
                    
            copy[j]["gruplar"][0].extend(group1["gruplar"][0])  # karakter2'nin grubuna karakter1'i ekle
            character_count[character2] = character_count.get(character2, 0) + character_count.get(character1, 0)  # karakter2'nin geçiş sayısına karakter1'in geçiş sayısını ekle
            visited_groups.add(group1_index)

    def merge_similar_names_by_threshold(self, group1_index, character1, copy, visited_groups, character_count, merged_group, all_similars, threshold):
        for j in range(group1_index + 1, len(copy)):
                if j in visited_groups:
                    continue

                group2 = copy[j]
                character2 = group2["karakter"]

                if character1 != character2 and get_close_matches(character1, [character2], cutoff=threshold):
                    merged_group.extend(group2["gruplar"][0])
                    all_similars.append((character2, character_count.get(character2, 0)))
                    visited_groups.add(j)
            
    def merge_similar_named_groups(self, groups, character_count, threshold):
        new_groups = []
        groups_copy = groups.copy()
        visited_groups = set()

        for i, group1 in enumerate(groups_copy):   # önce tek kelimelik karakterleri kontrol et
            if i in visited_groups:
                continue

            character1 = group1["karakter"]
            if len(character1.split()) == 1:   # karakter1 tek kelimeden oluşuyorsa
                matched_characters = []
                self.merge_single_word_character(i, character1, matched_characters, groups_copy, visited_groups, character_count, group1)

        for i, group1 in enumerate(groups_copy):  # normal benzerlik kontrolü
            if i in visited_groups:
                continue

            character1 = group1["karakter"]
            merged_group = group1["gruplar"][0].copy()
            all_similars = [(character1, character_count.get(character1, 0))]

            self.merge_similar_names_by_threshold(i, character1, groups_copy, visited_groups, character_count, merged_group, all_similars, threshold)

            most_frequently_mentioned_char = max(all_similars, key=lambda x: x[1])[0]  # en çok geçen karakteri seç
            total_mention = sum([sayi for _, sayi in all_similars])

            new_groups.append({
                "cekirdek": self.get_head_word(most_frequently_mentioned_char),
                "karakter": most_frequently_mentioned_char,
                "gruplar": [list(set(merged_group))],
                "geçiş_sayısı": total_mention
            })

        return new_groups
    
    def _extract_ner_characters(self, doc):
        for ent in doc.ents:
            clean_char = self.clean_word(ent.text)
            if ent.label_ == "PERSON" and len(ent.text) > 1:
                if clean_char not in self.characters and clean_char not in self.turkce_stopwords:
                    self.characters.add(clean_char)


    def _extract_subject_characters(self, doc):
        for token in doc:
            if token.dep_ in ["nsubj", "nsubjpass"] and token.pos_ in ["NOUN", "PROPN", "ADJ"]:
                verb = token.head
                if verb.pos_ == "VERB" and verb.lemma_ in self.insansi_eylemler:
                    phrase = self.get_modifier_chain(token)
                    if phrase:
                        if all(word not in self.turkce_stopwords for word in phrase.split()):
                            self.characters.add(phrase)
                    else:
                        clean_char = self.clean_word(token.lemma_)
                        if (clean_char not in self.characters and clean_char not in self.turkce_stopwords and len(clean_char) > 1):
                            self.characters.add(clean_char)
    
    def _assign_sentences_to_groups(self, doc, group, sentences):
        main_char = group["karakter"]
        alternatives = set()
        for sub_group in group["gruplar"]:
            alternatives.update(sub_group)
        alternatives = sorted(alternatives, key=lambda x: -len(x))

        patterns = [r'\b' + re.escape(alt) + r'([a-zçğıöşü]{0,6})?\b' for alt in alternatives]

        for sent in sentences:
            sent_text = sent.text
            sent_lower = sent_text.lower()
            candidates = []

            for token in sent:
                if token.pos_ in ["NOUN", "PROPN", "ADJ"]:
                    raw = self.get_modifier_chain(token) or token.text
                    cleaned = self.clean_word(raw)
                    candidates.append(cleaned)

            for pattern in patterns:
                if any(re.search(pattern, candidate) for candidate in candidates):
                    self.sentences_by_character[main_char].append(sent_text.strip())
                    self.character_counter[main_char] += len(re.findall(pattern, sent_lower))
                    break


    def tum_islemler(self):
        with open(self.path, "r", encoding="utf-8") as file:
            sentence = file.read()

        sentence = self.text_preprocess(sentence)
        doc = self.nlp(sentence)


        # # Spacy NER (kişiler)
        # for ent in doc.ents:
        #     temiz_karakter = self.clean_word(ent.text)
        #     # print("1", temiz_karakter)
        #     if ent.label_ == "PERSON" and temiz_karakter not in self.characters and len(ent.text) > 1:
        #         # print("2",temiz_karakter)
        #         if temiz_karakter not in self.turkce_stopwords:
        #             print("3",temiz_karakter)
        #             self.characters.add(temiz_karakter)

        # # Fiil özneleri ve birleşik isimler
        # for token in doc:
        #     if token.dep_ in ["nsubj", "nsubjpass"] and token.pos_ in ["NOUN", "PROPN", "ADJ"]:
        #         birlesik = self.get_modifier_chain(token)
        #         if birlesik:
        #             if token.head.pos_ == "VERB" and token.head.lemma_ in self.insansi_eylemler and all(kelime not in self.turkce_stopwords for kelime in birlesik.split()):
        #                 self.characters.add(birlesik)
        #                 # print("!!!!!!!!!!!!!!!!!!!!!!", birlesik, token.text)
        #                 # print("Cümle:", token.sent.text.strip())  # ← burada cümle yazdırılır
        #         # Doğrudan token'ı alıyoruz
        #         else:
        #             # Doğrudan token'ı alıyoruz
        #             temiz_karakter = self.clean_word(token.lemma_)
        #             # print("----------", temiz_karakter)
        #             if token.head.pos_ == "VERB" and token.head.lemma_ in self.insansi_eylemler and temiz_karakter not in self.characters:
        #                 if temiz_karakter not in self.turkce_stopwords and len(temiz_karakter) > 1:
        #                     self.characters.add(temiz_karakter)
        #                     # print("++++++", temiz_karakter, token.lemma_, token.head.lemma_)

        self._extract_ner_characters(doc)
        self._extract_subject_characters(doc)

        print("Karakterler:")
        for character in self.characters:
            print(character)
        print("--------------------")

        karakter_listesi = list(self.characters)

        groups = self.group_similars_by_head_word(karakter_listesi, self.nlp)

        
        print("\nGruplandırılmış Karakter Tamlamaları (Çekirdek Kelimeye Göre):")
        for i, group in enumerate(groups, 1):
            print(f"Grup {i}: {group}")

        # duzenlenmis_metin = sentence
        sentences = list(doc.sents)

        # for grup in gruplar:
        #     ana_karakter = grup["karakter"]
            
        #     # Tüm alternatif ifadeleri topla
        #     alternatifler = set()
        #     for alt_grup in grup["gruplar"]:
        #         alternatifler.update(alt_grup)
            
        #     # En uzun eşleşmeler öncelikli
        #     alternatifler = sorted(alternatifler, key=lambda x: -len(x))
        #     patternler = [r'\b' + re.escape(ifade) + r'([a-zçğıöşü]{0,6})?\b' for ifade in alternatifler]

        #     for sent in cümleler:
        #         sent_text = sent.text
        #         sent_lower = sent_text.lower()

        #         # Aday karakter bul
        #         adaylar = []
        #         for token in sent:
        #             if token.pos_ in ["NOUN", "PROPN", "ADJ"]:
        #                 aday_raw = self.get_modifier_chain(token) or token.text
        #                 temiz_aday = self.clean_word(aday_raw)
        #                 adaylar.append(temiz_aday)

        #         eşleşti_mi = False
        #         for pattern in patternler:
        #             for aday in adaylar:
        #                 if re.search(pattern, aday):
        #                     self.sentences_by_character[ana_karakter].append(sent_text.strip())
        #                     self.character_counter[ana_karakter] += len(re.findall(pattern, sent_lower))
        #                     eşleşti_mi = True
        #                     break
        #             if eşleşti_mi:
        #                 break

        for group in groups:
            self._assign_sentences_to_groups(doc, group, sentences)


        # Yazdırma kısmı
        print("\nKarakter Geçiş Sayıları:")
        for character, count in self.character_counter.items():
            print(f"{character}: {count}")

        groups = self.merge_similar_named_groups(groups, self.character_counter, threshold=0.7500001)
        print("\nbirleştirme:")
        for i, group in enumerate(groups, 1):
            print(f"Grup {i}: {group}")

        print("\nBirleştirilmiş Gruplar (Geçiş Sayılarıyla):")
        for i, group in enumerate(groups, 1):
            print(f"Grup {i}: Karakter = {group['karakter']} | Geçiş Sayısı = {group['geçiş_sayısı']}")
            print(f"  → Gruplar: {group['gruplar'][0]}")

        # Geçiş sayısına göre azalan şekilde sırala ve yazdır
        print("\nBirleştirilmiş Gruplar (Geçiş Sayılarıyla, Sıralı):")
        sirali_gruplar = sorted([g for g in groups if g["geçiş_sayısı"] >= 5], key=lambda x: x["geçiş_sayısı"], reverse=True)
        for i, group in enumerate(sirali_gruplar, 1):
            print(f"Karakter = {group['karakter']} | Geçiş Sayısı = {group['geçiş_sayısı']}")

        convert_sorted_groups = []
        for group in sirali_gruplar:
            character = group["karakter"]
            mention = group["geçiş_sayısı"]
            convert_sorted_groups.append((character, mention))


        with open("karakterler.txt", "w", encoding="utf-8") as f:
            for group in groups:
                if group["geçiş_sayısı"] >= 5:
                    ana_karakter = group["karakter"]
                    alternatifler = list(set(group["gruplar"][0]) - {ana_karakter})
                    satir = ",".join([ana_karakter] + alternatifler)
                    f.write(satir + "\n")    
        
        return convert_sorted_groups
    
# if __name__ == "__main__":
#     analiz = MetinAnaliz('NLP-Hikayeler\hikayeler\zehrakisahikaye.txt')
#     analiz.tum_islemler()
   