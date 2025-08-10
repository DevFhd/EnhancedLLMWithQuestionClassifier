import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
import requests
import langdetect
from deep_translator import GoogleTranslator
import os

# ======= إعداد الموديلات =======
model_cat_path = "models/model_cat"
model_cat0_path = "models/model_cat0"

label_map1 = {
    0: "DESCRIPTION", 1: "ENTITY", 2: "ABBREVIATION", 3: "HUMAN",
    4: "NUMERIC", 5: "LOCATION"
}

label_map2 = {
    0: "manner", 1: "cremat", 2: "animal", 3: "exp", 4: "ind", 5: "gr", 6: "title",
    7: "def", 8: "date", 9: "reason", 10: "event", 11: "state", 12: "desc", 13: "count",
    14: "other", 15: "letter", 16: "religion", 17: "food", 18: "country", 19: "color",
    20: "termeq", 21: "city", 22: "body", 23: "dismed", 24: "mount", 25: "money",
    26: "product", 27: "period", 28: "substance", 29: "sport", 30: "plant", 31: "techmeth",
    32: "volsize", 33: "instru", 34: "abb", 35: "speed", 36: "word", 37: "lang",
    38: "perc", 39: "code", 40: "dist", 41: "temp", 42: "symbol", 43: "ord",
    44: "veh", 45: "weight", 46: "currency"
}

label_map1_ar = {
    0: "وصف", 1: "كيان", 2: "اختصار", 3: "شخص",
    4: "رقمي", 5: "مكان"
}

label_map2_ar = {
    0: "طريقة", 1: "موت", 2: "حيوان", 3: "تجربة", 4: "فرد", 5: "شجرة",
    6: "عنوان", 7: "تعريف", 8: "تاريخ", 9: "سبب", 10: "حدث", 11: "حالة",
    12: "وصف", 13: "عدد", 14: "أخرى", 15: "حرف", 16: "دين", 17: "طعام",
    18: "دولة", 19: "لون", 20: "مصطلح", 21: "مدينة", 22: "جسم", 23: "مرض",
    24: "جبل", 25: "مال", 26: "منتج", 27: "فترة", 28: "مادة", 29: "رياضة",
    30: "نبات", 31: "طريقة تقنية", 32: "حجم", 33: "أداة", 34: "اختصار",
    35: "سرعة", 36: "كلمة", 37: "لغة", 38: "نسبة", 39: "كود", 40: "مسافة",
    41: "درجة حرارة", 42: "رمز", 43: "ترتيب", 44: "مركبة", 45: "وزن", 46: "عملة"
}

@st.cache_resource
def load_models():
    tokenizer_cat = AutoTokenizer.from_pretrained(model_cat_path)
    model_cat = AutoModelForSequenceClassification.from_pretrained(model_cat_path)
    tokenizer_cat0 = AutoTokenizer.from_pretrained(model_cat0_path)
    model_cat0 = AutoModelForSequenceClassification.from_pretrained(model_cat0_path)
    return tokenizer_cat, model_cat, tokenizer_cat0, model_cat0

tokenizer_cat, model_cat, tokenizer_cat0, model_cat0 = load_models()

# ======= إعداد Ollama =======
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi"

def translate_to_english(text):
    return GoogleTranslator(source='auto', target='en').translate(text)

def query_ollama(question, cat1, cat2, original_question, lang):
    # رد خاص لو السؤال عن قصيدة صوت صفير البلبل
    ssb ="""صوت صفير البلبل\n
--------------\n
الاصمعي\n
--------------\n
صَــوْتُ صَـفـيرِ الـبُـلْبُلِ هَـيَّـجَ قَـلْـبِيَ الـثَـمِلِ\n
الــــــــمَـــــــاءُ وَالــــــــزَّهْـــــــرُ مَــــــــعَـــــــاً\n
مَــــــــــعَ زَهــــــــــرِ لَـــــحْــــظِ الـــمُـــقَـــلِ\n
وَأَنْـــــــــــــــتَ يَـــــاسَـــــيِّـــــدَ لِـــــــــــــــي\n
وَسَـــــــيِّـــــــدِي وَمَـــــــوْلَـــــــى لِـــــــــــــي\n
فَـــكَــمْ فَـــكَــمْ تَــيَّـمَـنِـي غُـــزَيِّــلٌ عَـقَـيْـقَـلـي\n
قَـطَّـفْـتُ مِــنْ وَجْـنَـتِهِ مِــنْ لَـثْـمِ وَرْدِ الـخَـجَلِ\n
فَــقَـالَ لاَ لاَ لاَ لاَ لاَ لاَ وَقَـــدْ غَـــدَا مُـهَـرْوِلِ\n
والـخُـودُ مَـالَـتْ طَـرَبَـاً مِــنْ فِـعْلِ هَـذَا الـرَّجُلِ\n
فَـوَلْـوَلَـتْ وَوَلْــوَلَـتُ وَلـــي وَلـــي يَــاوَيْـلَ لِــي\n
قَــالَـتْ لَـــهُ حِـيْـنَ كَــذَا انْـهَـضْ وَجِــدْ بِـالـنَّقَلِ\n
وَفِــتْــيَـةٍ سَـقَـوْنَـنِـي قَــهْــوَةً كَـالـعَـسَـلَ لِــــي\n
شَـمَـمْـتُـهَا بِــأَنْـفِـي أَزْكَــــى مِــــنَ الـقَـرَنْـفُلِ\n
فِــي وَسْــطِ بُـسْـتَانٍ حُـلِي بـالزَّهْرِ وَالـسُرُورُ لِـي\n
وَالـعُـودُ دَنْ دَنْــدَنَ لِـي وَالـطَّبْلُ طَـبْ طَـبَّلَ لِـي\n
طَــب طَــبِ طَـب طَـبِ طَـب طَـب طَـبَ لـي\n
وَالسَّقْفُ قَدْ سَقْسَقَ لِي وَالرَّقْصُ قَدْ طَابَ إلِي\n
شَـــوَى شَـــوَى وَشَـاهِـشُ عَـلَـى وَرَقْ سَـفَـرجَلِ\n
وَغَـــرَّدَ الـقَـمْـرُ يَـصِـيـحُ مِـــنْ مَـلَـلٍ فِــي مَـلَـلِ\n
فَـــلَــوْ تَــرَانِــي رَاكِــبــاً عَــلَــى حِــمَــارٍ أَهْــــزَلِ\n
يَــمْـشِـي عَــلَــى ثَــلاثَــةٍ كَـمَـشْـيَةِ الـعَـرَنْـجِلِ\n
وَالـنَّـاسُ تَـرْجِـمْ جَـمَـلِي فِــي الـسُـوقِ بـالقُلْقُلَلِ\n
وَالــكُـلُّ كَـعْـكَـعْ كَـعِـكَعْ خَـلْـفِي وَمِــنْ حُـوَيْـلَلِي\n
لـكِـنْ مَـشَـيتُ هَـارِبـا مِــنْ خَـشْـيَةِ الـعَـقَنْقِلِي\n
إِلَــــــى لِـــقَـــاءِ مَـــلِـــكٍ مُــعَــظَّــمٍ مُــبَــجَّــلِ\n
يَــأْمُــرُلِـي بِــخِـلْـعَـةٍ حَــمْــرَاءُ كَــالــدَّمْ دَمَــلِــي\n
أَجُــــــرُّ فِــيــهَـا مَــاشِــيـاً مُــبَــغْـدِدَاً لــلــذيَّـلِ\n
أَنَــا الأَدِيْـبُ الأَلْـمَعِي مِـنْ حَـيِّ أَرْضِ الـمُوْصِلِ\n
نَـظَـمْتُ قِـطَـعاً زُخْـرِفَتْ يَـعْجَزُ عَـنْهَا الأَدْبُ لِـي\n
أَقُــولُ فِــي مَـطْـلَعِهَا صَــوْتُ صَـفيرِ الـبُلْبُلِ\n"""
    if "صوت صفير البلبل" in original_question or "ما هي قصيدة صوت صفير البلبل" in original_question and lang == "العربية":
        return ssb
    elif  "صوت صفير البلبل" in original_question or "ما هي قصيدة صوت صفير البلبل" and lang == "English":
        return translate_to_english(ssb)
    else:
    # البرومبت العام
        prompt = f"""
You are an expert assistant. The user asked a question.

Question: {question}

The question belongs to these categories:
- Main category: {cat1}
- Subcategory: {cat2}

Answer clearly and informatively.
""".strip()

    # إرسال البرومبت للموديل
        payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
        }

        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=200)
            if response.status_code == 200:
                return response.json().get("response", "No response received.")
            else:
                return f"❌ Error {response.status_code}: {response.text}"
        except requests.exceptions.ConnectionError:
            return "❌ Ollama server not reachable. Make sure it's running."
        except Exception as e:
            return f"❌ Exception: {str(e)}"

# ======= وظائف مساعدة =======

def translate_to_arabic(text):
    return GoogleTranslator(source='auto', target='ar').translate(text)

def classify_cat1(question):
    inputs = tokenizer_cat(question, return_tensors="pt", truncation=True, padding=True)
    outputs = model_cat(**inputs)
    pred = torch.argmax(outputs.logits).item()
    return label_map1.get(pred, "UNKNOWN"), pred

def classify_cat2(question):
    inputs = tokenizer_cat0(question, return_tensors="pt", truncation=True, padding=True)
    outputs = model_cat0(**inputs)
    pred = torch.argmax(outputs.logits).item()
    return label_map2.get(pred, "UNKNOWN"), pred

# ======= نصوص متعددة اللغات =======
def get_text(key, lang):
    texts = {
        "title": {
            "العربية": "🚀 مساعد AI محسن",
            "English": "🚀 Enhanced AI assistant"
        },
        "subtitle": {
            "العربية": "أدخل سؤالك واحصل على اجابة محسنة من الذكاء الاصطناعي!",
            "English": "Enter your question and get enhanced AI answer!"
        },
        "chat_title": {
            "العربية": "💬 المحادثة",
            "English": "💬 Chat"
        },
        "input_label": {
            "العربية": "📝 اكتب سؤالك هنا",
            "English": "📝 Write your question here"
        },
        "ask_button": {
            "العربية": "💥 اسأل الآن",
            "English": "💥 Ask Now"
        },
        "reset_button": {
            "العربية": "🔄 إعادة التعيين",
            "English": "🔄 Reset"
        },
        "spinner_classify": {
            "العربية": "🔍 جار التصنيف...",
            "English": "🔍 Classifying..."
        },
        "spinner_ollama": {
            "العربية": "🤖 جار جلب الإجابة من Ollama...",
            "English": "🤖 Fetching answer from Ollama..."
        },
        "classification_label": {
            "العربية": "**📂 التصنيف:**",
            "English": "**📂 Classification:**"
        },
        "main_category": {
            "العربية": "- الفئة الرئيسية",
            "English": "- Main category"
        },
        "sub_category": {
            "العربية": "- الفئة الفرعية",
            "English": "- Subcategory"
        },
        "elapsed_time": {
            "العربية": "⏱️ الوقت",
            "English": "⏱️ Time"
        },
    }
    return texts.get(key, {}).get(lang, key)

# ======= تصميم الواجهة =======

st.set_page_config(page_title="AI Question Classifier", layout="wide")

lang_choice = st.radio("اختر لغة العرض / Choose display language:", ["العربية", "English"], index=0, horizontal=True)
direction = "rtl" if lang_choice == "العربية" else "ltr"

st.markdown(f"<h1 style='text-align: center; color: #ff4b5c;'>{get_text('title', lang_choice)}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: gray;'>{get_text('subtitle', lang_choice)}</p>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown(f"<div style='direction: {direction}; text-align: {'right' if direction=='rtl' else 'left'};'>{get_text('chat_title', lang_choice)}</div>", unsafe_allow_html=True)

for msg in st.session_state.messages:
    st.markdown(
        f"""
        <div style='direction:{direction}; text-align:center;
        background:{"#ff4b5c" if msg["role"]=="user" else "#293C50"};
        color:white; padding:10px; border-radius:12px;
        margin:10px auto; max-width:70%;'>
        {msg['text']}
        </div>
        """, unsafe_allow_html=True
    )

with st.form("chat_form"):
    st.markdown('<div id="form-wrapper">', unsafe_allow_html=True)
    user_input = st.text_area(get_text("input_label", lang_choice), height=80)

    col1, col_spacer, col2 = st.columns([3, 1, 3])

    with col1:
        reset = st.form_submit_button(get_text("reset_button", lang_choice))
    with col2:
        send = st.form_submit_button(get_text("ask_button", lang_choice), type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<script>
document.addEventListener('keydown', function(event) {
    if (event.ctrlKey && event.key === 'Enter') {
        const form = document.querySelector('#form-wrapper')?.closest('form');
        if (form) {
            form.requestSubmit();
        }
    }
});

const buttons = document.querySelectorAll("div.stButton > button");
buttons.forEach(btn => {
    btn.style.width = "100%";
    btn.style.padding = "12px 0";
    btn.style.fontSize = "18px";
    btn.style.borderRadius = "8px";
});
</script>
""", unsafe_allow_html=True)


if "pending_question" in st.session_state:
    question = st.session_state.pending_question
    del st.session_state.pending_question  # نحذفها عشان ما تعيد نفسها

    # ترجم السؤال إذا لازم
    original_question = question
    if lang_choice == "العربية" and not langdetect.detect(question) == 'en':
        question = translate_to_english(question)

    with st.spinner(get_text("spinner_classify", lang_choice)):
        cat1, cat1_idx = classify_cat1(question)
        cat2, cat2_idx = classify_cat2(question)
        if lang_choice == "العربية":
            cat1 = label_map1_ar.get(cat1_idx, cat1)
            cat2 = label_map2_ar.get(cat2_idx, cat2)

    with st.spinner(get_text("spinner_ollama", lang_choice)):
        start_time = time.time()
        answer = query_ollama(question, cat1, cat2, original_question, lang_choice)
        if lang_choice == "العربية":
            answer = translate_to_arabic(answer)
        elapsed = round(time.time() - start_time, 2)

    full_response = f"{answer}\n\n{get_text('classification_label', lang_choice)}\n{get_text('main_category', lang_choice)}: {cat1}\n{get_text('sub_category', lang_choice)}: {cat2}\n{get_text('elapsed_time', lang_choice)}: {elapsed} ثانية"

    st.session_state.messages.append({"role": "bot", "text": full_response})
    st.rerun()

if send and user_input.strip() != "":
    original_question = user_input.strip()
    question = original_question
    # أضف رسالة المستخدم فورًا
    st.session_state.messages.append({"role": "user", "text": original_question})
    # نحفظ السؤال في state عشان نستخدمه بعد إعادة التحميل
    st.session_state.pending_question = original_question
    st.rerun()

    # ترجمة السؤال للإنجليزية إذا اللغة العربية مختارة
    if lang_choice == "العربية" and not langdetect.detect(question) == 'en':
        question = translate_to_english(question)

    with st.spinner(get_text("spinner_classify", lang_choice)):
        cat1, cat1_idx = classify_cat1(question)
        cat2, cat2_idx = classify_cat2(question)

        # ترجمة التصنيفات للغة العربية إذا مختارة
        if lang_choice == "العربية":
            cat1 = label_map1_ar.get(cat1_idx, cat1)
            cat2 = label_map2_ar.get(cat2_idx, cat2)

    with st.spinner(get_text("spinner_ollama", lang_choice)):
        start_time = time.time()
        answer = query_ollama(question, cat1, cat2)
        # ترجمة الجواب للعربية إذا اللغة عربية
        if lang_choice == "العربية":
            answer = translate_to_arabic(answer)
        elapsed = round(time.time() - start_time, 2)
    second = "second" if lang_choice == "English" else "ثانية"
    full_response = f"{answer}\n\n{get_text('classification_label', lang_choice)}\n{get_text('main_category', lang_choice)}: {cat1}\n{get_text('sub_category', lang_choice)}:\n {cat2}\n{get_text('elapsed_time', lang_choice)}: {elapsed}" + second

    st.session_state.messages.append({"role": "user", "text": original_question})
    st.session_state.messages.append({"role": "bot", "text": full_response})
    st.rerun()

if reset:
    st.session_state.messages.clear()
    st.rerun()
