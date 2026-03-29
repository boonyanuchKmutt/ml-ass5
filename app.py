import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="US Crime Rate Predictor", page_icon="🔍", layout="wide")
st.title("🔍 US Crime Rate Predictor")
st.markdown("""This app uses a **Logistic Regression model trained with SMOTE oversampling** 
to predict whether a community has a **high violent crime rate**.""")

FEATURE_NAMES = [
    "population", "householdsize", "racepctblack", "racePctWhite", "racePctAsian",
    "racePctHisp", "agePct12t21", "agePct12t29", "agePct16t24", "agePct65up",
    "numbUrban", "pctUrban", "medIncome", "pctWWage", "pctWFarmSelf",
    "pctWInvInc", "pctWSocSec", "pctWPubAsst", "pctWRetire", "medFamInc",
    "perCapInc", "whitePerCap", "blackPerCap", "indianPerCap", "AsianPerCap",
    "OtherPerCap", "HispPerCap", "NumUnderPov", "PctPopUnderPov", "PctLess9thGrade",
    "PctNotHSGrad", "PctBSorMore", "PctUnemployed", "PctEmploy", "PctEmplManu",
    "PctEmplProfServ", "PctOccupMgmtProf", "MalePctDivorce", "MalePctNevMarr", "FemalePctDiv",
    "TotalPctDiv", "PersPerFam", "PctFam2Par", "PctKids2Par", "PctYoungKids2Par",
    "PctTeen2Par", "PctWorkMomYoungKids", "PctWorkMom", "NumIlleg", "PctIlleg",
    "NumImmig", "PctImmigRecent", "PctImmigRec5", "PctImmigRec8", "PctImmigRec10",
    "PctRecentImmig", "PctRecImmig5", "PctRecImmig8", "PctRecImmig10", "PctSpeakEnglOnly",
    "PctNotSpeakEnglWell", "PctLargHouseFam", "PctLargHouseOccup", "PersPerOccupHous", "PersPerOwnOccHous",
    "PersPerRentOccHous", "PctPersOwnOccup", "PctPersDenseHous", "PctHousLess3BR", "MedNumBR",
    "HousVacant", "PctHousOccup", "PctHousOwnOcc", "PctVacantBoarded", "PctVacMore6Mos",
    "MedYrHousBuilt", "PctHousNoPhone", "PctWOFullPlumb", "OwnOccLowQuart", "OwnOccMedVal",
    "OwnOccHiQuart", "RentLowQ", "RentMedian", "RentHighQ", "MedRent",
    "MedRentPctHousInc", "MedOwnCostPctInc", "MedOwnCostPctIncNoMtg", "NumInShelters", "NumStreet",
    "PctForeignBorn", "PctBornSameState", "PctSameHouse85", "PctSameCity85", "PctSameState85",
    "LemasSwornFT", "LemasSwFTPerPop", "LemasSwFTFieldOps", "LemasSwFTFieldPerPop", "LemasTotalReq",
]

FEATURE_DESCRIPTIONS = {
    "population": "จำนวนประชากรทั้งหมด",
    "householdsize": "ขนาดครัวเรือนเฉลี่ย",
    "racepctblack": "% ประชากรผิวดำ",
    "racePctWhite": "% ประชากรผิวขาว",
    "racePctAsian": "% ประชากรเอเชีย",
    "racePctHisp": "% ประชากรฮิสแปนิก",
    "agePct12t21": "% อายุ 12-21 ปี",
    "agePct12t29": "% อายุ 12-29 ปี",
    "agePct16t24": "% อายุ 16-24 ปี",
    "agePct65up": "% อายุ 65 ปีขึ้นไป",
    "numbUrban": "จำนวนประชากรในเขตเมือง",
    "pctUrban": "% ประชากรในเขตเมือง",
    "medIncome": "รายได้ครัวเรือนมัธยฐาน",
    "pctWWage": "% ครัวเรือนที่มีรายได้จากค่าจ้าง",
    "pctWFarmSelf": "% ครัวเรือนที่มีรายได้จากฟาร์ม",
    "pctWInvInc": "% ครัวเรือนที่มีรายได้จากการลงทุน",
    "pctWSocSec": "% ครัวเรือนที่รับประกันสังคม",
    "pctWPubAsst": "% ครัวเรือนที่รับสวัสดิการรัฐ",
    "pctWRetire": "% ครัวเรือนที่มีรายได้บำนาญ",
    "medFamInc": "รายได้ครอบครัวมัธยฐาน",
    "perCapInc": "รายได้ต่อหัว",
    "whitePerCap": "รายได้ต่อหัวของคนผิวขาว",
    "blackPerCap": "รายได้ต่อหัวของคนผิวดำ",
    "indianPerCap": "รายได้ต่อหัวของคนอินเดียนแดง",
    "AsianPerCap": "รายได้ต่อหัวของคนเอเชีย",
    "OtherPerCap": "รายได้ต่อหัวของคนกลุ่มอื่น",
    "HispPerCap": "รายได้ต่อหัวของคนฮิสแปนิก",
    "NumUnderPov": "จำนวนคนที่อยู่ใต้เส้นความยากจน",
    "PctPopUnderPov": "% ประชากรที่อยู่ใต้เส้นความยากจน",
    "PctLess9thGrade": "% ที่การศึกษาต่ำกว่า ม.3",
    "PctNotHSGrad": "% ที่ไม่จบมัธยมปลาย",
    "PctBSorMore": "% ที่จบปริญญาตรีขึ้นไป",
    "PctUnemployed": "% ว่างงาน",
    "PctEmploy": "% มีงานทำ",
    "PctEmplManu": "% ทำงานภาคการผลิต",
    "PctEmplProfServ": "% ทำงานภาคบริการวิชาชีพ",
    "PctOccupMgmtProf": "% ทำงานระดับบริหาร/วิชาชีพ",
    "MalePctDivorce": "% ผู้ชายที่หย่าร้าง",
    "MalePctNevMarr": "% ผู้ชายที่ไม่เคยแต่งงาน",
    "FemalePctDiv": "% ผู้หญิงที่หย่าร้าง",
    "TotalPctDiv": "% ประชากรที่หย่าร้างทั้งหมด",
    "PersPerFam": "จำนวนคนเฉลี่ยต่อครอบครัว",
    "PctFam2Par": "% ครอบครัวที่มีพ่อแม่ครบ",
    "PctKids2Par": "% เด็กที่อยู่กับพ่อแม่ครบ",
    "PctYoungKids2Par": "% เด็กเล็กที่อยู่กับพ่อแม่ครบ",
    "PctTeen2Par": "% วัยรุ่นที่อยู่กับพ่อแม่ครบ",
    "PctWorkMomYoungKids": "% แม่ที่ทำงานและมีเด็กเล็ก",
    "PctWorkMom": "% แม่ที่ทำงาน",
    "NumIlleg": "จำนวนเด็กที่เกิดนอกสมรส",
    "PctIlleg": "% เด็กที่เกิดนอกสมรส",
    "NumImmig": "จำนวนผู้อพยพ",
    "PctImmigRecent": "% ผู้อพยพที่เพิ่งเข้ามา",
    "PctImmigRec5": "% ผู้อพยพใน 5 ปีล่าสุด",
    "PctImmigRec8": "% ผู้อพยพใน 8 ปีล่าสุด",
    "PctImmigRec10": "% ผู้อพยพใน 10 ปีล่าสุด",
    "PctRecentImmig": "% ผู้อพยพใหม่ต่อประชากร",
    "PctRecImmig5": "% ผู้อพยพ 5 ปีต่อประชากร",
    "PctRecImmig8": "% ผู้อพยพ 8 ปีต่อประชากร",
    "PctRecImmig10": "% ผู้อพยพ 10 ปีต่อประชากร",
    "PctSpeakEnglOnly": "% ที่พูดแต่ภาษาอังกฤษ",
    "PctNotSpeakEnglWell": "% ที่พูดอังกฤษไม่คล่อง",
    "PctLargHouseFam": "% ครอบครัวใหญ่ (6+ คน)",
    "PctLargHouseOccup": "% บ้านที่มีผู้อยู่อาศัยมาก",
    "PersPerOccupHous": "จำนวนคนเฉลี่ยต่อบ้านที่มีคนอยู่",
    "PersPerOwnOccHous": "จำนวนคนเฉลี่ยต่อบ้านที่เจ้าของอยู่เอง",
    "PersPerRentOccHous": "จำนวนคนเฉลี่ยต่อบ้านเช่า",
    "PctPersOwnOccup": "% คนที่อยู่บ้านตัวเอง",
    "PctPersDenseHous": "% คนที่อยู่บ้านแออัด",
    "PctHousLess3BR": "% บ้านที่มีน้อยกว่า 3 ห้องนอน",
    "MedNumBR": "จำนวนห้องนอนมัธยฐาน",
    "HousVacant": "จำนวนบ้านว่าง",
    "PctHousOccup": "% บ้านที่มีคนอยู่",
    "PctHousOwnOcc": "% บ้านที่เจ้าของอยู่เอง",
    "PctVacantBoarded": "% บ้านว่างที่ถูกปิดตาย",
    "PctVacMore6Mos": "% บ้านว่างมากกว่า 6 เดือน",
    "MedYrHousBuilt": "ปีที่สร้างบ้านมัธยฐาน",
    "PctHousNoPhone": "% บ้านที่ไม่มีโทรศัพท์",
    "PctWOFullPlumb": "% บ้านที่ไม่มีระบบประปาครบ",
    "OwnOccLowQuart": "มูลค่าบ้านควอไทล์ต่ำ",
    "OwnOccMedVal": "มูลค่าบ้านมัธยฐาน",
    "OwnOccHiQuart": "มูลค่าบ้านควอไทล์สูง",
    "RentLowQ": "ค่าเช่าควอไทล์ต่ำ",
    "RentMedian": "ค่าเช่ามัธยฐาน",
    "RentHighQ": "ค่าเช่าควอไทล์สูง",
    "MedRent": "ค่าเช่าเฉลี่ย",
    "MedRentPctHousInc": "% ค่าเช่าต่อรายได้ครัวเรือน",
    "MedOwnCostPctInc": "% ค่าใช้จ่ายบ้านต่อรายได้ (มีจำนอง)",
    "MedOwnCostPctIncNoMtg": "% ค่าใช้จ่ายบ้านต่อรายได้ (ไม่มีจำนอง)",
    "NumInShelters": "จำนวนคนในที่พักพิง",
    "NumStreet": "จำนวนคนไร้บ้าน",
    "PctForeignBorn": "% ที่เกิดต่างประเทศ",
    "PctBornSameState": "% ที่เกิดในรัฐเดียวกัน",
    "PctSameHouse85": "% ที่อยู่บ้านเดิมตั้งแต่ปี 1985",
    "PctSameCity85": "% ที่อยู่เมืองเดิมตั้งแต่ปี 1985",
    "PctSameState85": "% ที่อยู่รัฐเดิมตั้งแต่ปี 1985",
    "LemasSwornFT": "จำนวนตำรวจประจำการเต็มเวลา",
    "LemasSwFTPerPop": "ตำรวจต่อประชากร 100,000 คน",
    "LemasSwFTFieldOps": "จำนวนตำรวจปฏิบัติการภาคสนาม",
    "LemasSwFTFieldPerPop": "ตำรวจภาคสนามต่อประชากร",
    "LemasTotalReq": "จำนวนการร้องขอความช่วยเหลือทั้งหมด",
}

def predict(values):
    arr = np.array(values).reshape(1, -1)
    scaled = scaler.transform(arr)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0]
    crime_prob = prob[list(model.classes_).index(1)] * 100
    return pred, crime_prob

def show_result(pred, crime_prob):
    st.divider()
    if pred == 1:
        st.error(f"⚠️ **HIGH CRIME RATE** — Probability: {crime_prob:.2f}%")
    else:
        st.success(f"✅ **LOW CRIME RATE** — Probability: {crime_prob:.2f}%")
    st.progress(int(crime_prob))
    st.caption(f"Model: Logistic Regression + SMOTE | Crime probability: {crime_prob:.2f}%")

tab1, tab2 = st.tabs(["🎲 Random Sample", "✏️ Manual Input"])

with tab1:
    st.markdown("กดปุ่มเพื่อสุ่มข้อมูล community แล้วดูผล prediction")
    if st.button("🎲 สุ่มข้อมูลใหม่", use_container_width=True):
        seed = np.random.randint(0, 9999)
        np.random.seed(seed)
        random_vals = np.random.uniform(0, 1, size=model.n_features_in_)
        st.session_state['random_vals'] = random_vals.tolist()
        st.session_state['random_seed'] = seed

    if 'random_vals' in st.session_state:
        vals = st.session_state['random_vals']
        st.subheader("📋 ข้อมูลที่สุ่มได้")
        display_df = pd.DataFrame({
            "Feature": FEATURE_NAMES,
            "คำอธิบาย": [FEATURE_DESCRIPTIONS.get(f, "-") for f in FEATURE_NAMES],
            "ค่า": [f"{v:.4f}" for v in vals]
        })
        st.dataframe(display_df, use_container_width=True, height=400)
        if st.button("🔍 Predict จากข้อมูลนี้", use_container_width=True):
            pred, crime_prob = predict(vals)
            show_result(pred, crime_prob)
    else:
        st.info("👆 กดปุ่มสุ่มข้อมูลก่อนเพื่อเริ่มต้น")

with tab2:
    st.markdown("กรอกค่า feature ทีละตัว (ค่าอยู่ในช่วง 0.0 – 1.0 เนื่องจาก normalize แล้ว)")
    manual_vals = []
    cols = st.columns(3)
    for i, fname in enumerate(FEATURE_NAMES):
        col = cols[i % 3]
        desc = FEATURE_DESCRIPTIONS.get(fname, "")
        with col:
            val = st.number_input(
                label=f"**{fname}**",
                help=desc,
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
                format="%.4f",
                key=f"manual_{i}"
            )
            manual_vals.append(val)
    st.divider()
    if st.button("🔍 Predict", use_container_width=True, key="predict_manual"):
        pred, crime_prob = predict(manual_vals)
        show_result(pred, crime_prob)

st.divider()
st.markdown("**Assignment 5 | Imbalanced Data Classification | KMUTT**")
