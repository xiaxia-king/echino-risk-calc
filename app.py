import streamlit as st
import pandas as pd
import joblib
import os
import re
import plotly.graph_objects as go

# ==========================================
# 1. 网页全局配置 (Global Config)
# ==========================================
st.set_page_config(page_title="Echinococcosis Risk Screening", page_icon="🏥", layout="wide")

st.markdown("""
<style>
    .report-box { padding: 20px; border-radius: 10px; background-color: #f8f9fa; margin-top: 20px; margin-bottom: 20px; }
    .high-risk { border-left: 5px solid #C73E1D; background-color: #fdf3f2; }
    .mid-risk { border-left: 5px solid #FFA500; background-color: #fff9f0; }
    .low-risk { border-left: 5px solid #2ca02c; background-color: #f4fbf4; }
    .stSelectbox label { font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🏥 包虫病高危人群在线风险筛查平台")
st.markdown("**(Online Risk Screening Platform for Echinococcosis)**")
st.markdown("基于随机森林模型构建，旨在辅助基层医疗人员进行快速风险分层。<br>*(Built on a Random Forest model to assist healthcare workers in rapid risk stratification.)*", unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# 2. 加载模型 (Load Model)
# ==========================================
MODEL_PATH = "random_forest_best_model.joblib"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()
if model is None:
    st.error(f"⚠️ 找不到模型文件 (Model file not found): {MODEL_PATH}")
    st.stop()

# ==========================================
# 3. 核心字典映射 (Mapping Dictionaries)
# ==========================================
DRI_MEAN, DRI_SCALE = 2.1500712838598335, 3.4787767539870194
LRI_MEAN, LRI_SCALE = 1.4454490883169506, 2.3666996986490885

# 严格对齐基线表翻译 (Align with Baseline Table)
dict_region = {
    "1=西宁市 (Xining City)": 1, "2=海东市 (Haidong City)": 2, "3=海西州 (Haixi Prefecture)": 3, 
    "4=海南州 (Hainan Prefecture)": 4, "5=海北州 (Haibei Prefecture)": 5, "6=黄南州 (Huangnan Prefecture)": 6, 
    "7=玉树州 (Yushu Prefecture)": 7, "8=果洛州 (Golog Prefecture)": 8
}
dict_ethnicity = {"汉族 (Han)": 0, "藏族 (Tibetan)": 1, "回族 (Hui)": 2, "其他少数民族 (Other Ethnic minority)": 3}
dict_edu = {
    "从未上过学 (Never attended school)": 0, "小学 (Primary school)": 1, "初中 (Middle school)": 2, 
    "高中及以上 (High school or higher)": 3, "大学及以上 (University or higher)": 4
}
dict_occ = {"无业 (Unemployed)": 0, "农民 (Farmer)": 1, "牧民 (Herder)": 2, "其他 (Other)": 3}
dict_income = {"≤1000": 0, "1001-3000": 1, "3001-5000": 2, "≥5001": 3}
dict_yes_no_0_yes = {"是 (Yes)": 0, "否 (No)": 1}
dict_yes_no_1_yes = {"无 (None/No)": 0, "有 (Yes)": 1}
dict_pck = {"合格 (Qualified)": 1, "不合格 (Unqualified)": 0}

# 频率字典对齐 (Frequency Alignment)
dict_hbe = {"从不 (Never)": 0, "偶尔 (Sometimes)": 1, "每次 (Every time)": 2}
dict_dcf = {"无接触 (No Contact)": 0, "偶尔抚摸 (Occasionally Pet)": 1, "经常搂抱/抚摸 (Frequent Hugging/Petting)": 2}
dict_freq_often = {"经常 (Often)": 0, "偶尔 (Sometimes)": 1, "从不 (Never)": 2}
dict_dwsm = {"加盖容器 (Capped container)": 0, "无盖容器 (Uncapped container)": 1, "其他 (Others)": 2}
dict_gm = {
    "自由放牧 (Free Grazing)": 0, "固定放牧 (Fixed Grazing)": 1, "围栏放牧 (Fenced Grazing)": 2, 
    "季节轮换 (Seasonal Rotation)": 3, "混合放牧 (Mixed Grazing)": 4, "小区放牧 (Paddock Grazing)": 5, 
    "不放牧 (No Grazing)": 6
}

# ==========================================
# 4. 交互界面 (User Interface)
# ==========================================
input_data = {}
st.header("📋 流行病学特征输入 (Epidemiological Characteristics)")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 基本特征 (Demographics)")
    input_data['Region'] = dict_region[st.selectbox("地区 (Region)", list(dict_region.keys()))]
    input_data['Ethnicity'] = dict_ethnicity[st.selectbox("民族 (Ethnicity)", list(dict_ethnicity.keys()))]
    input_data['Education_Level'] = dict_edu[st.selectbox("教育程度 (Education Level)", list(dict_edu.keys()))]
    input_data['Occupation'] = dict_occ[st.selectbox("职业 (Occupation)", list(dict_occ.keys()))]
    input_data['Religious_Belief'] = dict_yes_no_1_yes[st.selectbox("宗教信仰 (Religious Belief)", list(dict_yes_no_1_yes.keys()), index=0)]
    input_data['Monthly_income'] = dict_income[st.selectbox("家庭月收入 (Monthly Income)", list(dict_income.keys()))]
    input_data['Chronic_illness_history'] = dict_yes_no_1_yes[st.selectbox("慢病史 (Chronic illness history)", list(dict_yes_no_1_yes.keys()))]
    input_data['PCK'] = dict_pck[st.selectbox("防治知识考核 (PCK: Prevention and control knowledge)", list(dict_pck.keys()))]

with col2:
    st.subheader("🐕 动物接触与指数 (Animal Contacts)")
    dog_choice = st.selectbox("是否养狗 (Dog ownership)", ["是 (Yes)", "否 (No)"], index=1)
    input_data['Dog_ownership'] = 0 if "Yes" in dog_choice else 1
    
    raw_dri = 0.0
    if "Yes" in dog_choice:
        q1 = st.selectbox("1. 养狗方式 (Rearing method)", ["固定/拴养 (0分)", "散养 (3分)"])
        q2 = st.selectbox("2. 进入厨房 (Enters kitchen)", ["经常 (3分)", "偶尔 (1分)", "从不 (0分)"])
        q3 = st.selectbox("3. 养狗目的 (Purpose)", ["放牧 (3分)", "看家 (2分)", "宠物 (0分)"])
        q4 = st.selectbox("4. 犬粪处理 (Feces disposal)", ["不处理 (3分)", "肥料 (2分)", "焚烧/深埋 (0分)"])
        q5 = st.selectbox("5. 驱虫频率 (Deworming)", ["定期 (0分)", "偶尔 (2分)", "从不 (3分)"])
        raw_dri = float(sum([int(re.search(r'\((\d+)分\)', q).group(1)) for q in [q1, q2, q3, q4, q5]]))
    input_data['DRI'] = (raw_dri - DRI_MEAN) / DRI_SCALE

    live_choice = st.selectbox("是否养家畜 (Livestock ownership)", ["是 (Yes)", "否 (No)"], index=1)
    input_data['Livestock_ownership'] = 0 if "Yes" in live_choice else 1
    raw_lri = 0.0
    if "Yes" in live_choice:
        ql1 = st.selectbox("1. 家畜驱虫 (Livestock deworming)", ["定期 (0分)", "偶尔 (2分)", "从不 (3分)"])
        ql2 = st.selectbox("2. 共用水源 (Shared water source)", ["是 (3分)", "否 (0分)"])
        ql3 = st.selectbox("3. 兽医治疗 (Veterinary treatment)", ["经常 (0分)", "偶尔 (2分)", "从不 (3分)"])
        raw_lri = float(sum([int(re.search(r'\((\d+)分\)', q).group(1)) for q in [ql1, ql2, ql3]]))
    input_data['LRI'] = (raw_lri - LRI_MEAN) / LRI_SCALE

    input_data['NYWS'] = dict_yes_no_0_yes[st.selectbox("近一年野外游泳史 (NYWS: Swimming in the wild)", list(dict_yes_no_0_yes.keys()), index=1)]
    input_data['HSS'] = dict_yes_no_0_yes[st.selectbox("家周围是否有屠宰场或肉店 (HSS: Slaughterhouse around the house)", list(dict_yes_no_0_yes.keys()), index=1)]
    input_data['DCF'] = dict_dcf[st.selectbox("与狗接触频率 (DCF: Dog Contact Frequency)", list(dict_dcf.keys()))]
    input_data['CRVF'] = dict_freq_often[st.selectbox("给狗喂食生的家畜脏器 (CRVF: Canines feed raw livestock viscera)", list(dict_freq_often.keys()), index=2)]
    input_data['CDOF'] = dict_yes_no_0_yes[st.selectbox("家畜病变脏器喂狗 (CDOF: Feeding canines with diseased organs)", list(dict_yes_no_0_yes.keys()), index=1)]

with col3:
    st.subheader("🏕️ 环境与习惯 (Environment & Habits)")
    input_data['RWD'] = dict_freq_often[st.selectbox("饮用未煮沸的水 (RWD: Drinking raw water)", list(dict_freq_often.keys()), index=2)]
    input_data['HSWA'] = dict_freq_often[st.selectbox("家周围野生动物出现 (HSWA: With wild animals around the house)", list(dict_freq_often.keys()), index=2)]
    input_data['DWSM'] = dict_dwsm[st.selectbox("饮用水储存方式 (DWSM: Drinking water storage mode)", list(dict_dwsm.keys()))]
    input_data['GM'] = dict_gm[st.selectbox("放牧方式 (GM: Grazing mode)", list(dict_gm.keys()), index=6)]
    input_data['HBE'] = dict_hbe[st.selectbox("饭前洗手 (HBE: Handwashing before eating)", list(dict_hbe.keys()), index=2)]
    input_data['RMR'] = dict_yes_no_0_yes[st.selectbox("读过包虫病宣教文章 (RMR: Read relevant materials)", list(dict_yes_no_0_yes.keys()), index=1)]
    input_data['Alcohol'] = dict_yes_no_0_yes[st.selectbox("是否饮酒 (Alcohol consumption)", list(dict_yes_no_0_yes.keys()), index=1)]

# ==========================================
# 5. 预测与报告 (Prediction & Report)
# ==========================================
st.markdown("---")
_, center_col, _ = st.columns([1, 2, 1])
if center_col.button("🚀 开始评估 (Start Risk Assessment)", type="primary", use_container_width=True):
    features_order = ["Region", "Ethnicity", "Education_Level", "Occupation", "Religious_Belief", "Monthly_income", "Chronic_illness_history", "PCK", "Dog_ownership", "DRI", "Livestock_ownership", "LRI", "NYWS", "HSS", "DCF", "CRVF", "CDOF", "RWD", "HSWA", "DWSM", "GM", "HBE", "RMR", "Alcohol"]
    input_df = pd.DataFrame([input_data], columns=features_order)
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("📊 评估报告与干预指南 (Report & Guidelines)")

    # 仪表盘 UI (Gauge Chart)
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        number = {'suffix': "%", 'font': {'size': 40}},
        title = {'text': "Overall Probability of Infection", 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 20], 'color': "#e8f5e9"},      
                {'range': [20, 46.8], 'color': "#fff3e0"},   
                {'range': [46.8, 100], 'color': "#ffebee"}   
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 46.8}
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

    if probability >= 0.468:
        st.markdown(f"""
        <div class="report-box high-risk">
            <h3 style="color: #C73E1D; margin-top: 0;">⚠️ 高危预警 (High Risk Alert)</h3>
            <p>系统评估感染概率为 {probability*100:.1f}%。 <i>(Infection probability: {probability*100:.1f}%.)</i></p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("#### 🎯 靶向干预建议 (Targeted Interventions)：")
        st.markdown("1. **临床转诊优先级 (Clinical Referral)**：强烈建议将该对象列为一级筛查名单，尽早安排腹部 B 超检测。<br>*(It is strongly recommended to list this individual for first-level screening and arrange abdominal ultrasound as soon as possible.)*", unsafe_allow_html=True)
        
        if input_data['CDOF'] == 0 or input_data['CRVF'] in [0, 1]:
            st.markdown("2. **源头管控 (Source Control)**：发现生喂或病变脏器喂狗行为。必须纠正，强调病死畜脏器的深埋或无害化处理。<br>*(Feeding raw or diseased organs to dogs was identified. Immediate correction is mandatory, emphasizing deep burial or incineration of diseased viscera.)*", unsafe_allow_html=True)
        if raw_dri >= 4.0:
            st.markdown("3. **犬只管理 (Dog Management)**：犬只风险指数(DRI)较高。需强制指导其进行犬只定期驱虫及粪便处理。<br>*(High DRI score requires mandatory deworming and safe disposal of dog feces.)*", unsafe_allow_html=True)
        if input_data['RWD'] in [0, 1]:
            st.markdown("4. **水源安全 (Water Safety)**：存在饮用生水风险。建议普及饮水安全知识，倡导彻底煮沸后饮用。<br>*(Risk of raw water consumption. Advocate for boiling water and public hygiene knowledge.)*", unsafe_allow_html=True)
    
    elif probability >= 0.20:
        st.markdown(f"""<div class="report-box mid-risk"><h3>🔔 中等风险 (Intermediate Risk)</h3><p>建议增加社区随访频次。 <i>(Increased community follow-up frequency recommended.)</i></p></div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="report-box low-risk"><h3>✅ 风险可控 (Low Risk)</h3><p>建议维持现有良好卫生习惯。 <i>(Routine hygiene habits maintained.)</i></p></div>""", unsafe_allow_html=True)
