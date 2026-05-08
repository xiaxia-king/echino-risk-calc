import streamlit as st
import pandas as pd
import joblib
import os
import re
import plotly.graph_objects as go

# ==========================================
# 1. 网页全局配置与 UI 样式 (Global Config & UI)
# ==========================================
st.set_page_config(page_title="Echinococcosis Risk Screening", page_icon="🏥", layout="wide")

st.markdown("""
<style>
    .report-box { padding: 20px; border-radius: 10px; background-color: #f8f9fa; margin-top: 20px; margin-bottom: 20px; }
    .high-risk { border-left: 5px solid #C73E1D; background-color: #fdf3f2; }
    .mid-risk { border-left: 5px solid #FFA500; background-color: #fff9f0; }
    .low-risk { border-left: 5px solid #2ca02c; background-color: #f4fbf4; }
</style>
""", unsafe_allow_html=True)

st.title("🏥 包虫病高危人群在线风险筛查平台")
st.markdown("**(Online Risk Screening Platform for Echinococcosis in High-Risk Populations)**")
st.markdown("基于机器学习随机森林模型构建，旨在辅助基层医疗人员进行快速风险分层。 <br> *(Built on a Random Forest machine learning model to assist primary healthcare workers in rapid risk stratification.)*", unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# 2. 加载机器学习模型 (Load Model)
# ==========================================
MODEL_PATH = "random_forest_best_model.joblib"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return None

model = load_model()

if model is None:
    st.error(f"⚠️ 找不到模型文件 (Model file not found)！请确保模型已存在于根目录: {MODEL_PATH}")
    st.stop()

# ==========================================
# 3. 核心参数与字典映射 (Parameters & Dictionaries)
# ==========================================
DRI_MEAN, DRI_SCALE = 2.1500712838598335, 3.4787767539870194
LRI_MEAN, LRI_SCALE = 1.4454490883169506, 2.3666996986490885

dict_region = {"1=西宁市": 1, "2=海东市": 2, "3=海西州": 3, "4=海南州": 4, "5=海北州": 5, "6=黄南州": 6, "7=玉树州": 7, "8=果洛州": 8}
dict_ethnicity = {"汉族": 0, "藏族": 1, "回族": 2, "其他少数民族": 3}
dict_edu = {"从未上过学": 0, "小学": 1, "初中": 2, "高中及以上": 3, "大学及以上": 4}
dict_occ = {"无业": 0, "农民": 1, "牧民": 2, "其他": 3}
dict_income = {"＜1000": 0, "1001-3000": 1, "3001-5000": 2, "＞5000": 3}
dict_yes_no_0_yes = {"是": 0, "否": 1}
dict_yes_no_1_yes = {"无": 0, "有": 1}
dict_pck = {"不合格": 0, "合格": 1}
dict_freq_0_never = {"从不": 0, "偶尔": 1, "经常": 2}
dict_freq_0_often = {"经常": 0, "偶尔": 1, "从不": 2}
dict_dwsm = {"加盖容器": 0, "无盖容器": 1, "其他方式": 2}
dict_gm = {"自由放牧": 0, "固定放牧": 1, "围栏放牧": 2, "季节轮换": 3, "混合放牧": 4, "小区放牧": 5, "不放牧": 6}

# ==========================================
# 4. 前端交互界面 (User Interface)
# ==========================================
input_data = {}
st.header("📋 居民流行病学特征输入 (Input Epidemiological Characteristics)")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 基本特征 (Demographics)")
    input_data['Region'] = dict_region[st.selectbox("地区 (Region)", list(dict_region.keys()))]
    input_data['Ethnicity'] = dict_ethnicity[st.selectbox("民族 (Ethnicity)", list(dict_ethnicity.keys()))]
    input_data['Education_Level'] = dict_edu[st.selectbox("教育程度 (Education Level)", list(dict_edu.keys()))]
    input_data['Occupation'] = dict_occ[st.selectbox("职业 (Occupation)", list(dict_occ.keys()))]
    input_data['Religious_Belief'] = dict_yes_no_1_yes[st.selectbox("宗教信仰 (Religious Belief)", list(dict_yes_no_1_yes.keys()), index=1)]
    input_data['Monthly_income'] = dict_income[st.selectbox("家庭月收入 (Monthly Income)", list(dict_income.keys()))]
    input_data['Chronic_illness_history'] = dict_yes_no_1_yes[st.selectbox("慢病史 (Chronic Illness History)", list(dict_yes_no_1_yes.keys()))]
    input_data['PCK'] = dict_pck[st.selectbox("防治知识考核 (Prevention Knowledge Score)", list(dict_pck.keys()))]

with col2:
    st.subheader("🐕 动物接触与指数 (Animal Contacts)")
    dog_choice = st.selectbox("是否养狗 (Dog Ownership)", list(dict_yes_no_0_yes.keys()), index=1)
    input_data['Dog_ownership'] = dict_yes_no_0_yes[dog_choice]
    raw_dri = 0.0
    if dog_choice == "是":
        q1_dog = st.selectbox("1. 养狗方式 (Dog rearing method)", ["固定/拴养 (0分)", "散养/半散养 (3分)"])
        q2_dog = st.selectbox("2. 狗是否能进入厨房 (Dog enters kitchen)", ["经常 (3分)", "偶尔 (1分)", "从不 (0分)"])
        q3_dog = st.selectbox("3. 养狗目的 (Purpose of keeping)", ["牧羊犬/放牧 (3分)", "看家 (2分)", "宠物 (0分)"])
        q4_dog = st.selectbox("4. 犬粪处理方式 (Feces disposal)", ["不处理 (3分)", "作肥料 (2分)", "深埋或焚烧 (0分)"])
        q5_dog = st.selectbox("5. 给狗驱虫频率 (Deworming frequency)", ["定期 (0分)", "偶尔 (2分)", "从不 (3分)"])
        raw_dri = float(sum([int(re.search(r'\((\d+)分\)', q).group(1)) for q in [q1_dog, q2_dog, q3_dog, q4_dog, q5_dog]]))
    input_data['DRI'] = (raw_dri - DRI_MEAN) / DRI_SCALE

    live_choice = st.selectbox("是否养家畜 (Livestock Ownership)", list(dict_yes_no_0_yes.keys()), index=1)
    input_data['Livestock_ownership'] = dict_yes_no_0_yes[live_choice]
    raw_lri = 0.0
    if live_choice == "是":
        q1_l = st.selectbox("1. 家畜驱虫频率 (Livestock deworming)", ["定期 (0分)", "偶尔 (2分)", "从不 (3分)"])
        q2_l = st.selectbox("2. 狗与家畜是否共用水源 (Shared water source)", ["否 (0分)", "是 (3分)"])
        q3_l = st.selectbox("3. 病畜是否接受兽医治疗 (Veterinary treatment)", ["经常 (0分)", "偶尔 (2分)", "从不 (3分)"])
        raw_lri = float(sum([int(re.search(r'\((\d+)分\)', q).group(1)) for q in [q1_l, q2_l, q3_l]]))
    input_data['LRI'] = (raw_lri - LRI_MEAN) / LRI_SCALE

    input_data['NYWS'] = dict_yes_no_0_yes[st.selectbox("近一年野外水源游泳史 (Swimming in wild waters)", list(dict_yes_no_0_yes.keys()), index=1)]
    input_data['HSS'] = dict_yes_no_0_yes[st.selectbox("家周围有无屠宰场 (Slaughterhouse nearby)", list(dict_yes_no_0_yes.keys()), index=1)]
    input_data['DCF'] = dict_freq_0_never[st.selectbox("与狗接触频率 (Dog contact frequency)", list(dict_freq_0_never.keys()))]
    input_data['CRVF'] = dict_freq_0_often[st.selectbox("喂狗生脏器频率 (Raw viscera feeding)", list(dict_freq_0_often.keys()), index=2)]
    input_data['CDOF'] = dict_yes_no_0_yes[st.selectbox("病变脏器是否喂狗 (Diseased viscera to dogs)", list(dict_yes_no_0_yes.keys()), index=1)]

with col3:
    st.subheader("🏕️ 环境与习惯 (Environment & Habits)")
    input_data['RWD'] = dict_freq_0_often[st.selectbox("饮用生水频率 (Drinking unboiled water)", list(dict_freq_0_often.keys()), index=2)]
    input_data['HSWA'] = dict_freq_0_often[st.selectbox("野生动物出现频率 (Wild animal sightings)", list(dict_freq_0_often.keys()), index=2)]
    input_data['DWSM'] = dict_dwsm[st.selectbox("饮用水储存方式 (Water storage method)", list(dict_dwsm.keys()))]
    input_data['GM'] = dict_gm[st.selectbox("放牧方式 (Grazing method)", list(dict_gm.keys()), index=6)]
    input_data['HBE'] = dict_freq_0_never[st.selectbox("饭前洗手频率 (Washing hands before meals)", list(dict_freq_0_never.keys()), index=2)]
    input_data['RMR'] = dict_yes_no_0_yes[st.selectbox("是否读过宣教文章 (Read educational articles)", list(dict_yes_no_0_yes.keys()), index=1)]
    input_data['Alcohol'] = dict_yes_no_0_yes[st.selectbox("是否饮酒 (Alcohol consumption)", list(dict_yes_no_0_yes.keys()), index=1)]

# ==========================================
# 5. 预测引擎与评估报告 (Prediction & Report)
# ==========================================
st.markdown("---")
_, center_col, _ = st.columns([1, 2, 1])
if center_col.button("🚀 开始风险评估 (Start Risk Assessment)", type="primary", use_container_width=True):
    features_order = ["Region", "Ethnicity", "Education_Level", "Occupation", "Religious_Belief", "Monthly_income", "Chronic_illness_history", "PCK", "Dog_ownership", "DRI", "Livestock_ownership", "LRI", "NYWS", "HSS", "DCF", "CRVF", "CDOF", "RWD", "HSWA", "DWSM", "GM", "HBE", "RMR", "Alcohol"]
    input_df = pd.DataFrame([input_data], columns=features_order)
    probability = model.predict_proba(input_df)[0][1]

    with st.expander("🛠️ 开发者核对模式 (Developer Debug Mode)"):
        st.dataframe(input_df)

    st.subheader("📊 评估报告与干预指南 (Report & Guidelines)")

    # 绘制高级风险仪表盘
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        number = {'suffix': "%", 'font': {'size': 40, 'color': '#333333'}},
        title = {'text': "系统综合评估感染概率<br><span style='font-size:14px;color:gray'>Overall Probability of Infection</span>", 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "rgba(0,0,0,0)"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': "#e8f5e9"},      
                {'range': [20, 46.8], 'color': "#fff3e0"},   
                {'range': [46.8, 100], 'color': "#ffebee"}   
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # 动态输出高低危结论与完全恢复的靶向干预
    if probability >= 0.468:
        st.markdown(f"""
        <div class="report-box high-risk">
            <h3 style="color: #C73E1D; margin-top: 0;">⚠️ 高危预警 (High Risk Alert)</h3>
            <p style="font-size: 16px;">系统评估感染概率为 <b>{probability * 100:.1f}%</b>，超过临床截断值 (46.8%)。<br>
            <i>(The estimated probability of infection is {probability * 100:.1f}%, exceeding the clinical cutoff.)</i></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🎯 靶向干预建议 (Targeted Interventions)：")
        st.markdown("1. **临床转诊优先级 (Clinical Referral)**：强烈建议将该对象列为一级筛查名单，尽早安排腹部 B 超检测。")
        
        interventions = []
        if input_data['CDOF'] == 0 or input_data['CRVF'] in [0, 1]:
            interventions.append("❗ **源头管控 (Source Control)**：发现生喂或病变脏器喂狗行为。必须立即纠正，强调病死牛羊脏器的深埋或无害化处理。")
        if raw_dri >= 4.0:
            interventions.append("❗ **犬只管理 (Dog Management)**：犬只风险指数(DRI)较高。需强制指导其进行犬只定期驱虫及粪便的安全处置。")
        if input_data['RWD'] in [0, 1] or input_data['DWSM'] in [1, 2]:
            interventions.append("❗ **水源暴露 (Water Safety)**：存在饮用生水或水源暴露风险。建议普及饮水安全知识，倡导彻底煮沸后饮用。")
        if input_data['GM'] in [0, 4]:
            interventions.append("❗ **环境暴露 (Environment Exposure)**：处于高暴露放牧模式。建议在游牧期间加强个人防护及手卫生。")

        if interventions:
            for item in interventions:
                st.markdown(item)
        else:
            st.markdown("📌 请结合当地实际情况，开展综合防病知识宣教。")

    elif probability >= 0.20:
        st.markdown(f"""
        <div class="report-box mid-risk">
            <h3 style="color: #FFA500; margin-top: 0;">🔔 中等风险 (Intermediate Risk)</h3>
            <p style="font-size: 16px;">系统评估感染概率为 <b>{probability * 100:.1f}%</b>，需引起重视。<br>
            <i>(The estimated probability of infection is {probability * 100:.1f}%, requiring attention.)</i></p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("#### 🛡️ 常规干预建议 (Routine Interventions)：")
        st.markdown("- **加强监测 (Monitoring)**：建议增加该人群的社区随访频次，重点科普包虫病传播途径。")
    else:
        st.markdown(f"""
        <div class="report-box low-risk">
            <h3 style="color: #2ca02c; margin-top: 0;">✅ 风险可控 (Low Risk)</h3>
            <p style="font-size: 16px;">系统评估感染概率为 <b>{probability * 100:.1f}%</b>，处于相对安全区间。<br>
            <i>(The estimated probability of infection is {probability * 100:.1f}%, within the safe range.)</i></p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("#### 🛡️ 常规防护建议 (Routine Advice)：")
        st.markdown("- 该居民目前发病风险较低，建议维持现有的良好卫生与生产习惯。")
        st.markdown("- 可按常规频次参与社区普筛，无需占用紧急医疗资源。")
