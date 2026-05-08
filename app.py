import streamlit as st
import pandas as pd
import joblib
import os
import re
import plotly.graph_objects as go

# ==========================================
# 1. 网页全局配置与 UI 样式
# ==========================================
st.set_page_config(page_title="包虫病风险预测系统", page_icon="🏥", layout="wide")

# 自定义 CSS 样式
st.markdown("""
<style>
    .report-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8f9fa;
        border-left: 5px solid #2E86AB;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .high-risk { border-left: 5px solid #C73E1D; background-color: #fdf3f2; }
    .mid-risk { border-left: 5px solid #FFA500; background-color: #fff9f0; }
    .low-risk { border-left: 5px solid #2ca02c; background-color: #f4fbf4; }
</style>
""", unsafe_allow_html=True)

st.title("🏥 包虫病高危人群在线风险筛查平台")
st.markdown("**(Echinococcosis Risk Screening Platform)**")
st.markdown("基于机器学习随机森林 (Random Forest) 模型构建，旨在辅助基层医疗人员进行快速风险分层。")
st.markdown("---")

# ==========================================
# 2. 加载机器学习模型
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
    st.error(f"⚠️ 找不到模型文件！请确保模型已存在于根目录: {MODEL_PATH}")
    st.stop()

# ==========================================
# 3. 核心参数与字典映射
# ==========================================
# 标准化参数
DRI_MEAN, DRI_SCALE = 2.1500712838598335, 3.4787767539870194
LRI_MEAN, LRI_SCALE = 1.4454490883169506, 2.3666996986490885

# 映射字典
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
# 4. 前端交互界面
# ==========================================
input_data = {}
st.header("📋 居民流行病学特征输入")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 基本特征")
    input_data['Region'] = dict_region[st.selectbox("地区 (Region)", list(dict_region.keys()))]
    input_data['Ethnicity'] = dict_ethnicity[st.selectbox("民族 (Ethnicity)", list(dict_ethnicity.keys()))]
    input_data['Education_Level'] = dict_edu[st.selectbox("教育程度", list(dict_edu.keys()))]
    input_data['Occupation'] = dict_occ[st.selectbox("职业", list(dict_occ.keys()))]
    input_data['Religious_Belief'] = dict_yes_no_1_yes[st.selectbox("宗教信仰", list(dict_yes_no_1_yes.keys()), index=1)]
    input_data['Monthly_income'] = dict_income[st.selectbox("家庭月收入", list(dict_income.keys()))]
    input_data['Chronic_illness_history'] = dict_yes_no_1_yes[st.selectbox("慢病史", list(dict_yes_no_1_yes.keys()))]
    input_data['PCK'] = dict_pck[st.selectbox("防治知识考核 (PCK)", list(dict_pck.keys()))]

with col2:
    st.subheader("🐕 动物接触与指数")
    dog_choice = st.selectbox("是否养狗 (Dog Ownership)", list(dict_yes_no_0_yes.keys()), index=1)
    input_data['Dog_ownership'] = dict_yes_no_0_yes[dog_choice]
    raw_dri = 0.0
    if dog_choice == "是":
        q1_dog = st.selectbox("1. 养狗方式", ["固定/拴养 (0分)", "散养/半散养 (3分)"])
        q2_dog = st.selectbox("2. 狗是否能进入厨房", ["经常 (3分)", "偶尔 (1分)", "从不 (0分)"])
        q3_dog = st.selectbox("3. 养狗目的", ["牧羊犬/放牧 (3分)", "看家 (2分)", "宠物 (0分)"])
        q4_dog = st.selectbox("4. 犬粪处理方式", ["不处理 (3分)", "作肥料 (2分)", "深埋或焚烧 (0分)"])
        q5_dog = st.selectbox("5. 给狗驱虫频率", ["定期 (0分)", "偶尔 (2分)", "从不 (3分)"])
        raw_dri = float(sum([int(re.search(r'\((\d+)分\)', q).group(1)) for q in [q1_dog, q2_dog, q3_dog, q4_dog, q5_dog]]))
    input_data['DRI'] = (raw_dri - DRI_MEAN) / DRI_SCALE

    live_choice = st.selectbox("是否养家畜 (Livestock)", list(dict_yes_no_0_yes.keys()), index=1)
    input_data['Livestock_ownership'] = dict_yes_no_0_yes[live_choice]
    raw_lri = 0.0
    if live_choice == "是":
        q1_l = st.selectbox("1. 家畜驱虫频率", ["定期 (0分)", "偶尔 (2分)", "从不 (3分)"])
        q2_l = st.selectbox("2. 狗与家畜是否共用水源", ["否 (0分)", "是 (3分)"])
        q3_l = st.selectbox("3. 病畜是否接受兽医治疗", ["经常 (0分)", "偶尔 (2分)", "从不 (3分)"])
        raw_lri = float(sum([int(re.search(r'\((\d+)分\)', q).group(1)) for q in [q1_l, q2_l, q3_l]]))
    input_data['LRI'] = (raw_lri - LRI_MEAN) / LRI_SCALE

    input_data['NYWS'] = dict_yes_no_0_yes[st.selectbox("野外水源游泳史", list(dict_yes_no_0_yes.keys()), index=1)]
    input_data['HSS'] = dict_yes_no_0_yes[st.selectbox("家周围有无屠宰场", list(dict_yes_no_0_yes.keys()), index=1)]
    input_data['DCF'] = dict_freq_0_never[st.selectbox("与狗接触频率", list(dict_freq_0_never.keys()))]
    input_data['CRVF'] = dict_freq_0_often[st.selectbox("狗喂食生脏器频率", list(dict_freq_0_often.keys()), index=2)]
    input_data['CDOF'] = dict_yes_no_0_yes[st.selectbox("病变脏器是否喂狗", list(dict_yes_no_0_yes.keys()), index=1)]

with col3:
    st.subheader("🏕️ 环境与习惯")
    input_data['RWD'] = dict_freq_0_often[st.selectbox("饮用生水频率", list(dict_freq_0_often.keys()), index=2)]
    input_data['HSWA'] = dict_freq_0_often[st.selectbox("野生动物出现频率", list(dict_freq_0_often.keys()), index=2)]
    input_data['DWSM'] = dict_dwsm[st.selectbox("饮用水储存方式", list(dict_dwsm.keys()))]
    input_data['GM'] = dict_gm[st.selectbox("放牧方式", list(dict_gm.keys()), index=6)]
    input_data['HBE'] = dict_freq_0_never[st.selectbox("饭前洗手频率", list(dict_freq_0_never.keys()), index=2)]
    input_data['RMR'] = dict_yes_no_0_yes[st.selectbox("是否读过宣教文章", list(dict_yes_no_0_yes.keys()), index=1)]
    input_data['Alcohol'] = dict_yes_no_0_yes[st.selectbox("是否饮酒", list(dict_yes_no_0_yes.keys()), index=1)]

# ==========================================
# 5. 预测与仪表盘展示
# ==========================================
st.markdown("---")
_, center_col, _ = st.columns([1, 2, 1])
if center_col.button("🚀 开始风险评估 (Start Assessment)", type="primary", use_container_width=True):
    features_order = ["Region", "Ethnicity", "Education_Level", "Occupation", "Religious_Belief", "Monthly_income", "Chronic_illness_history", "PCK", "Dog_ownership", "DRI", "Livestock_ownership", "LRI", "NYWS", "HSS", "DCF", "CRVF", "CDOF", "RWD", "HSWA", "DWSM", "GM", "HBE", "RMR", "Alcohol"]
    input_df = pd.DataFrame([input_data], columns=features_order)
    probability = model.predict_proba(input_df)[0][1]

    with st.expander("🛠️ 开发者核对模式"):
        st.dataframe(input_df)

    # 📊 绘制高级风险仪表盘
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        number = {'suffix': "%", 'font': {'size': 40}},
        title = {'text': "系统综合评估感染概率", 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 20], 'color': "#e8f5e9"},      # 低：绿
                {'range': [20, 46.8], 'color': "#fff3e0"},   # 中：黄
                {'range': [46.8, 100], 'color': "#ffebee"}   # 高：红
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 46.8}
        }
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # 结果与建议
    if probability >= 0.468:
        st.error(f"⚠️ 高危预警：该居民感染概率为 {probability*100:.1f}%，超过截断值。")
        st.markdown("#### 🎯 靶向干预建议：")
        st.markdown("1. **立即筛查**：建议尽快安排腹部 B 超检查。")
        if input_data['CDOF'] == 0: st.markdown("- **源头管控**：严禁病变脏器喂狗，需无害化处理。")
        if raw_dri >= 4.0: st.markdown("- **犬只管理**：DRI得分较高，需强制进行犬只驱虫。")
    elif probability >= 0.20:
        st.warning(f"🔔 中风险提示：该居民感染概率为 {probability*100:.1f}%。")
        st.markdown("#### 🛡️ 防护建议：建议加强卫生知识宣教，增加随访频次。")
    else:
        st.success(f"✅ 风险可控：该居民感染概率为 {probability*100:.1f}%，处于安全区间。")
        st.markdown("#### 🛡️ 常规建议：维持良好习惯，按常规频次参与筛查。")
