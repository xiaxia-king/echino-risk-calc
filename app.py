import streamlit as st
import pandas as pd
import joblib
import os

# ==========================================
# 1. 网页全局配置与 UI 样式
# ==========================================
st.set_page_config(page_title="包虫病风险预测系统", page_icon="🏥", layout="wide")

# 自定义 CSS 样式（增加高级感，但不喧宾夺主）
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
    .high-risk {
        border-left: 5px solid #C73E1D;
        background-color: #fdf3f2;
    }
    .low-risk {
        border-left: 5px solid #2ca02c;
        background-color: #f4fbf4;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏥 包虫病高危人群在线风险筛查平台")
st.markdown("**(Echinococcosis Risk Screening Platform)**")
st.markdown("基于大规模社区流行病学调查与随机森林 (Random Forest) 机器学习模型构建。")
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
    st.error(f"⚠️ 找不到模型文件！请确保模型已存在于: {MODEL_PATH}")
    st.stop()  # 停止渲染下面的页面

# ==========================================
# 3. 核心参数与字典映射 (The "Codebook")
# ==========================================
# 提取到的标准化参数
DRI_MEAN = 2.1500712838598335
DRI_SCALE = 3.4787767539870194
LRI_MEAN = 1.4454490883169506
LRI_SCALE = 2.3666996986490885

# 分类变量的字典映射 (UI展示值 -> 模型特征值)
dict_region = {"1=西宁市": 1, "2=海东市": 2, "3=海西州": 3, "4=海南州": 4, "5=海北州": 5, "6=黄南州": 6, "7=玉树州": 7,
               "8=果洛州": 8}
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
# 4. 前端交互界面与动态逻辑
# ==========================================
input_data = {}

st.header("📋 请输入居民流行病学特征 (Epidemiological Characteristics)")
col1, col2, col3 = st.columns(3)

# ----------------- 第一列：人口学特征 -----------------
with col1:
    st.subheader("👤 基本特征 (Demographics)")
    input_data['Region'] = dict_region[st.selectbox("地区 (Region)", list(dict_region.keys()))]
    input_data['Ethnicity'] = dict_ethnicity[st.selectbox("民族 (Ethnicity)", list(dict_ethnicity.keys()))]
    input_data['Education_Level'] = dict_edu[st.selectbox("教育程度 (Education Level)", list(dict_edu.keys()))]
    input_data['Occupation'] = dict_occ[st.selectbox("职业 (Occupation)", list(dict_occ.keys()))]
    input_data['Religious_Belief'] = dict_yes_no_1_yes[
        st.selectbox("宗教信仰 (Religious Belief)", list(dict_yes_no_1_yes.keys()), index=1)]
    input_data['Monthly_income'] = dict_income[st.selectbox("家庭月收入 (Monthly Income)", list(dict_income.keys()))]
    input_data['Chronic_illness_history'] = dict_yes_no_1_yes[
        st.selectbox("慢病史 (Chronic Illness)", list(dict_yes_no_1_yes.keys()))]
    input_data['PCK'] = dict_pck[st.selectbox("防治知识考核 (PCK)", list(dict_pck.keys()))]

# ----------------- 第二列：动物接触与指数计算 -----------------
with col2:
    st.subheader("🐕 动物接触与管理 (Animal Contacts)")

    # 动态逻辑 1：犬只风险指数 (DRI) 计算
    dog_choice = st.selectbox("是否养狗 (Dog Ownership)", list(dict_yes_no_0_yes.keys()), index=1)
    input_data['Dog_ownership'] = dict_yes_no_0_yes[dog_choice]

    raw_dri = 0.0
    if dog_choice == "是":
        st.info("👇 请填写详细养犬管理行为：")
        # 精准匹配最新的打分规则
        q1_dog = st.selectbox("1. 养狗方式", ["固定/拴养 (0分)", "散养/半散养 (3分)"])
        q2_dog = st.selectbox("2. 狗是否能进入厨房", ["经常 (3分)", "偶尔 (1分)", "从不 (0分)"])
        q3_dog = st.selectbox("3. 养狗目的", ["牧羊犬/放牧 (3分)", "看家 (2分)", "宠物 (0分)"])
        q4_dog = st.selectbox("4. 犬粪处理方式", ["不处理 (3分)", "作肥料 (2分)", "深埋或焚烧 (0分)"])
        q5_dog = st.selectbox("5. 给狗驱虫频率", ["定期 (0分)", "偶尔 (2分)", "从不 (3分)"])

        # 智能提取括号里的数字作为分数
        import re

        score1 = int(re.search(r'\((\d+)分\)', q1_dog).group(1))
        score2 = int(re.search(r'\((\d+)分\)', q2_dog).group(1))
        score3 = int(re.search(r'\((\d+)分\)', q3_dog).group(1))
        score4 = int(re.search(r'\((\d+)分\)', q4_dog).group(1))
        score5 = int(re.search(r'\((\d+)分\)', q5_dog).group(1))

        raw_dri = float(score1 + score2 + score3 + score4 + score5)
        st.write(f"*当前计算的 DRI 原始分为: {raw_dri}*")

    # 对 DRI 进行标准化！
    input_data['DRI'] = (raw_dri - DRI_MEAN) / DRI_SCALE

    # 动态逻辑 2：家畜风险指数 (LRI) 计算
    live_choice = st.selectbox("是否养家畜 (Livestock Ownership)", list(dict_yes_no_0_yes.keys()), index=1)
    input_data['Livestock_ownership'] = dict_yes_no_0_yes[live_choice]

    raw_lri = 0.0
    if live_choice == "是":
        st.info("👇 请填写详细家畜管理行为：")
        # 精准匹配最新的打分规则
        q1_live = st.selectbox("1. 家畜驱虫频率", ["定期 (0分)", "偶尔 (2分)", "从不 (3分)"])
        q2_live = st.selectbox("2. 狗与家畜是否共用水源", ["否 (0分)", "是 (3分)"])
        q3_live = st.selectbox("3. 病畜是否接受兽医治疗", ["经常 (0分)", "偶尔 (2分)", "从不 (3分)"])

        # 智能提取分数
        import re

        score6 = int(re.search(r'\((\d+)分\)', q1_live).group(1))
        score7 = int(re.search(r'\((\d+)分\)', q2_live).group(1))
        score8 = int(re.search(r'\((\d+)分\)', q3_live).group(1))

        raw_lri = float(score6 + score7 + score8)
        st.write(f"*当前计算的 LRI 原始分为: {raw_lri}*")

    # 对 LRI 进行标准化！
    input_data['LRI'] = (raw_lri - LRI_MEAN) / LRI_SCALE

    st.markdown("---")
    input_data['NYWS'] = dict_yes_no_0_yes[
        st.selectbox("近一年野外水源游泳史 (NYWS)", list(dict_yes_no_0_yes.keys()), index=1)]
    input_data['HSS'] = dict_yes_no_0_yes[
        st.selectbox("家周围有无屠宰场 (HSS)", list(dict_yes_no_0_yes.keys()), index=1)]
    input_data['DCF'] = dict_freq_0_never[st.selectbox("与狗接触频率 (DCF)", list(dict_freq_0_never.keys()), index=0)]
    input_data['CRVF'] = dict_freq_0_often[
        st.selectbox("狗喂食生脏器频率 (CRVF)", list(dict_freq_0_often.keys()), index=2)]
    input_data['CDOF'] = dict_yes_no_0_yes[
        st.selectbox("病变脏器是否喂狗 (CDOF)", list(dict_yes_no_0_yes.keys()), index=1)]

# ----------------- 第三列：环境与其他行为 -----------------
with col3:
    st.subheader("🏕️ 环境与其他行为 (Environment & Behaviors)")
    input_data['RWD'] = dict_freq_0_often[st.selectbox("饮用生水频率 (RWD)", list(dict_freq_0_often.keys()), index=2)]
    input_data['HSWA'] = dict_freq_0_often[
        st.selectbox("家附近野生动物出现频率 (HSWA)", list(dict_freq_0_often.keys()), index=2)]
    input_data['DWSM'] = dict_dwsm[st.selectbox("饮用水储存方式 (DWSM)", list(dict_dwsm.keys()), index=0)]
    input_data['GM'] = dict_gm[st.selectbox("放牧方式 (GM)", list(dict_gm.keys()), index=6)]
    input_data['HBE'] = dict_freq_0_never[st.selectbox("饭前洗手频率 (HBE)", list(dict_freq_0_never.keys()), index=2)]
    input_data['RMR'] = dict_yes_no_0_yes[
        st.selectbox("是否读过宣教文章 (RMR)", list(dict_yes_no_0_yes.keys()), index=1)]
    input_data['Alcohol'] = dict_yes_no_0_yes[
        st.selectbox("是否饮酒 (Alcohol)", list(dict_yes_no_0_yes.keys()), index=1)]

# ==========================================
# 5. 预测引擎与靶向干预生成
# ==========================================
st.markdown("---")
# 创建居中大按钮
_, center_col, _ = st.columns([1, 2, 1])
if center_col.button("🚀 开始风险评估 (Start Risk Assessment)", type="primary", use_container_width=True):

    # 严格匹配模型训练时的特征顺序
    features_order = [
        "Region", "Ethnicity", "Education_Level", "Occupation", "Religious_Belief",
        "Monthly_income", "Chronic_illness_history", "PCK",
        "Dog_ownership", "DRI", "Livestock_ownership",
        "LRI", "NYWS", "HSS",
        "DCF", "CRVF", "CDOF", "RWD",
        "HSWA", "DWSM", "GM", "HBE",
        "RMR", "Alcohol"
    ]

    # 转换为 DataFrame
    input_df = pd.DataFrame([input_data], columns=features_order)

    # ====== 新增：透视眼调试工具 ======
    #with st.expander("🛠️ 开发者核对模式：点击查看传给底层的真实数据"):
    #    st.write("请将下面这一行数字，与你 test_data.xlsx 里的那名患者数据进行逐列严格核对：")
    #    st.dataframe(input_df)
    # ==================================

    # 预测概率
    probability = model.predict_proba(input_df)[0][1]

    # 获取特征重要性最高的前三项（简单的黑盒解释代偿）
    # 在实际运用中这可以增强医生的信心

    st.subheader("📊 评估报告与干预指南 (Report & Guidelines)")

    # 判断是否大于最佳阈值 (0.468 是根据你的日志提取的最佳 F1 阈值)
    if probability >= 0.468:
        # 高危展示模块
        st.markdown(f"""
        <div class="report-box high-risk">
            <h3 style="color: #C73E1D; margin-top: 0;">⚠️ 高危预警 (High Risk Alert)</h3>
            <p style="font-size: 18px;">系统综合评估表明，该居民的包虫病感染概率为 <b>{probability * 100:.1f}%</b>，超过了临床截断值 (46.8%)。</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🎯 靶向干预建议 (Targeted Interventions)：")
        st.markdown("1. **临床转诊优先级**：强烈建议将该对象列为一级筛查名单，尽早安排腹部 B 超检测。")

        # 动态捕捉高危因素并给出建议
        interventions = []
        if input_data['CDOF'] == 0 or input_data['CRVF'] in [0, 1]:
            interventions.append(
                "❗ **源头管控风险**：发现生喂或病变脏器喂狗行为。必须立即纠正，强调病死牛羊脏器的深埋或无害化处理。")
        if raw_dri >= 4.0:
            interventions.append(
                "❗ **犬只管理风险**：犬只风险指数(DRI)较高。需强制指导其进行犬只定期驱虫及粪便的安全处置。")
        if input_data['RWD'] in [0, 1] or input_data['DWSM'] in [1, 2]:
            interventions.append(
                "❗ **水源暴露风险**：存在饮用生水或水源暴露风险。建议普及饮水安全知识，倡导彻底煮沸后饮用。")
        if input_data['GM'] in [0, 4]:
            interventions.append("❗ **环境暴露风险**：处于高暴露游牧模式。建议在游牧期间加强个人防护及手卫生(饭前洗手)。")

        if interventions:
            for item in interventions:
                st.markdown(item)
        else:
            st.markdown("📌 请结合当地实际情况，开展综合防病知识宣教。")

    else:
        # 低危展示模块
        st.markdown(f"""
        <div class="report-box low-risk">
            <h3 style="color: #2ca02c; margin-top: 0;">✅ 风险可控 (Low Risk)</h3>
            <p style="font-size: 18px;">系统综合评估表明，该居民的包虫病感染概率为 <b>{probability * 100:.1f}%</b>，处于相对安全区间。</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🛡️ 常规防护建议 (Routine Advice)：")
        st.markdown("- 该居民目前发病风险较低，建议维持现有的良好卫生与生产习惯。")
        st.markdown("- 可按常规频次参与社区普筛，无需占用紧急医疗资源。")