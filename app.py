import os
import re
import glob
import joblib
import random
import string
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from streamlit_ketcher import st_ketcher
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors, Lipinski, Draw
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
import shap
from Bio import Entrez


# =========================
# 页面配置
# =========================
st.set_page_config(
    page_title="药物分子智能分析与活性预测平台 | Intelligent Drug Molecule Analysis and Activity Prediction Platform",
    page_icon="🔬",
    layout="wide"
)


# =========================
# 基础工具函数
# =========================
def get_csv_files():
    csv_files = glob.glob("./data/*.csv")
    if not csv_files:
        st.warning("没有在 ./data/ 文件夹中找到 CSV 文件，请先放入 BBBP.csv 或 clintox.csv。")
    return csv_files


def find_smiles_column(data):
    possible_cols = ["smiles", "SMILES", "Smiles", "canonical_smiles", "mol"]
    for col in possible_cols:
        if col in data.columns:
            return col
    return None


def display_data_summary(data):
    st.subheader("数据集概况")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("样本数", data.shape[0])
    with col2:
        st.metric("列数", data.shape[1])
    with col3:
        st.metric("缺失值总数", int(data.isna().sum().sum()))

    st.subheader("数据预览")
    st.dataframe(data.head(20), use_container_width=True)

    st.subheader("列名")
    st.write(list(data.columns))

    st.subheader("缺失值统计")
    missing_df = data.isna().sum().reset_index()
    missing_df.columns = ["列名", "缺失值数量"]
    st.dataframe(missing_df, use_container_width=True)

    numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()
    if numeric_columns:
        st.subheader("描述性统计")
        st.dataframe(data[numeric_columns].describe(), use_container_width=True)

        st.subheader("数值型特征分布")
        for col in numeric_columns[:5]:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(data[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"{col} Distribution")
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("当前数据集中没有数值型列。")


def create_project_directory(dataset_name, label_column):
    os.makedirs("./projects", exist_ok=True)
    safe_dataset = os.path.splitext(dataset_name)[0]
    safe_label = str(label_column).replace("/", "_").replace(" ", "_")
    random_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    project_name = datetime.now().strftime("%Y-%m-%d-%H-%M") + f"_{safe_dataset}_{safe_label}_{random_id}"
    project_dir = os.path.join("./projects", project_name)
    os.makedirs(project_dir, exist_ok=True)
    return project_dir


def mol_to_fp(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = fpgen.GetFingerprint(mol)
    arr = np.zeros((2048,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def save_input_data_with_fingerprint(data, project_dir, label_column):
    smiles_col = find_smiles_column(data)
    if smiles_col is None:
        st.error('无法找到 SMILES 列，请确认数据中有 "smiles" 或 "SMILES" 列。')
        return None

    records = []
    invalid_count = 0

    for _, row in data.iterrows():
        smiles = row[smiles_col]
        label = row[label_column]

        if pd.isna(smiles) or pd.isna(label):
            continue

        fp = mol_to_fp(smiles)
        if fp is None:
            invalid_count += 1
            continue

        records.append(list(fp) + [label])

    if not records:
        st.error("没有成功生成任何分子指纹，请检查 SMILES 列。")
        return None

    fp_columns = [f"fp_{i}" for i in range(2048)] + ["label"]
    fingerprint_df = pd.DataFrame(records, columns=fp_columns)

    output_file = os.path.join(project_dir, "input.csv")
    fingerprint_df.to_csv(output_file, index=False)

    meta = pd.DataFrame({
        "字段": ["smiles_col", "label_column", "valid_samples", "invalid_smiles"],
        "值": [smiles_col, label_column, fingerprint_df.shape[0], invalid_count]
    })
    meta.to_csv(os.path.join(project_dir, "meta.csv"), index=False)

    st.success(f"指纹数据已保存：{output_file}")
    st.info(f"有效样本数：{fingerprint_df.shape[0]}；无效 SMILES 数：{invalid_count}")
    return output_file


def preprocess_data(fp_file):
    data = pd.read_csv(fp_file)
    data = data.dropna()

    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna()
    return data


def plot_roc_curve(fpr, tpr, roc_auc, project_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    save_path = os.path.join(project_dir, "roc_curve.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    st.pyplot(fig)
    plt.close(fig)


def plot_confusion_matrix_fig(confusion, project_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    save_path = os.path.join(project_dir, "confusion_matrix.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    st.pyplot(fig)
    plt.close(fig)


def plot_feature_importance(importance, project_dir, top_n=20):
    importance_df = pd.DataFrame({
        "feature": [f"fp_{i}" for i in range(len(importance))],
        "importance": importance
    })
    importance_df = importance_df.sort_values("importance", ascending=False).head(top_n)
    importance_df.to_csv(os.path.join(project_dir, "feature_importance_top20.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=importance_df, x="importance", y="feature", ax=ax)
    ax.set_title("Top 20 Fingerprint Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Fingerprint bit")
    save_path = os.path.join(project_dir, "feature_importance.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    st.pyplot(fig)
    plt.close(fig)


def build_qsar_model(model_type, model_params):
    """
    根据页面选择构建不同的 QSAR 二分类模型。
    单模型用于对比，Stacking/Voting 用于多模型融合。
    """
    if model_type == "随机森林 Random Forest":
        return RandomForestClassifier(
            n_estimators=model_params["n_estimators"],
            max_depth=model_params["max_depth"],
            max_features=model_params["max_features"],
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )

    if model_type == "逻辑回归 Logistic Regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            ))
        ])

    if model_type == "支持向量机 SVM":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=model_params["svm_c"],
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=42
            ))
        ])

    if model_type == "梯度提升树 Gradient Boosting":
        return GradientBoostingClassifier(
            n_estimators=model_params["gb_n_estimators"],
            learning_rate=model_params["gb_learning_rate"],
            max_depth=model_params["gb_max_depth"],
            random_state=42
        )

    if model_type == "MLP 神经网络":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=0.0001,
                learning_rate_init=0.001,
                max_iter=300,
                random_state=42,
                early_stopping=True
            ))
        ])

    if model_type == "Voting 融合模型":
        estimators = [
            ("lr", Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
            ])),
            ("svm", Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(C=model_params["svm_c"], kernel="rbf", probability=True, class_weight="balanced", random_state=42))
            ])),
            ("rf", RandomForestClassifier(
                n_estimators=model_params["n_estimators"],
                max_depth=model_params["max_depth"],
                max_features=model_params["max_features"],
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            )),
            ("gb", GradientBoostingClassifier(
                n_estimators=model_params["gb_n_estimators"],
                learning_rate=model_params["gb_learning_rate"],
                max_depth=model_params["gb_max_depth"],
                random_state=42
            ))
        ]
        return VotingClassifier(
            estimators=estimators,
            voting="soft",
            n_jobs=-1
        )

    if model_type == "Stacking 融合模型":
        estimators = [
            ("lr", Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
            ])),
            ("svm", Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(C=model_params["svm_c"], kernel="rbf", probability=True, class_weight="balanced", random_state=42))
            ])),
            ("rf", RandomForestClassifier(
                n_estimators=model_params["n_estimators"],
                max_depth=model_params["max_depth"],
                max_features=model_params["max_features"],
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            )),
            ("gb", GradientBoostingClassifier(
                n_estimators=model_params["gb_n_estimators"],
                learning_rate=model_params["gb_learning_rate"],
                max_depth=model_params["gb_max_depth"],
                random_state=42
            ))
        ]
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
            stack_method="predict_proba",
            cv=5,
            n_jobs=-1,
            passthrough=False
        )

    raise ValueError(f"未知模型类型：{model_type}")


def get_positive_probability(model, X):
    """
    统一获取阳性类别概率，兼容不同 sklearn 模型。
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    return model.predict(X)


def plot_model_comparison(results_df, project_dir):
    """
    绘制不同模型的 Accuracy 和 AUC 对比图。
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = results_df.melt(id_vars="model", value_vars=["Accuracy", "AUC"], var_name="metric", value_name="value")
    sns.barplot(data=plot_df, x="model", y="value", hue="metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.tick_params(axis="x", rotation=25)
    save_path = os.path.join(project_dir, "model_comparison.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    st.pyplot(fig)
    plt.close(fig)


def plot_feature_importance_safe(model, project_dir):
    """
    只有树模型具有 feature_importances_，融合模型或 SVM/MLP 不强制绘制该图。
    """
    if hasattr(model, "feature_importances_"):
        plot_feature_importance(model.feature_importances_, project_dir)
    elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("clf"), "feature_importances_"):
        plot_feature_importance(model.named_steps["clf"].feature_importances_, project_dir)
    else:
        st.info("当前模型没有直接的 feature_importances_ 属性，已跳过特征重要性图。可在预测模块使用 SHAP 或概率结果进行解释。")


def train_and_save_model(fp_file, project_dir, model_type, model_params, compare_models=False):
    data = preprocess_data(fp_file)

    if data.shape[0] < 10:
        st.error("有效样本太少，无法训练模型。")
        return None, None, None

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].astype(int)

    unique_labels = sorted(y.unique())
    if len(unique_labels) != 2:
        st.error(f"当前标签不是二分类标签，检测到的标签为：{unique_labels}")
        return None, None, None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError as e:
        st.error(f"数据划分失败：{e}")
        return None, None, None

    if compare_models:
        st.subheader("基础模型与融合模型对比")
        candidate_model_types = [
            "逻辑回归 Logistic Regression",
            "支持向量机 SVM",
            "随机森林 Random Forest",
            "梯度提升树 Gradient Boosting",
            "Voting 融合模型",
            "Stacking 融合模型"
        ]
        compare_results = []

        for mt in candidate_model_types:
            try:
                temp_model = build_qsar_model(mt, model_params)
                temp_model.fit(X_train, y_train)
                temp_pred = temp_model.predict(X_test)
                temp_prob = get_positive_probability(temp_model, X_test)
                temp_acc = accuracy_score(y_test, temp_pred)
                temp_auc = auc(*roc_curve(y_test, temp_prob)[:2])
                
                # 增加名称映射，将带中文的模型名称转为纯英文，防止画图乱码
                en_model_map = {
                    "逻辑回归 Logistic Regression": "Logistic Regression",
                    "支持向量机 SVM": "SVM",
                    "随机森林 Random Forest": "Random Forest",
                    "梯度提升树 Gradient Boosting": "Gradient Boosting",
                    "Voting 融合模型": "Voting Ensemble",
                    "Stacking 融合模型": "Stacking Ensemble"
                }
                en_mt = en_model_map.get(mt, mt)
                
                # 这里改成 append en_mt
                compare_results.append({"model": en_mt, "Accuracy": temp_acc, "AUC": temp_auc})
            except Exception as e:
                st.warning(f"{mt} 训练或评价失败：{e}")
        
        if compare_results:
            results_df = pd.DataFrame(compare_results)
            results_df.to_csv(os.path.join(project_dir, "model_comparison.csv"), index=False)
            st.dataframe(results_df, use_container_width=True)
            plot_model_comparison(results_df, project_dir)

    model = build_qsar_model(model_type, model_params)

    try:
        model.fit(X_train, y_train)
    except Exception as e:
        st.error(f"模型训练失败：{e}")
        return None, None, None

    joblib.dump(model, os.path.join(project_dir, "model.pkl"))

    model_info = pd.DataFrame({
        "字段": ["model_type", "feature", "fingerprint", "test_size", "random_state"],
        "值": [model_type, "Morgan Fingerprint", "radius=2, fpSize=2048", "0.2", "42"]
    })
    model_info.to_csv(os.path.join(project_dir, "model_info.csv"), index=False)

    y_pred = model.predict(X_test)
    y_prob = get_positive_probability(model, X_test)

    acc = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    metrics_df = pd.DataFrame({
        "metric": ["Accuracy", "AUC"],
        "value": [acc, roc_auc]
    })
    metrics_df.to_csv(os.path.join(project_dir, "metrics.csv"), index=False)

    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).T.to_csv(os.path.join(project_dir, "classification_report.csv"))

    st.subheader("最终模型评价结果")
    st.info(f"当前最终模型：{model_type}")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{acc:.4f}")
    with col2:
        st.metric("AUC", f"{roc_auc:.4f}")

    plot_roc_curve(fpr, tpr, roc_auc, project_dir)
    plot_confusion_matrix_fig(confusion_matrix(y_test, y_pred), project_dir)
    plot_feature_importance_safe(model, project_dir)

    return model, acc, roc_auc

# =========================
# PubMed / PMC 与 LLM 函数
# =========================
def search_pmc(keyword, retmax=5):
    """
    在 PMC 中检索文献，返回 PMCID 列表。
    注意：Entrez.email 在页面中由用户输入，这里不再写死。
    """
    handle = Entrez.esearch(db="pmc", term=keyword, retmode="xml", retmax=retmax)
    record = Entrez.read(handle)
    return record.get("IdList", [])


def fetch_article_details(pmcid):
    """
    获取 PMC XML 格式全文。
    """
    handle = Entrez.efetch(db="pmc", id=pmcid, retmode="xml")
    record = Entrez.read(handle)
    return record


def clean_text(text):
    """
    清理空格、换行和多余符号。
    """
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def xml_to_text(element):
    """
    递归提取 Bio.Entrez 解析出的 XML/DictElement/List 中的纯文本。
    解决直接 str(DictElement) 导致页面显示一堆 DictElement(...) 的问题。
    """
    if element is None:
        return ""

    if isinstance(element, str):
        return element

    if isinstance(element, (int, float)):
        return str(element)

    if isinstance(element, list):
        return " ".join(xml_to_text(x) for x in element)

    if isinstance(element, dict):
        parts = []

        # Bio.Entrez 中部分文本可能存在 attributes 或 #text 一类字段
        for key, value in element.items():
            if key in ["attributes"]:
                continue
            parts.append(xml_to_text(value))

        return " ".join(parts)

    return str(element)


def extract_paragraphs_from_section(section):
    """
    从 PMC 正文章节中提取段落文本，尽量避免把表格、图注、引用对象等结构整体转成字符串。
    """
    paragraphs = []

    if isinstance(section, dict):
        if "title" in section:
            title = clean_text(xml_to_text(section["title"]))
            if title:
                paragraphs.append(title)

        if "p" in section:
            p_items = section["p"]
            if not isinstance(p_items, list):
                p_items = [p_items]
            for p in p_items:
                p_text = clean_text(xml_to_text(p))
                if p_text:
                    paragraphs.append(p_text)

        if "sec" in section:
            sec_items = section["sec"]
            if not isinstance(sec_items, list):
                sec_items = [sec_items]
            for sub_sec in sec_items:
                paragraphs.extend(extract_paragraphs_from_section(sub_sec))

    elif isinstance(section, list):
        for item in section:
            paragraphs.extend(extract_paragraphs_from_section(item))

    return paragraphs


def safe_get_article_text(article_details):
    """
    从 PMC XML 解析结果中提取题目、摘要和正文纯文本。
    """
    try:
        article = article_details[0]
        meta = article["front"]["article-meta"]

        title = ""
        if "title-group" in meta and "article-title" in meta["title-group"]:
            title = clean_text(xml_to_text(meta["title-group"]["article-title"]))

        abstract_text = ""
        if "abstract" in meta:
            abstract_text = clean_text(xml_to_text(meta["abstract"]))
        if not abstract_text:
            abstract_text = "该文献未提供摘要。"

        body_text = ""
        if "body" in article:
            body = article["body"]

            paragraphs = []
            if isinstance(body, dict) and "sec" in body:
                paragraphs = extract_paragraphs_from_section(body["sec"])
            else:
                paragraphs = extract_paragraphs_from_section(body)

            if paragraphs:
                body_text = "\n\n".join(paragraphs)
            else:
                body_text = clean_text(xml_to_text(body))

        if not body_text:
            body_text = "该文献未提供正文或正文无法解析。"

        return title, abstract_text, body_text

    except Exception as e:
        return "", f"摘要解析失败：{e}", f"正文解析失败：{e}"


# =========================
# 页面导航
# =========================
sidebar_option = st.sidebar.selectbox(
    "选择功能",
    [
        "首页",
        "数据展示",
        "模型训练",
        "活性预测",
        "分子性质分析",
        "分子相似性搜索",
        "知识获取"
    ]
)


# =========================
# 首页
# =========================
if sidebar_option == "首页":
    st.markdown(
        """
        <h1 style="text-align: center; color: #2E7D32;">药物分子智能分析与活性预测平台</h1>
        <h3 style="text-align: center; color: #666;">Intelligent Drug Molecule Analysis and Activity Prediction Platform</h3>
        <p style="text-align: center; font-size: 18px; color: #555;">
        基于 RDKit、QSAR 模型和分子信息学技术，实现药物活性预测、分子性质分析、相似性搜索与文献知识获取。
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <style>
            .card {
                background-color: #f9f9f9;
                border: 1px solid #d1d1d1;
                border-radius: 12px;
                box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.08);
                padding: 20px;
                margin-top: 20px;
                min-height: 120px;
            }
            .card-title {
                font-size: 20px;
                font-weight: bold;
                color: #2E7D32;
            }
            .card-description {
                color: #666;
                font-size: 14px;
                margin-top: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    cards = [
        ("数据展示", "读取 BBBP、ClinTox 等药物数据集，展示数据结构、缺失值和特征分布。"),
        ("模型训练", "将 SMILES 转换为 Morgan 指纹，可训练随机森林、SVM、梯度提升树、MLP 或 Stacking 融合 QSAR 二分类模型。"),
        ("活性预测", "调用已训练模型，对新输入的 SMILES 进行活性或毒性预测。"),
        ("分子性质分析", "计算分子量、LogP、TPSA、氢键受体/供体数和 Lipinski 五规则。"),
        ("分子相似性搜索", "基于 Morgan 指纹和 Tanimoto 系数，在数据集中搜索相似分子。"),
        ("知识获取", "从 PMC 文献中检索药物相关文献，并展示题目、摘要和全文节选。")
    ]

    for i in range(0, len(cards), 3):
        cols = st.columns(3)
        for col, (title, desc) in zip(cols, cards[i:i+3]):
            with col:
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="card-title">{title}</div>
                        <div class="card-description">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown(
        """
        <footer style="text-align: center; margin-top: 50px;">
            <p style="font-size: 14px; color: #888;">© 2026 Intelligent Drug Molecule Analysis and Activity Prediction Platform</p>
        </footer>
        """,
        unsafe_allow_html=True
    )


# =========================
# 数据展示
# =========================
elif sidebar_option == "数据展示":
    st.title("数据展示")
    csv_files = get_csv_files()
    if csv_files:
        dataset_choice = st.sidebar.selectbox("选择数据集", [os.path.basename(file) for file in csv_files])
        selected_file = csv_files[[os.path.basename(file) for file in csv_files].index(dataset_choice)]
        data = pd.read_csv(selected_file)
        st.info(f"当前数据集：{dataset_choice}")
        display_data_summary(data)


# =========================
# 模型训练
# =========================
elif sidebar_option == "模型训练":
    st.title("模型训练：QSAR 二分类模型")
    st.write("本模块使用 RDKit 将 SMILES 转换为 Morgan 指纹，支持单一模型训练和多模型融合训练。")

    csv_files = get_csv_files()
    if csv_files:
        dataset_choice = st.sidebar.selectbox("选择数据集", [os.path.basename(file) for file in csv_files])
        selected_file = csv_files[[os.path.basename(file) for file in csv_files].index(dataset_choice)]
        data = pd.read_csv(selected_file)

        st.subheader("数据预览")
        st.dataframe(data.head(), use_container_width=True)

        smiles_col = find_smiles_column(data)
        if smiles_col is None:
            st.error("该数据集没有检测到 SMILES 列，无法训练模型。")
        else:
            st.success(f"检测到 SMILES 列：{smiles_col}")

            candidate_labels = [col for col in data.columns if col != smiles_col and pd.api.types.is_numeric_dtype(data[col])]
            if not candidate_labels:
                candidate_labels = [col for col in data.columns if col != smiles_col]

            default_label = candidate_labels[0] if candidate_labels else data.columns[0]
            if dataset_choice.lower() == "bbbp.csv" and "p_np" in data.columns:
                default_label = "p_np"
            if dataset_choice.lower() == "clintox.csv" and "CT_TOX" in data.columns:
                default_label = "CT_TOX"

            label_column = st.sidebar.selectbox(
                "选择标签列",
                candidate_labels,
                index=candidate_labels.index(default_label) if default_label in candidate_labels else 0
            )

            st.info(f"当前选择标签列：{label_column}")
            st.write("标签分布：")
            st.dataframe(data[label_column].value_counts(dropna=False).reset_index().rename(columns={"index": "标签", label_column: "数量"}))

            model_type = st.sidebar.selectbox(
                "选择模型类型",
                [
                    "Stacking 融合模型",
                    "Voting 融合模型",
                    "随机森林 Random Forest",
                    "梯度提升树 Gradient Boosting",
                    "支持向量机 SVM",
                    "逻辑回归 Logistic Regression",
                    "MLP 神经网络"
                ]
            )

            compare_models = st.sidebar.checkbox("同时对比多个基础模型和融合模型", value=True)

            st.sidebar.subheader("模型参数")
            model_params = {
                "n_estimators": st.sidebar.slider("随机森林 n_estimators", 50, 500, 150),
                "max_depth": st.sidebar.slider("随机森林 max_depth", 1, 30, 10),
                "max_features": st.sidebar.slider("随机森林 max_features", 0.1, 1.0, 0.3),
                "svm_c": st.sidebar.slider("SVM C", 0.1, 10.0, 1.0),
                "gb_n_estimators": st.sidebar.slider("梯度提升树 n_estimators", 50, 300, 100),
                "gb_learning_rate": st.sidebar.slider("梯度提升树 learning_rate", 0.01, 0.30, 0.05),
                "gb_max_depth": st.sidebar.slider("梯度提升树 max_depth", 1, 5, 3)
            }

            if st.sidebar.button("开始训练模型"):
                project_dir = create_project_directory(dataset_choice, label_column)
                fp_file = save_input_data_with_fingerprint(data, project_dir, label_column)
                if fp_file is not None:
                    model, acc, roc_auc = train_and_save_model(
                        fp_file,
                        project_dir,
                        model_type=model_type,
                        model_params=model_params,
                        compare_models=compare_models
                    )
                    if model is not None:
                        st.success(f"训练完成，模型已保存到：{os.path.join(project_dir, 'model.pkl')}")


# =========================
# 活性预测
# =========================
elif sidebar_option == "活性预测":
    st.title("活性预测")
    st.write("本模块调用已训练好的 QSAR 模型，对用户输入的 SMILES 进行预测。")

    projects = sorted(glob.glob("./projects/*"), reverse=True)
    valid_projects = [p for p in projects if os.path.exists(os.path.join(p, "model.pkl"))]

    if not valid_projects:
        st.warning("没有找到已训练的项目。请先在“模型训练”模块训练模型。")
    else:
        project_names = [os.path.basename(project) for project in valid_projects]
        project_name = st.selectbox("选择训练好的模型项目", project_names)
        selected_project_dir = valid_projects[project_names.index(project_name)]

        model_filename = os.path.join(selected_project_dir, "model.pkl")
        model = joblib.load(model_filename)
        st.success(f"已加载模型：{model_filename}")

        default_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        molecule = st.text_input("输入分子 SMILES", default_smiles)

        try:
            smile_code = st_ketcher(molecule)
        except Exception:
            smile_code = molecule
            st.info("当前环境未正常加载 Ketcher，已使用文本框中的 SMILES。")

        if smile_code:
            st.markdown(f"当前 SMILES：`{smile_code}`")

            mol = Chem.MolFromSmiles(smile_code)

            if mol is None:
                st.error("无法解析该 SMILES，请输入有效结构。")
            else:
                st.image(Draw.MolToImage(mol, size=(300, 300)))

                fingerprint = mol_to_fp(smile_code)

                if fingerprint is not None:
                    fingerprint_2d = pd.DataFrame([fingerprint], columns=[f"fp_{i}" for i in range(2048)])

                    prediction = model.predict(fingerprint_2d)[0]
                    prob = model.predict_proba(fingerprint_2d)[0]

                    st.subheader("预测结果")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("预测类别", str(prediction))

                    with col2:
                        st.metric("阳性类别概率", f"{prob[-1]:.4f}")

                    prob_df = pd.DataFrame({
                        "类别": [str(c) for c in model.classes_],
                        "预测概率": prob
                    })

                    st.dataframe(prob_df, use_container_width=True)

                    st.subheader("SHAP解释")
                    st.caption(
                        "Morgan 指纹是 2048 位二进制结构特征。"
                        "下图展示对当前分子预测结果影响最大的前 20 个指纹位点。"
                    )

                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(fingerprint_2d)

                        # 兼容不同 SHAP 版本的输出格式
                        if isinstance(shap_values, list):
                            # RandomForest 二分类通常返回 [class0, class1]
                            sv = shap_values[-1][0]
                        else:
                            shap_values = np.array(shap_values)

                            if shap_values.ndim == 3:
                                # 可能是 shape = (样本数, 特征数, 类别数)
                                sv = shap_values[0, :, -1]
                            elif shap_values.ndim == 2:
                                # 可能是 shape = (样本数, 特征数)
                                sv = shap_values[0]
                            else:
                                sv = shap_values.reshape(-1)

                        sv = np.array(sv).reshape(-1)

                        top_n = min(20, len(sv))
                        top_idx = np.argsort(np.abs(sv))[-top_n:]
                        top_idx = top_idx[np.argsort(sv[top_idx])]

                        fig, ax = plt.subplots(figsize=(8, 6))

                        ax.barh(
                            range(top_n),
                            sv[top_idx]
                        )

                        ax.set_yticks(range(top_n))
                        ax.set_yticklabels([f"FP_{i}" for i in top_idx])
                        ax.set_xlabel("SHAP value")
                        ax.set_title("Top 20 SHAP fingerprint features")

                        st.pyplot(fig)
                        plt.close(fig)

                        shap_df = pd.DataFrame({
                            "指纹位点": [f"FP_{i}" for i in top_idx[::-1]],
                            "SHAP值": sv[top_idx[::-1]],
                            "影响方向": [
                                "提高阳性预测概率" if v > 0 else "降低阳性预测概率"
                                for v in sv[top_idx[::-1]]
                            ]
                        })

                        st.dataframe(shap_df, use_container_width=True)

                    except Exception as e:
                        st.warning(
                            f"SHAP解释生成失败，但预测结果正常。错误信息：{e}"
                        )

# =========================
# 分子性质分析
# =========================
elif sidebar_option == "分子性质分析":
    st.title("分子性质分析")
    st.write("输入 SMILES 后，平台会计算常见分子描述符和 Lipinski 五规则。")

    smiles = st.text_input("请输入 SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")

    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("SMILES 解析失败，请输入有效结构。")
        else:
            st.success("SMILES 解析成功")
            st.image(Draw.MolToImage(mol, size=(300, 300)))

            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hba = Lipinski.NumHAcceptors(mol)
            hbd = Lipinski.NumHDonors(mol)
            rot = Lipinski.NumRotatableBonds(mol)
            rings = Lipinski.RingCount(mol)

            result = pd.DataFrame({
                "性质": ["分子量 MolWt", "脂水分配系数 LogP", "极性表面积 TPSA", "氢键受体数 HBA", "氢键供体数 HBD", "可旋转键数", "环数"],
                "数值": [round(mw, 3), round(logp, 3), round(tpsa, 3), hba, hbd, rot, rings]
            })
            st.dataframe(result, use_container_width=True)

            violations = []
            if mw > 500:
                violations.append("分子量 > 500")
            if logp > 5:
                violations.append("LogP > 5")
            if hba > 10:
                violations.append("氢键受体数 > 10")
            if hbd > 5:
                violations.append("氢键供体数 > 5")

            st.subheader("Lipinski 五规则判断")
            if len(violations) == 0:
                st.success("符合 Lipinski 五规则，具有较好的类药性。")
            else:
                st.warning("不完全符合 Lipinski 五规则。")
                st.write("违反项：", violations)


# =========================
# 分子相似性搜索
# =========================
elif sidebar_option == "分子相似性搜索":
    st.title("分子相似性搜索")
    st.write("基于 Morgan 指纹和 Tanimoto 相似度，在数据集中寻找与查询分子最相似的化合物。")

    csv_files = get_csv_files()
    if csv_files:
        dataset_choice = st.selectbox("选择分子库", [os.path.basename(file) for file in csv_files])
        selected_file = csv_files[[os.path.basename(file) for file in csv_files].index(dataset_choice)]
        data = pd.read_csv(selected_file)

        default_smiles_col = find_smiles_column(data)
        smiles_col = st.selectbox(
            "选择 SMILES 列",
            data.columns.tolist(),
            index=data.columns.tolist().index(default_smiles_col) if default_smiles_col in data.columns else 0
        )

        query_smiles = st.text_input("请输入查询分子的 SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")
        top_n = st.slider("返回最相似分子数量", 3, 20, 5)

        if st.button("开始搜索"):
            query_mol = Chem.MolFromSmiles(query_smiles)
            if query_mol is None:
                st.error("查询 SMILES 无法解析。")
            else:
                st.image(Draw.MolToImage(query_mol, size=(250, 250)), caption="查询分子")

                fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
                query_fp = fpgen.GetFingerprint(query_mol)
                results = []

                for idx, row in data.iterrows():
                    mol = Chem.MolFromSmiles(str(row[smiles_col]))
                    if mol is None:
                        continue
                    fp = fpgen.GetFingerprint(mol)
                    sim = DataStructs.TanimotoSimilarity(query_fp, fp)

                    item = {
                        "序号": idx,
                        "SMILES": row[smiles_col],
                        "Tanimoto相似度": round(sim, 4)
                    }
                    if "name" in data.columns:
                        item["name"] = row["name"]
                    if "p_np" in data.columns:
                        item["p_np"] = row["p_np"]
                    if "CT_TOX" in data.columns:
                        item["CT_TOX"] = row["CT_TOX"]
                    if "FDA_APPROVED" in data.columns:
                        item["FDA_APPROVED"] = row["FDA_APPROVED"]
                    results.append(item)

                if not results:
                    st.warning("没有找到可解析的分子。")
                else:
                    result_df = pd.DataFrame(results)
                    result_df = result_df.sort_values("Tanimoto相似度", ascending=False).head(top_n)
                    st.subheader("相似性搜索结果")
                    st.dataframe(result_df, use_container_width=True)

                    st.subheader("Top 分子结构")
                    mols = [Chem.MolFromSmiles(s) for s in result_df["SMILES"].tolist()]
                    legends = [f"Sim={s}" for s in result_df["Tanimoto相似度"].tolist()]
                    img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(220, 180), legends=legends)
                    st.image(img)


# =========================
# 知识获取
# =========================
elif sidebar_option == "知识获取":
    st.title("知识获取")
    st.write("本模块从 PMC 文献中获取题目、摘要和全文节选，用于辅助药物设计相关文献阅读。")

    Entrez.email = st.text_input("请输入 Entrez Email", "your_email@example.com")
    keyword = st.text_input("请输入 PMC 检索关键词", '"Clinical Toxicology" AND Chemical')
    retmax = st.slider("返回文献数量", 1, 10, 5)

    if st.button("搜索 PMC 文献"):
        try:
            if not Entrez.email or Entrez.email == "your_email@example.com":
                st.warning("建议填写真实 Entrez Email，以便稳定访问 NCBI。")

            pmcid_list = search_pmc(keyword, retmax=retmax)
            st.session_state["pmcid_list"] = pmcid_list

            if len(pmcid_list) == 0:
                st.warning("没有检索到相关 PMC 文献。")
            else:
                st.success(f"搜索到 {len(pmcid_list)} 篇文献。")
                st.write(pmcid_list)

        except Exception as e:
            st.error(f"PMC 搜索失败：{e}")

    if "pmcid_list" in st.session_state and st.session_state["pmcid_list"]:
        pmcid = st.selectbox("选择 PMCID", st.session_state["pmcid_list"])

        if st.button("获取文献信息"):
            try:
                article_details = fetch_article_details(pmcid)
                title, abstract, full_text = safe_get_article_text(article_details)

                st.session_state["article_title"] = title
                st.session_state["article_abstract"] = abstract
                st.session_state["article_full_text"] = full_text

            except Exception as e:
                st.error(f"获取文献失败：{e}")

    if "article_abstract" in st.session_state:
        st.subheader("文献信息")

        st.info(f"题目：{st.session_state.get('article_title', '')}")

        st.subheader("摘要")
        st.text_area(
            "摘要内容",
            st.session_state.get("article_abstract", ""),
            height=220
        )

        st.subheader("全文节选")
        st.text_area(
            "全文内容",
            st.session_state.get("article_full_text", "")[:5000],
            height=300
        )
