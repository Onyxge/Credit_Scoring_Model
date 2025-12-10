# Credit Scoring model Project

## Credit Scoring Business Understanding

### 1. Basel II Accord & Model Interpretability
**Question:** How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

**Answer:** The Basel II Accord requires financial institutions to maintain rigorously tested and documented internal rating systems. This regulation emphasizes "transparency" and "auditability." Therefore, our model cannot be a "black box." We must be able to explain *why* a customer was classified as high-risk. This influences our choice to use techniques like Weight of Evidence (WoE), which provides clear, interpretable links between features (like transaction frequency) and risk, ensuring compliance with regulatory standards for capital allocation.

### 2. Proxy Variable Strategy
**Question:** Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks?

**Answer:** **Necessity:** We are working with transactional eCommerce data, not historical loan ledgers. There is no column stating "Defaulted." To build a supervised learning model, we must engineer a target variable (labels) using RFM (Recency, Frequency, Monetary) analysis to identify "disengaged" or "low-value" behavior that mimics credit risk.
**Risks:** The primary risk is **misclassification bias**. A customer might have low transaction frequency simply because they use a competitor or buy seasonally, not because they are a credit risk. If our proxy is inaccurate, we risk rejecting good customers (revenue loss) or approving bad ones (credit loss).

### 3. Model Selection Trade-offs
**Question:** What are the key trade-offs between using a simple, interpretable model (Logistic Regression with WoE) versus a complex, high-performance model (Gradient Boosting)?

**Answer:** * **Logistic Regression (with WoE):**
    * *Pros:* Highly interpretable, easy to explain to regulators (Basel II compliant), fast to train.
    * *Cons:* May miss complex, non-linear patterns in the data, potentially lower predictive accuracy.
* **Gradient Boosting (XGBoost/LightGBM):**
    * *Pros:* Typically achieves higher accuracy by capturing complex non-linear relationships.
    * *Cons:* "Black box" nature makes it harder to explain individual decisions to stakeholders and regulators.
    * *Strategy:* We will use MLflow to track both and determine if the accuracy gain of Boosting justifies the loss in interpretability.