import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from src.models.loan_recovery_model import LoanRecoveryModel
from src.utils.data_generator import generate_loan_data
from src.preprocessing.data_processor import LoanDataProcessor

# Set page configuration
st.set_page_config(
    page_title="Smart Loan Recovery System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define functions
@st.cache_data
def load_sample_data():
    """Load or generate sample data."""
    data_path = "data/loan_data.csv"
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    else:
        data = generate_loan_data(n_samples=1000)
        os.makedirs("data", exist_ok=True)
        data.to_csv(data_path, index=False)
        return data

@st.cache_resource
def load_model(model_type="random_forest"):
    """Load the trained model."""
    model_path = f"models/loan_recovery_{model_type}.pkl"

    # Check if model exists, if not train it
    if not os.path.exists(model_path):
        st.info(f"Model not found. Training a new {model_type} model...")
        from src.train_model import train_and_save_model
        train_and_save_model(model_type=model_type)

    return LoanRecoveryModel.load_model(model_path)

def predict_recovery(model, data):
    """Make predictions using the model."""
    recovery_probs = model.predict(data)
    return recovery_probs

def plot_recovery_distribution(data):
    """Plot the distribution of recovery status."""
    fig, ax = plt.subplots(figsize=(10, 6))
    recovery_counts = data['recovery_status'].value_counts()
    labels = ['Not Recovered', 'Recovered']
    ax.bar(labels, recovery_counts.values)
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Loan Recovery Status')
    for i, v in enumerate(recovery_counts.values):
        ax.text(i, v + 5, str(v), ha='center')

    # Add percentage labels
    total = len(data)
    for i, v in enumerate(recovery_counts.values):
        percentage = v / total * 100
        ax.text(i, v/2, f"{percentage:.1f}%", ha='center', color='white', fontweight='bold')

    return fig

def plot_feature_importance(model):
    """Plot feature importance."""
    return model.plot_feature_importance(top_n=10)

def plot_recovery_by_feature(data, feature, is_categorical=False):
    """Plot recovery rate by a specific feature."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if is_categorical:
        # For categorical features
        recovery_by_feature = data.groupby(feature)['recovery_status'].mean().sort_values()
        counts = data.groupby(feature).size()

        # Create a bar plot
        bars = ax.bar(recovery_by_feature.index, recovery_by_feature.values * 100)
        ax.set_ylabel('Recovery Rate (%)')
        ax.set_title(f'Recovery Rate by {feature.replace("_", " ").title()}')
        ax.set_ylim(0, 100)

        # Add count labels
        for i, (idx, count) in enumerate(counts.items()):
            ax.text(i, 5, f"n={count}", ha='center', color='white', fontweight='bold')

        # Rotate x-axis labels if needed
        if len(recovery_by_feature) > 5:
            plt.xticks(rotation=45, ha='right')
    else:
        # For numerical features, create bins
        if feature in ['age', 'loan_term', 'previous_defaults', 'days_past_due']:
            # These features have a small range, so we can use them directly
            data['feature_bin'] = data[feature]
        else:
            # Create bins for continuous features
            data['feature_bin'] = pd.qcut(data[feature], 5, duplicates='drop')

        # Calculate recovery rate by bin
        recovery_by_bin = data.groupby('feature_bin')['recovery_status'].mean().sort_index()
        counts = data.groupby('feature_bin').size()

        # Create a bar plot
        bars = ax.bar(range(len(recovery_by_bin)), recovery_by_bin.values * 100)
        ax.set_ylabel('Recovery Rate (%)')
        ax.set_title(f'Recovery Rate by {feature.replace("_", " ").title()}')
        ax.set_ylim(0, 100)

        # Set x-axis labels
        if feature in ['age', 'loan_term', 'previous_defaults', 'days_past_due']:
            ax.set_xticks(range(len(recovery_by_bin)))
            ax.set_xticklabels(recovery_by_bin.index)
        else:
            # Format bin labels
            bin_labels = []
            for bin_range in recovery_by_bin.index:
                if hasattr(bin_range, 'left') and hasattr(bin_range, 'right'):
                    bin_labels.append(f"{bin_range.left:.1f}-{bin_range.right:.1f}")
                else:
                    bin_labels.append(str(bin_range))

            ax.set_xticks(range(len(recovery_by_bin)))
            ax.set_xticklabels(bin_labels)
            plt.xticks(rotation=45, ha='right')

        # Add count labels
        for i, count in enumerate(counts.values):
            ax.text(i, 5, f"n={count}", ha='center', color='white', fontweight='bold')

        # Add feature name to x-axis
        ax.set_xlabel(feature.replace("_", " ").title())

    plt.tight_layout()
    return fig

# Main application
def main():
    # Header
    st.title("Smart Loan Recovery System")
    st.image("https://img.icons8.com/color/96/000000/loan.png", width=100)

    # Load data and model
    data = load_sample_data()

    # Load Random Forest model only
    model = load_model("random_forest")

    # Prediction page
    st.title("Predict Loan Recovery")

    st.write("""
    Use this tool to predict the probability of recovering a loan based on customer and loan information.
    You can either:
    1. Enter information for a single loan
    2. Upload a CSV file with multiple loans
    """)

    prediction_type = st.radio("Prediction Type", ["Single Loan", "Batch Prediction"])

    if prediction_type == "Single Loan":
        st.subheader("Enter Loan Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            employment_status = st.selectbox(
                "Employment Status",
                ["Employed", "Self-employed", "Unemployed", "Retired"]
            )
            annual_income = st.number_input("Annual Income ($)", min_value=0, value=60000)

        with col2:
            credit_score = st.slider("Credit Score", 300, 850, 650)
            loan_amount = st.number_input("Loan Amount ($)", min_value=1000, value=20000)
            interest_rate = st.slider("Interest Rate (%)", 1.0, 25.0, 8.0, 0.1)
            loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])

        with col3:
            payment_history = st.selectbox(
                "Payment History",
                ["Excellent", "Good", "Fair", "Poor", "Very Poor"]
            )
            days_past_due = st.number_input("Days Past Due", min_value=0, value=0)
            previous_defaults = st.number_input("Previous Defaults", min_value=0, max_value=10, value=0)

        # Calculate derived features
        monthly_payment = (loan_amount * (interest_rate/100/12) *
                          (1 + interest_rate/100/12)**(loan_term)) / \
                          ((1 + interest_rate/100/12)**(loan_term) - 1)

        debt_to_income = (monthly_payment * 12) / max(1, annual_income)

        # Display calculated values
        st.subheader("Calculated Values")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Monthly Payment", f"${monthly_payment:.2f}")
        with col2:
            st.metric("Debt-to-Income Ratio", f"{debt_to_income*100:.2f}%")

        # Create input dataframe
        input_data = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'employment_status': [employment_status],
            'annual_income': [annual_income],
            'credit_score': [credit_score],
            'loan_amount': [loan_amount],
            'interest_rate': [interest_rate],
            'loan_term': [loan_term],
            'payment_history': [payment_history],
            'days_past_due': [days_past_due],
            'previous_defaults': [previous_defaults],
            'monthly_payment': [monthly_payment],
            'debt_to_income': [debt_to_income]
        })

        # Make prediction
        if st.button("Predict Recovery Probability"):
            with st.spinner("Calculating recovery probability..."):
                recovery_prob = predict_recovery(model, input_data)[0]

                # Display result
                st.subheader("Prediction Result")

                # Create gauge chart for probability
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.barh([0], [100], color='lightgray', height=0.5)
                ax.barh([0], [recovery_prob * 100], color='green' if recovery_prob >= 0.5 else 'red', height=0.5)
                ax.set_xlim(0, 100)
                ax.set_yticks([])
                ax.set_xticks([0, 25, 50, 75, 100])
                ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
                ax.axvline(50, color='gray', linestyle='--', alpha=0.5)
                ax.text(recovery_prob * 100, 0, f"{recovery_prob*100:.1f}%",
                        ha='center', va='center', fontweight='bold', color='black')

                st.pyplot(fig)

                # Recommendation
                st.subheader("Recovery Assessment")
                if recovery_prob >= 0.8:
                    st.success("High probability of recovery. Standard collection procedures recommended.")
                elif recovery_prob >= 0.5:
                    st.info("Moderate probability of recovery. Consider offering a payment plan.")
                elif recovery_prob >= 0.3:
                    st.warning("Low probability of recovery. Consider debt restructuring or settlement offers.")
                else:
                    st.error("Very low probability of recovery. Consider debt write-off or third-party collection.")

                # Risk factors
                st.subheader("Key Risk Factors")
                risk_factors = []

                if credit_score < 600:
                    risk_factors.append("Low credit score")
                if days_past_due > 30:
                    risk_factors.append("Significant payment delay")
                if previous_defaults > 0:
                    risk_factors.append("History of defaults")
                if debt_to_income > 0.4:
                    risk_factors.append("High debt-to-income ratio")
                if payment_history in ["Poor", "Very Poor"]:
                    risk_factors.append("Poor payment history")

                if risk_factors:
                    for factor in risk_factors:
                        st.write(f"• {factor}")
                else:
                    st.write("No significant risk factors identified.")

    else:  # Batch prediction
        st.subheader("Upload CSV File")
        st.write("""
        Upload a CSV file with loan information. The file should contain the following columns:
        age, gender, employment_status, annual_income, credit_score, loan_amount, interest_rate,
        loan_term, payment_history, days_past_due, previous_defaults
        """)

        # Sample file download
        sample_data = data.sample(5).drop(['customer_id', 'recovery_status'], axis=1, errors='ignore')

        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(sample_data)
        st.download_button(
            "Download Sample CSV",
            csv,
            "sample_loans.csv",
            "text/csv",
            key='download-csv'
        )

        # File upload
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            # Load and display the data
            batch_data = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(batch_data.head())

            # Check for required columns
            required_cols = ['age', 'gender', 'employment_status', 'annual_income',
                            'credit_score', 'loan_amount', 'interest_rate',
                            'loan_term', 'payment_history', 'days_past_due',
                            'previous_defaults']

            missing_cols = [col for col in required_cols if col not in batch_data.columns]

            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                # Calculate derived features if not present
                if 'monthly_payment' not in batch_data.columns:
                    batch_data['monthly_payment'] = (
                        batch_data['loan_amount'] * (batch_data['interest_rate']/100/12) *
                        (1 + batch_data['interest_rate']/100/12)**(batch_data['loan_term'])
                    ) / (
                        (1 + batch_data['interest_rate']/100/12)**(batch_data['loan_term']) - 1
                    )

                if 'debt_to_income' not in batch_data.columns:
                    batch_data['debt_to_income'] = (batch_data['monthly_payment'] * 12) / batch_data['annual_income'].replace(0, 1)

                # Make predictions
                if st.button("Run Batch Prediction"):
                    with st.spinner("Processing batch predictions..."):
                        # Make predictions
                        recovery_probs = predict_recovery(model, batch_data)

                        # Add predictions to the dataframe
                        batch_data['recovery_probability'] = recovery_probs
                        batch_data['recovery_prediction'] = (recovery_probs >= 0.5).astype(int)

                        # Display results
                        st.subheader("Prediction Results")
                        st.dataframe(batch_data)

                        # Summary statistics
                        st.subheader("Summary")
                        avg_prob = batch_data['recovery_probability'].mean() * 100
                        predicted_recoveries = batch_data['recovery_prediction'].sum()
                        recovery_rate = predicted_recoveries / len(batch_data) * 100

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Average Recovery Probability", f"{avg_prob:.2f}%")
                        with col2:
                            st.metric("Predicted Recovery Rate", f"{recovery_rate:.2f}% ({predicted_recoveries}/{len(batch_data)})")

                        # Distribution of probabilities
                        st.subheader("Distribution of Recovery Probabilities")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.histplot(batch_data['recovery_probability'], bins=20, kde=True, ax=ax)
                        ax.set_xlabel("Recovery Probability")
                        ax.set_ylabel("Count")
                        ax.axvline(0.5, color='red', linestyle='--')
                        ax.text(0.5, ax.get_ylim()[1]*0.9, "Decision Threshold",
                                rotation=90, va='top', ha='right', color='red')
                        st.pyplot(fig)

                        # Download results
                        csv = convert_df_to_csv(batch_data)
                        st.download_button(
                            "Download Results CSV",
                            csv,
                            "loan_recovery_predictions.csv",
                            "text/csv",
                            key='download-results'
                        )



if __name__ == "__main__":
    main()
