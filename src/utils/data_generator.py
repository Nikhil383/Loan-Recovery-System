import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_loan_data(n_samples=1000, seed=42):
    """
    Generate synthetic loan data for the loan recovery system.

    Parameters:
    -----------
    n_samples : int
        Number of loan records to generate
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing synthetic loan data
    """
    np.random.seed(seed)
    random.seed(seed)

    # Customer information
    customer_ids = [f'CUST{i:06d}' for i in range(1, n_samples + 1)]
    ages = np.random.randint(22, 65, n_samples)
    genders = np.random.choice(['Male', 'Female'], n_samples)

    # Employment information
    employment_statuses = np.random.choice(
        ['Employed', 'Self-employed', 'Unemployed', 'Retired'],
        n_samples,
        p=[0.65, 0.20, 0.10, 0.05]
    )
    annual_incomes = []
    for status in employment_statuses:
        if status == 'Employed':
            annual_incomes.append(np.random.normal(60000, 20000))
        elif status == 'Self-employed':
            annual_incomes.append(np.random.normal(75000, 30000))
        elif status == 'Unemployed':
            annual_incomes.append(np.random.normal(15000, 10000))
        else:  # Retired
            annual_incomes.append(np.random.normal(40000, 15000))

    # Credit information
    credit_scores = []
    for income in annual_incomes:
        base_score = 300 + (income / 100000) * 400  # Higher income tends to have higher credit score
        credit_scores.append(min(850, max(300, int(np.random.normal(base_score, 50)))))

    # Loan information
    loan_amounts = []
    for income, credit in zip(annual_incomes, credit_scores):
        # Higher income and credit score can get larger loans
        max_loan = income * (0.5 + (credit - 300) / 850)
        loan_amounts.append(np.random.uniform(5000, max_loan))

    interest_rates = []
    for credit in credit_scores:
        # Lower credit scores get higher interest rates
        base_rate = 15 - (credit - 300) * (10 / 550)  # Range from ~5% to ~15%
        interest_rates.append(max(5, min(15, base_rate + np.random.normal(0, 1))))

    loan_terms = np.random.choice([12, 24, 36, 48, 60], n_samples)

    # Loan performance
    payment_histories = []
    for credit in credit_scores:
        # Better credit scores tend to have better payment histories
        if credit > 750:
            payment_histories.append(np.random.choice(['Excellent', 'Good', 'Fair'], p=[0.8, 0.15, 0.05]))
        elif credit > 650:
            payment_histories.append(np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], p=[0.4, 0.4, 0.15, 0.05]))
        elif credit > 550:
            payment_histories.append(np.random.choice(['Good', 'Fair', 'Poor'], p=[0.3, 0.5, 0.2]))
        else:
            payment_histories.append(np.random.choice(['Fair', 'Poor', 'Very Poor'], p=[0.3, 0.5, 0.2]))

    days_past_due = []
    for history in payment_histories:
        if history == 'Excellent':
            days_past_due.append(np.random.choice([0, 0, 0, 0, np.random.randint(1, 10)], p=[0.9, 0.025, 0.025, 0.025, 0.025]))
        elif history == 'Good':
            days_past_due.append(np.random.choice([0, np.random.randint(1, 15), np.random.randint(15, 30)], p=[0.7, 0.2, 0.1]))
        elif history == 'Fair':
            days_past_due.append(np.random.choice([0, np.random.randint(1, 30), np.random.randint(30, 60)], p=[0.5, 0.3, 0.2]))
        elif history == 'Poor':
            days_past_due.append(np.random.choice([np.random.randint(1, 30), np.random.randint(30, 90), np.random.randint(90, 120)], p=[0.3, 0.4, 0.3]))
        else:  # Very Poor
            days_past_due.append(np.random.choice([np.random.randint(30, 90), np.random.randint(90, 180), np.random.randint(180, 365)], p=[0.2, 0.4, 0.4]))

    # Previous defaults
    previous_defaults = []
    for credit, history in zip(credit_scores, payment_histories):
        if credit < 500 or history in ['Poor', 'Very Poor']:
            previous_defaults.append(np.random.choice([0, 1, 2, 3], p=[0.2, 0.4, 0.3, 0.1]))
        elif credit < 650:
            previous_defaults.append(np.random.choice([0, 1], p=[0.8, 0.2]))
        else:
            previous_defaults.append(np.random.choice([0, 1], p=[0.95, 0.05]))

    # Recovery status (target variable)
    recovery_status = []
    for credit, history, dpd, defaults in zip(credit_scores, payment_histories, days_past_due, previous_defaults):
        # Factors affecting recovery:
        # 1. Credit score
        # 2. Payment history
        # 3. Days past due
        # 4. Previous defaults

        recovery_prob = 0.9  # Base probability

        # Adjust based on credit score
        if credit < 500:
            recovery_prob -= 0.3
        elif credit < 650:
            recovery_prob -= 0.1

        # Adjust based on payment history
        if history == 'Very Poor':
            recovery_prob -= 0.4
        elif history == 'Poor':
            recovery_prob -= 0.2
        elif history == 'Fair':
            recovery_prob -= 0.1

        # Adjust based on days past due
        if dpd > 180:
            recovery_prob -= 0.4
        elif dpd > 90:
            recovery_prob -= 0.3
        elif dpd > 30:
            recovery_prob -= 0.15
        elif dpd > 0:
            recovery_prob -= 0.05

        # Adjust based on previous defaults
        recovery_prob -= 0.1 * defaults

        # Ensure probability is between 0 and 1
        recovery_prob = max(0.05, min(0.95, recovery_prob))

        recovery_status.append(np.random.choice([1, 0], p=[recovery_prob, 1-recovery_prob]))

    # Create DataFrame
    data = {
        'customer_id': customer_ids,
        'age': ages,
        'gender': genders,
        'employment_status': employment_statuses,
        'annual_income': annual_incomes,
        'credit_score': credit_scores,
        'loan_amount': loan_amounts,
        'interest_rate': interest_rates,
        'loan_term': loan_terms,
        'payment_history': payment_histories,
        'days_past_due': days_past_due,
        'previous_defaults': previous_defaults,
        'recovery_status': recovery_status  # 1 = recovered, 0 = not recovered
    }

    df = pd.DataFrame(data)

    # Add some additional calculated features
    df['monthly_payment'] = (df['loan_amount'] * (df['interest_rate']/100/12) *
                            (1 + df['interest_rate']/100/12)**(df['loan_term'])) / \
                            ((1 + df['interest_rate']/100/12)**(df['loan_term']) - 1)

    df['debt_to_income'] = (df['monthly_payment'] * 12) / df['annual_income']

    # Round numeric columns for readability
    df['annual_income'] = df['annual_income'].round(2)
    df['loan_amount'] = df['loan_amount'].round(2)
    df['interest_rate'] = df['interest_rate'].round(2)
    df['monthly_payment'] = df['monthly_payment'].round(2)
    df['debt_to_income'] = df['debt_to_income'].round(4)

    return df

if __name__ == "__main__":
    # Generate sample data
    loan_data = generate_loan_data(n_samples=1000)

    # Save to CSV
    import os
    os.makedirs('data', exist_ok=True)
    loan_data.to_csv('data/loan_data.csv', index=False)
    print(f"Generated {len(loan_data)} loan records and saved to data/loan_data.csv")

    # Display sample
    print("\nSample data:")
    print(loan_data.head())

    # Display summary statistics
    print("\nSummary statistics:")
    print(loan_data.describe())

    # Display recovery rate
    recovery_rate = loan_data['recovery_status'].mean() * 100
    print(f"\nOverall recovery rate: {recovery_rate:.2f}%")
