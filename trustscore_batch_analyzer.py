import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class TrustScoreAnalyzer:
    """
    Python implementation of the TrustScore algorithm for batch processing
    """
    
    def __init__(self):
        self.employment_scores = {
            'fulltime': 85,
            'selfemployed': 65,
            'parttime': 45,
            'unemployed': 10,
            'student': 25
        }
        
        self.loan_purpose_risk = {
            'home': 4,
            'car': 3,
            'education': 3,
            'medical': 2,
            'other': 1
        }
    
    def calculate_monthly_payment(self, loan_amount: float, term_years: int, apr: float = 0.06) -> float:
        """Calculate monthly loan payment"""
        monthly_rate = apr / 12
        num_payments = term_years * 12
        
        if monthly_rate == 0:
            return loan_amount / num_payments
        
        return loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / \
               ((1 + monthly_rate)**num_payments - 1)
    
    def calculate_trustscore_single(self, data: Dict) -> Dict:
        """Calculate TrustScore for a single record"""
        
        # Extract and convert data
        income = float(data.get('income', 0))
        credit_score = float(data.get('credit_score', 300))
        loan_amount = float(data.get('loan_amount', 0))
        age = int(data.get('age', 18))
        monthly_debt = float(data.get('monthly_debt', 0))
        loan_term = int(data.get('loan_term', 1))
        employment = data.get('employment', 'unemployed')
        loan_purpose = data.get('loan_purpose', 'other')
        gender = data.get('gender', 'prefer_not')
        residence = data.get('residence', 'rent')
        
        # Calculate financial ratios
        monthly_income = income / 12 if income > 0 else 0.01
        debt_to_income_ratio = (monthly_debt / monthly_income) * 100
        loan_to_income_ratio = (loan_amount / income) * 100 if income > 0 else 1000
        monthly_loan_payment = self.calculate_monthly_payment(loan_amount, loan_term)
        total_debt_ratio = ((monthly_debt + monthly_loan_payment) / monthly_income) * 100
        
        # === CREDIT RISK ASSESSMENT ===
        credit_risk_score = 0
        
        # Credit Score Impact (40% of decision)
        if credit_score >= 800:
            credit_risk_score += 40
        elif credit_score >= 750:
            credit_risk_score += 35
        elif credit_score >= 700:
            credit_risk_score += 30
        elif credit_score >= 650:
            credit_risk_score += 25
        elif credit_score >= 600:
            credit_risk_score += 20
        elif credit_score >= 550:
            credit_risk_score += 15
        elif credit_score >= 500:
            credit_risk_score += 10
        else:
            credit_risk_score += 5
        
        # Debt-to-Income Ratio Impact (25% of decision)
        if total_debt_ratio <= 28:
            credit_risk_score += 25
        elif total_debt_ratio <= 36:
            credit_risk_score += 20
        elif total_debt_ratio <= 43:
            credit_risk_score += 15
        elif total_debt_ratio <= 50:
            credit_risk_score += 10
        elif total_debt_ratio <= 60:
            credit_risk_score += 5
        
        # Employment Stability (20% of decision)
        employment_score = self.employment_scores.get(employment, 40)
        credit_risk_score += (employment_score / 100) * 20
        
        # Income Level (10% of decision)
        if income >= 100000:
            credit_risk_score += 10
        elif income >= 70000:
            credit_risk_score += 8
        elif income >= 50000:
            credit_risk_score += 6
        elif income >= 35000:
            credit_risk_score += 4
        elif income >= 25000:
            credit_risk_score += 2
        
        # Loan Purpose Risk (5% of decision)
        purpose_risk = self.loan_purpose_risk.get(loan_purpose, 1)
        credit_risk_score += purpose_risk
        
        # === FAIRNESS SCORE ===
        fairness_score = 70  # Base score
        
        if gender == 'prefer_not':
            fairness_score += 10
        if 25 <= age <= 65:
            fairness_score += 15
        elif 18 <= age <= 24:
            fairness_score += 10
        else:
            fairness_score += 5
        
        if residence == 'own':
            fairness_score += 5
        elif residence == 'rent':
            fairness_score += 3
        
        fairness_score = min(fairness_score, 100)
        
        # === TRANSPARENCY SCORE ===
        transparency_score = 60  # Base score
        
        if credit_score >= 700:
            transparency_score += 15
        elif credit_score >= 600:
            transparency_score += 10
        else:
            transparency_score += 5
        
        if debt_to_income_ratio <= 30:
            transparency_score += 10
        elif debt_to_income_ratio <= 40:
            transparency_score += 7
        else:
            transparency_score += 3
        
        if income >= 50000:
            transparency_score += 8
        elif income >= 30000:
            transparency_score += 5
        else:
            transparency_score += 2
        
        transparency_score = min(transparency_score, 100)
        
        # === PRIVACY SCORE ===
        privacy_score = 65  # Base score
        
        if gender == 'prefer_not':
            privacy_score += 15
        else:
            privacy_score += 5
        
        privacy_adjustments = {
            'medical': 10,
            'education': 8,
            'home': 6
        }
        privacy_score += privacy_adjustments.get(loan_purpose, 4)
        
        if employment == 'fulltime':
            privacy_score += 10
        elif employment == 'selfemployed':
            privacy_score += 5
        else:
            privacy_score += 7
        
        privacy_score = min(privacy_score, 100)
        
        # === OVERALL SCORE ===
        overall_score = round(
            (credit_risk_score * 0.6) +
            (fairness_score * 0.2) +
            (transparency_score * 0.15) +
            (privacy_score * 0.05)
        )
        overall_score = max(0, min(100, overall_score))
        
        # Decision categorization
        if overall_score >= 75 and credit_risk_score >= 70:
            decision = "HIGH_APPROVAL"
            risk_category = "Low Risk"
        elif overall_score >= 60 and credit_risk_score >= 50:
            decision = "MODERATE_APPROVAL"
            risk_category = "Medium Risk"
        elif overall_score >= 40 and credit_risk_score >= 30:
            decision = "LOW_APPROVAL"
            risk_category = "High Risk"
        else:
            decision = "LIKELY_DECLINE"
            risk_category = "Very High Risk"
        
        return {
            'overall_score': overall_score,
            'fairness_score': fairness_score,
            'transparency_score': transparency_score,
            'privacy_score': privacy_score,
            'credit_risk_score': credit_risk_score,
            'decision': decision,
            'risk_category': risk_category,
            'debt_to_income_ratio': debt_to_income_ratio,
            'total_debt_ratio': total_debt_ratio,
            'loan_to_income_ratio': loan_to_income_ratio,
            'monthly_payment': monthly_loan_payment,
            'employment_score': employment_score
        }
    
    def analyze_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process entire dataset and return results"""
        
        print(f"Processing {len(df)} records...")
        
        # Apply TrustScore calculation to each row
        results = []
        for idx, row in df.iterrows():
            try:
                score_result = self.calculate_trustscore_single(row.to_dict())
                score_result['record_id'] = idx
                results.append(score_result)
            except Exception as e:
                print(f"Error processing record {idx}: {e}")
                continue
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Merge with original data
        final_df = df.reset_index().merge(
            results_df, 
            left_index=True, 
            right_on='record_id', 
            how='left'
        )
        
        return final_df
    
    def generate_summary_report(self, results_df: pd.DataFrame) -> Dict:
        """Generate comprehensive analysis report"""
        
        total_records = len(results_df)
        
        # Decision distribution
        decision_counts = results_df['decision'].value_counts()
        decision_percentages = (decision_counts / total_records * 100).round(2)
        
        # Score statistics
        score_stats = {
            'overall_score': results_df['overall_score'].describe(),
            'fairness_score': results_df['fairness_score'].describe(),
            'transparency_score': results_df['transparency_score'].describe(),
            'privacy_score': results_df['privacy_score'].describe()
        }
        
        # Risk analysis
        risk_distribution = results_df['risk_category'].value_counts()
        
        # Bias analysis
        bias_analysis = {}
        
        # Age bias check
        if 'age' in results_df.columns:
            age_groups = pd.cut(results_df['age'], bins=[18, 30, 45, 60, 100], labels=['18-30', '31-45', '46-60', '60+'])
            age_scores = results_df.groupby(age_groups)['overall_score'].mean()
            bias_analysis['age_bias'] = {
                'max_difference': age_scores.max() - age_scores.min(),
                'by_group': age_scores.to_dict()
            }
        
        # Gender bias check
        if 'gender' in results_df.columns:
            gender_scores = results_df.groupby('gender')['overall_score'].mean()
            bias_analysis['gender_bias'] = gender_scores.to_dict()
        
        # Income bias analysis
        if 'income' in results_df.columns:
            income_quartiles = pd.qcut(results_df['income'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            income_scores = results_df.groupby(income_quartiles)['overall_score'].mean()
            bias_analysis['income_quartile_scores'] = income_scores.to_dict()
        
        return {
            'total_records': total_records,
            'decision_distribution': decision_counts.to_dict(),
            'decision_percentages': decision_percentages.to_dict(),
            'score_statistics': score_stats,
            'risk_distribution': risk_distribution.to_dict(),
            'bias_analysis': bias_analysis,
            'high_risk_count': len(results_df[results_df['overall_score'] < 40]),
            'approval_rate': len(results_df[results_df['decision'].isin(['HIGH_APPROVAL', 'MODERATE_APPROVAL'])]) / total_records * 100
        }
    
    def create_visualizations(self, results_df: pd.DataFrame, save_path: str = None):
        """Create comprehensive visualizations"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('TrustScore Batch Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. Score Distribution
        axes[0, 0].hist(results_df['overall_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Overall Score Distribution')
        axes[0, 0].set_xlabel('TrustScore')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(results_df['overall_score'].mean(), color='red', linestyle='--', label=f'Mean: {results_df["overall_score"].mean():.1f}')
        axes[0, 0].legend()
        
        # 2. Decision Categories
        decision_counts = results_df['decision'].value_counts()
        axes[0, 1].pie(decision_counts.values, labels=decision_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Decision Distribution')
        
        # 3. Risk Categories
        risk_counts = results_df['risk_category'].value_counts()
        axes[0, 2].bar(risk_counts.index, risk_counts.values, color=['green', 'yellow', 'orange', 'red'])
        axes[0, 2].set_title('Risk Category Distribution')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Score Components Comparison
        score_components = ['overall_score', 'fairness_score', 'transparency_score', 'privacy_score']
        box_data = [results_df[col] for col in score_components]
        axes[1, 0].boxplot(box_data, labels=[col.replace('_', '\n') for col in score_components])
        axes[1, 0].set_title('Score Components Distribution')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Credit Score vs TrustScore
        if 'credit_score' in results_df.columns:
            axes[1, 1].scatter(results_df['credit_score'], results_df['overall_score'], alpha=0.6)
            axes[1, 1].set_title('Credit Score vs TrustScore')
            axes[1, 1].set_xlabel('Credit Score')
            axes[1, 1].set_ylabel('TrustScore')
            
            # Add correlation
            correlation = results_df['credit_score'].corr(results_df['overall_score'])
            axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                           transform=axes[1, 1].transAxes, fontsize=10, 
                           bbox=dict(boxstyle='round', facecolor='wheat'))
        
        # 6. Debt Ratio Impact
        if 'total_debt_ratio' in results_df.columns:
            axes[1, 2].scatter(results_df['total_debt_ratio'], results_df['overall_score'], alpha=0.6, color='orange')
            axes[1, 2].set_title('Total Debt Ratio vs TrustScore')
            axes[1, 2].set_xlabel('Total Debt Ratio (%)')
            axes[1, 2].set_ylabel('TrustScore')
            axes[1, 2].axvline(43, color='red', linestyle='--', label='43% Limit')
            axes[1, 2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualizations saved to {save_path}")
        
        plt.show()

def create_sample_data(n_records: int = 100) -> pd.DataFrame:
    """Create sample dataset for testing"""
    
    np.random.seed(42)
    
    data = {
        'income': np.random.normal(50000, 20000, n_records).clip(15000, 150000),
        'credit_score': np.random.normal(650, 100, n_records).clip(300, 850),
        'loan_amount': np.random.normal(25000, 15000, n_records).clip(1000, 100000),
        'loan_term': np.random.choice([1, 2, 3, 5, 7, 10, 15, 20, 30], n_records),
        'age': np.random.randint(18, 75, n_records),
        'monthly_debt': np.random.normal(800, 400, n_records).clip(0, 3000),
        'employment': np.random.choice(['fulltime', 'selfemployed', 'parttime', 'unemployed', 'student'], 
                                     n_records, p=[0.6, 0.15, 0.15, 0.05, 0.05]),
        'loan_purpose': np.random.choice(['home', 'car', 'education', 'medical', 'other'], 
                                       n_records, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
        'gender': np.random.choice(['male', 'female', 'prefer_not'], n_records, p=[0.45, 0.45, 0.1]),
        'residence': np.random.choice(['own', 'rent', 'parents'], n_records, p=[0.4, 0.5, 0.1])
    }
    
    return pd.DataFrame(data)

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = TrustScoreAnalyzer()
    
    # Create or load your dataset
    # For demonstration, we'll create sample data
    df = create_sample_data(100)
    print("Sample data created:")
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    
    # Run batch analysis
    print("\n" + "="*50)
    print("RUNNING TRUSTSCORE BATCH ANALYSIS")
    print("="*50)
    
    results_df = analyzer.analyze_batch(df)
    
    # Generate summary report
    summary = analyzer.generate_summary_report(results_df)
    
    print(f"\nANALYSIS SUMMARY:")
    print("="*30)
    print(f"Total Records Processed: {summary['total_records']}")
    print(f"Overall Approval Rate: {summary['approval_rate']:.1f}%")
    print(f"High Risk Applications: {summary['high_risk_count']} ({summary['high_risk_count']/summary['total_records']*100:.1f}%)")
    
    print(f"\nDECISION BREAKDOWN:")
    for decision, count in summary['decision_distribution'].items():
        percentage = summary['decision_percentages'][decision]
        print(f"  {decision}: {count} ({percentage}%)")
    
    print(f"\nSCORE STATISTICS:")
    print(f"  Average TrustScore: {summary['score_statistics']['overall_score']['mean']:.1f}")
    print(f"  Score Range: {summary['score_statistics']['overall_score']['min']:.0f} - {summary['score_statistics']['overall_score']['max']:.0f}")
    
    if 'age_bias' in summary['bias_analysis']:
        print(f"\nBIAS ANALYSIS:")
        print(f"  Max Age Group Score Difference: {summary['bias_analysis']['age_bias']['max_difference']:.1f} points")
        if summary['bias_analysis']['age_bias']['max_difference'] > 10:
            print("  ⚠️  WARNING: Potential age bias detected!")
        else:
            print("  ✅ Age bias within acceptable range")
    
    # Create visualizations
    print(f"\nGenerating visualizations...")
    analyzer.create_visualizations(results_df, 'trustscore_analysis.png')
    
    # Save results to CSV
    results_df.to_csv('trustscore_results.csv', index=False)
    print(f"\nResults saved to 'trustscore_results.csv'")
    
    print(f"\nTop 10 Highest Scores:")
    print(results_df.nlargest(10, 'overall_score')[['overall_score', 'decision', 'risk_category', 'income', 'credit_score']].to_string())
    
    print(f"\nTop 10 Lowest Scores:")
    print(results_df.nsmallest(10, 'overall_score')[['overall_score', 'decision', 'risk_category', 'income', 'credit_score']].to_string())