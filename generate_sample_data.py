import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def create_sample_retail_data():
    """Create a sample retail dataset with realistic patterns and anomalies"""
    
    # Generate date range
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Base parameters
    n_products = 50
    n_stores = 10
    n_customers = 1000
    
    # Product categories and their typical price ranges
    categories = {
        'Electronics': (100, 2000),
        'Clothing': (20, 200),
        'Groceries': (5, 50),
        'Home & Garden': (30, 500),
        'Sports': (25, 300)
    }
    
    # Generate products
    products = []
    for i in range(n_products):
        category = random.choice(list(categories.keys()))
        price_range = categories[category]
        base_price = random.uniform(price_range[0], price_range[1])
        
        products.append({
            'product_id': f'P{i+1:03d}',
            'category': category,
            'base_price': round(base_price, 2)
        })
    
    # Generate transactions
    transactions = []
    transaction_id = 1
    
    for date in date_range:
        # Vary daily transaction volume (higher on weekends, lower on weekdays)
        if date.weekday() >= 5:  # Weekend
            daily_transactions = random.randint(80, 150)
        else:  # Weekday
            daily_transactions = random.randint(40, 100)
        
        # Holiday boost (simulate Black Friday, Christmas season)
        if date.month == 11 and date.day >= 20:  # Black Friday season
            daily_transactions = int(daily_transactions * 2.5)
        elif date.month == 12:  # Christmas season
            daily_transactions = int(daily_transactions * 1.8)
        
        for _ in range(daily_transactions):
            product = random.choice(products)
            store_id = f'STORE_{random.randint(1, n_stores):02d}'
            customer_id = f'CUST_{random.randint(1, n_customers):04d}'
            
            # Calculate price with some variation
            price_variation = random.uniform(0.9, 1.1)
            unit_price = round(product['base_price'] * price_variation, 2)
            
            # Quantity (most items are 1-3, but some bulk purchases)
            if random.random() < 0.8:
                quantity = random.randint(1, 3)
            else:
                quantity = random.randint(4, 10)
            
            # Calculate total
            total_amount = round(unit_price * quantity, 2)
            
            # Add seasonal discount patterns
            discount = 0
            if date.month == 11 and date.day >= 20:  # Black Friday
                discount = random.uniform(0.1, 0.5)
            elif date.month == 12:  # Christmas
                discount = random.uniform(0.05, 0.3)
            elif random.random() < 0.1:  # Random promotions
                discount = random.uniform(0.05, 0.2)
            
            discounted_amount = round(total_amount * (1 - discount), 2)
            
            transaction = {
                'transaction_id': f'T{transaction_id:06d}',
                'date': date.strftime('%Y-%m-%d'),
                'timestamp': f"{date.strftime('%Y-%m-%d')} {random.randint(8, 22):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}",
                'store_id': store_id,
                'customer_id': customer_id,
                'product_id': product['product_id'],
                'category': product['category'],
                'quantity': quantity,
                'unit_price': unit_price,
                'total_amount': total_amount,
                'discount_percent': round(discount * 100, 2),
                'final_amount': discounted_amount,
                'payment_method': random.choice(['Credit Card', 'Debit Card', 'Cash', 'Mobile Pay'])
            }
            
            transactions.append(transaction)
            transaction_id += 1
    
    # Create DataFrame
    df = pd.DataFrame(transactions)
    
    # Inject some anomalies for testing
    df = inject_anomalies(df)
    
    return df

def inject_anomalies(df):
    """Inject realistic anomalies into the dataset"""
    
    df_anomalies = df.copy()
    
    # 1. Price anomalies (unusually high or low prices)
    price_anomaly_indices = df_anomalies.sample(n=20).index
    for idx in price_anomaly_indices:
        if random.random() < 0.5:
            # Extremely high price (data entry error or premium item)
            df_anomalies.loc[idx, 'unit_price'] *= random.uniform(5, 15)
            df_anomalies.loc[idx, 'total_amount'] = df_anomalies.loc[idx, 'unit_price'] * df_anomalies.loc[idx, 'quantity']
            df_anomalies.loc[idx, 'final_amount'] = df_anomalies.loc[idx, 'total_amount'] * (1 - df_anomalies.loc[idx, 'discount_percent']/100)
        else:
            # Extremely low price (clearance or error)
            df_anomalies.loc[idx, 'unit_price'] *= random.uniform(0.01, 0.1)
            df_anomalies.loc[idx, 'total_amount'] = df_anomalies.loc[idx, 'unit_price'] * df_anomalies.loc[idx, 'quantity']
            df_anomalies.loc[idx, 'final_amount'] = df_anomalies.loc[idx, 'total_amount'] * (1 - df_anomalies.loc[idx, 'discount_percent']/100)
    
    # 2. Quantity anomalies (unusually large quantities)
    quantity_anomaly_indices = df_anomalies.sample(n=15).index
    for idx in quantity_anomaly_indices:
        df_anomalies.loc[idx, 'quantity'] = random.randint(50, 200)
        df_anomalies.loc[idx, 'total_amount'] = df_anomalies.loc[idx, 'unit_price'] * df_anomalies.loc[idx, 'quantity']
        df_anomalies.loc[idx, 'final_amount'] = df_anomalies.loc[idx, 'total_amount'] * (1 - df_anomalies.loc[idx, 'discount_percent']/100)
    
    # 3. Discount anomalies (unusually high discounts)
    discount_anomaly_indices = df_anomalies.sample(n=10).index
    for idx in discount_anomaly_indices:
        df_anomalies.loc[idx, 'discount_percent'] = random.uniform(80, 99)
        df_anomalies.loc[idx, 'final_amount'] = df_anomalies.loc[idx, 'total_amount'] * (1 - df_anomalies.loc[idx, 'discount_percent']/100)
    
    # 4. Temporal anomalies (transactions at unusual times)
    time_anomaly_indices = df_anomalies.sample(n=25).index
    for idx in time_anomaly_indices:
        # Very early morning or very late night transactions
        unusual_hour = random.choice([2, 3, 4, 5, 23, 0, 1])
        date_part = df_anomalies.loc[idx, 'timestamp'].split(' ')[0]
        df_anomalies.loc[idx, 'timestamp'] = f"{date_part} {unusual_hour:02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}"
    
    # 5. Customer behavior anomalies (same customer, multiple high-value transactions in short time)
    customer_anomaly_customer = df_anomalies['customer_id'].sample(n=1).iloc[0]
    customer_transactions = df_anomalies[df_anomalies['customer_id'] == customer_anomaly_customer].head(5)
    
    for idx in customer_transactions.index:
        df_anomalies.loc[idx, 'final_amount'] *= random.uniform(10, 20)
        df_anomalies.loc[idx, 'total_amount'] = df_anomalies.loc[idx, 'final_amount'] / (1 - df_anomalies.loc[idx, 'discount_percent']/100)
        df_anomalies.loc[idx, 'unit_price'] = df_anomalies.loc[idx, 'total_amount'] / df_anomalies.loc[idx, 'quantity']
    
    return df_anomalies

if __name__ == "__main__":
    # Generate sample data
    print("Generating sample retail dataset...")
    df = create_sample_retail_data()
    
    # Save to CSV
    output_file = "sample_retail_data.csv"
    df.to_csv(output_file, index=False)
    
    print(f"Sample dataset created: {output_file}")
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total transactions: {len(df):,}")
    print(f"Unique products: {df['product_id'].nunique()}")
    print(f"Unique customers: {df['customer_id'].nunique()}")
    print(f"Categories: {df['category'].unique()}")
    
    # Show some statistics
    print(f"\nPrice statistics:")
    print(f"Unit price range: ${df['unit_price'].min():.2f} - ${df['unit_price'].max():.2f}")
    print(f"Final amount range: ${df['final_amount'].min():.2f} - ${df['final_amount'].max():.2f}")
    print(f"Average transaction: ${df['final_amount'].mean():.2f}")
