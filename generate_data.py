import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

print("Starting data generation...")

# Load your existing products to get valid Product_IDs and prices
try:
    products_df = pd.read_csv('products.csv')
except FileNotFoundError:
    print("Error: products.csv not found. Please make sure it's in the same folder.")
    exit()

# --- Configuration ---
NUM_CUSTOMERS = 150
NUM_TRANSACTIONS = 1000
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime.now()

# --- Generate Data ---
customer_ids = [f"C-{i}" for i in range(1, NUM_CUSTOMERS + 1)]
product_ids = products_df['Product_ID'].tolist()

transactions = []
for _ in range(NUM_TRANSACTIONS):
    # Pick a random customer and product
    customer_id = random.choice(customer_ids)
    product_id = random.choice(product_ids)

    # Get product price from the original dataframe
    price = products_df.loc[products_df['Product_ID'] == product_id, 'Price'].iloc[0]

    # Generate a random date and quantity
    random_days = random.randint(0, (END_DATE - START_DATE).days)
    transaction_date = START_DATE + timedelta(days=random_days)
    quantity = random.randint(1, 4)

    transactions.append({
        'TransactionID': f"T-{1001 + _}",
        'CustomerID': customer_id,
        'ProductID': product_id,
        'TransactionDate': transaction_date,
        'Quantity': quantity,
        'PricePerItem': price,
        'TotalPrice': price * quantity
    })

# Create DataFrame and save
sales_df = pd.DataFrame(transactions)
sales_df.to_csv('sales_transactions.csv', index=False)

print(f"Successfully generated 'sales_transactions.csv' with {len(sales_df)} transactions.")