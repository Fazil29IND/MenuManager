"""
Customer Segmentation & Retention Dashboard
Author: S Amanullah Fazil
"""

import io
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Customer Segmentation & Retention Analyser - Menu Manager",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #ffffff; color: #000000; }
    header[data-testid="stHeader"] { background-color: #ffffff; }

    .main-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #000000;
        border-bottom: 2px solid #000000;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }

    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #000000;
        border-bottom: 1px solid #cccccc;
        padding-bottom: 0.3rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    .seg-card {
        background: #fafafa;
        border: 1px solid #d0d0d0;
        border-radius: 6px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }

    .seg-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: #000000;
        margin-bottom: 0.3rem;
    }

    .seg-desc {
        font-size: 0.85rem;
        color: #555555;
        margin-bottom: 0.8rem;
    }

    .seg-stat {
        font-size: 0.8rem;
        color: #333333;
    }

    .strategy-item {
        padding: 0.4rem 0;
        border-bottom: 1px solid #eeeeee;
        font-size: 0.9rem;
        color: #222222;
    }
</style>
""", unsafe_allow_html=True)

SEGMENTS = {
    0: {
        "name": "VIP",
        "desc": "High-value, loyal customers with frequent recent purchases.",
        "strategies": [
            "Provide premium, personalized service and exclusive offers.",
            "Offer early access to new products and VIP-only promotions.",
            "Create loyalty rewards with tiered benefits.",
            "Assign dedicated account managers for top-tier customers.",
            "Invite to exclusive events and product previews."
        ]
    },
    1: {
        "name": "Loyal Customers",
        "desc": "Consistent, regular buyers with moderate to high spending.",
        "strategies": [
            "Implement loyalty programs with points and rewards.",
            "Cross-sell complementary products based on purchase history.",
            "Send personalized product recommendations.",
            "Offer referral bonuses to encourage word-of-mouth.",
            "Provide birthday/anniversary special discounts."
        ]
    },
    2: {
        "name": "At-Risk",
        "desc": "Previously active customers showing declining engagement.",
        "strategies": [
            "Launch win-back email campaigns with special offers.",
            "Conduct surveys to understand reasons for disengagement.",
            "Offer time-limited comeback discounts (15–25% off).",
            "Re-engage through personalized product reminders.",
            "Consider direct outreach from customer success team."
        ]
    },
    3: {
        "name": "New Customers",
        "desc": "Recent or low-frequency buyers with growth potential.",
        "strategies": [
            "Create welcoming onboarding email sequences.",
            "Offer first-purchase follow-up discounts.",
            "Provide educational content about products/services.",
            "Encourage second purchase with targeted incentives.",
            "Collect feedback to improve first-time experience."
        ]
    }
}

def detect_columns(df):
    col_map = {k: None for k in ['customer_id', 'invoice_date', 'total_amount',
                                  'stock_code', 'invoice_no', 'quantity', 'unit_price']}
    for col in df.columns:
        cl = col.lower().replace('_', '').replace(' ', '')
        if cl in ('customerid', 'custid', 'customer'):
            col_map['customer_id'] = col
        elif cl in ('invoicedate', 'date', 'orderdate', 'transactiondate'):
            col_map['invoice_date'] = col
        elif cl in ('totalamount', 'total', 'amount', 'revenue', 'sales'):
            col_map['total_amount'] = col
        elif cl in ('stockcode', 'productid', 'itemid', 'sku', 'productcode'):
            col_map['stock_code'] = col
        elif cl in ('invoiceno', 'invoicenumber', 'orderid', 'transactionid'):
            col_map['invoice_no'] = col
        elif cl in ('quantity', 'qty'):
            col_map['quantity'] = col
        elif cl in ('unitprice', 'price', 'itemprice'):
            col_map['unit_price'] = col
    return col_map

@st.cache_data
def load_and_process(file_bytes, file_name):
    if file_name.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_excel(io.BytesIO(file_bytes))

    col_map = detect_columns(df)
    cid = col_map['customer_id']
    date_col = col_map['invoice_date']
    inv_col = col_map['invoice_no']
    stock_col = col_map['stock_code']

    if cid is None or date_col is None:
        return None, "Missing required columns: CustomerID or InvoiceDate."

    df = df.dropna(subset=[cid]).drop_duplicates()
    if inv_col and inv_col in df.columns:
        df = df[df[inv_col].astype(str).str.isnumeric()]

    ta = col_map['total_amount']
    qty = col_map['quantity']
    price = col_map['unit_price']
    
    if ta and ta in df.columns:
        df['TotalAmount'] = df[ta]
    elif qty and price and qty in df.columns and price in df.columns:
        df['TotalAmount'] = df[qty] * df[price]
    else:
        return None, "Need TotalAmount column or Quantity + UnitPrice columns."

    if price and price in df.columns:
        df = df[df[price] > 0]
    if qty and qty in df.columns:
        df = df[df[qty] >= 1]

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])

    snapshot = df[date_col].max() + pd.Timedelta(days=1)
    freq_col = inv_col if (inv_col and inv_col in df.columns) else date_col
    freq_agg = 'nunique' if (inv_col and inv_col in df.columns) else 'count'

    cust = df.groupby(cid).agg(
        Recency=(date_col, lambda x: (snapshot - x.max()).days),
        Frequency=(freq_col, freq_agg),
        Monetary=('TotalAmount', 'sum')
    ).reset_index()
    cust['AverageOrderValue'] = cust['Monetary'] / cust['Frequency']

    if stock_col and stock_col in df.columns:
        up = df.groupby(cid)[stock_col].nunique().reset_index(name='UniqueProducts')
        cust = cust.merge(up, on=cid, how='left')
    else:
        cust['UniqueProducts'] = cust['Frequency']

    features = ['Recency', 'Frequency', 'Monetary', 'AverageOrderValue', 'UniqueProducts']
    scaled = cust[features].copy()
    
    for f in features:
        scaled[f] = np.log1p(scaled[f])
        
    scaler = StandardScaler()
    scaled_arr = scaler.fit_transform(scaled)

    km = KMeans(n_clusters=4, random_state=42, n_init=10, max_iter=300)
    cust['Cluster'] = km.fit_predict(scaled_arr)

    profile = cust.groupby('Cluster').agg(
        r=('Recency', 'median'), f=('Frequency', 'median'), m=('Monetary', 'median')
    )
    
    for c in ['r', 'f', 'm']:
        mn, mx = profile[c].min(), profile[c].max()
        profile[c + 'n'] = (profile[c] - mn) / (mx - mn) if mx != mn else 0.5

    scores = []
    for idx in profile.index:
        s = (1 - profile.loc[idx, 'rn']) * 0.3 + profile.loc[idx, 'fn'] * 0.35 + profile.loc[idx, 'mn'] * 0.35
        scores.append((idx, s))
        
    scores.sort(key=lambda x: x[1], reverse=True)

    seg_map = {}
    for rank, (cluster_id, _) in enumerate(scores):
        seg_map[cluster_id] = rank

    cust['SegmentID'] = cust['Cluster'].map(seg_map)
    cust['Segment'] = cust['SegmentID'].map(lambda x: SEGMENTS[x]['name'])

    return cust, None

def main():
    st.markdown('<div class="main-title">Customer Segmentation & Retention Dashboard</div>', unsafe_allow_html=True)
    st.markdown("### Upload Data")
    
    uploaded = st.file_uploader("Upload CSV or Excel File", type=['csv', 'xlsx', 'xls'])

    if uploaded is None:
        st.markdown("Please upload your transaction data to see segments and retention recommendations.")
        return

    with st.spinner("Processing..."):
        cust, err = load_and_process(uploaded.read(), uploaded.name)

    if err:
        st.error(err)
        return

    st.markdown('<div class="section-header">Customer Segments</div>', unsafe_allow_html=True)

    for seg_id in range(4):
        seg = SEGMENTS[seg_id]
        data = cust[cust['SegmentID'] == seg_id]
        if len(data) == 0:
            continue

        st.markdown(f"""
        <div class="seg-card">
            <div class="seg-name">{seg['name']}</div>
            <div class="seg-desc">{seg['desc']}</div>
            <div class="seg-stat">
                Customers: <b>{len(data):,}</b> &nbsp;|&nbsp;
                Median Recency: <b>{data['Recency'].median():.0f} days</b> &nbsp;|&nbsp;
                Median Frequency: <b>{data['Frequency'].median():.0f}</b> &nbsp;|&nbsp;
                Median Monetary: <b>{data['Monetary'].median():,.2f}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Retention Recommendations</div>', unsafe_allow_html=True)

    for seg_id in range(4):
        seg = SEGMENTS[seg_id]
        data = cust[cust['SegmentID'] == seg_id]
        if len(data) == 0:
            continue

        st.markdown(f"**{seg['name']}**")
        for i, s in enumerate(seg['strategies'], 1):
            st.markdown(f'<div class="strategy-item">{i}. {s}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
