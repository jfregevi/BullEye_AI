import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import io

# Page configuration
st.set_page_config(
    page_title="Investment Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Investment profiles configuration
INVESTMENT_PROFILES = {
    "Prudent": {
        "risk_tolerance": 0.3,
        "rebalance_threshold": 0.15,
        "max_single_asset": 0.40,
        "preferred_allocation": {"BTC": 0.20, "ETH": 0.15, "S&P500": 0.45, "Bonds": 0.20}
    },
    "Balanced": {
        "risk_tolerance": 0.5,
        "rebalance_threshold": 0.20,
        "max_single_asset": 0.50,
        "preferred_allocation": {"BTC": 0.30, "ETH": 0.20, "S&P500": 0.35, "Tesla": 0.15}
    },
    "Aggressive": {
        "risk_tolerance": 0.8,
        "rebalance_threshold": 0.25,
        "max_single_asset": 0.60,
        "preferred_allocation": {"BTC": 0.40, "ETH": 0.25, "Tesla": 0.20, "S&P500": 0.15}
    }
}

# Sample market data (in production, this would come from APIs)
SAMPLE_PRICES = {
    "BTC": {"current": 45000, "history": np.random.normal(45000, 5000, 252).tolist()},
    "ETH": {"current": 3000, "history": np.random.normal(3000, 400, 252).tolist()},
    "S&P500": {"current": 4500, "history": np.random.normal(4500, 200, 252).tolist()},
    "Tesla": {"current": 250, "history": np.random.normal(250, 30, 252).tolist()},
    "Bonds": {"current": 100, "history": np.random.normal(100, 5, 252).tolist()}
}

# AI signals (simulated)
AI_SIGNALS = {
    "BTC": {"sentiment": "Bullish", "strength": 0.8, "reason": "Strong institutional adoption"},
    "ETH": {"sentiment": "Neutral", "strength": 0.5, "reason": "Awaiting major upgrades"},
    "S&P500": {"sentiment": "Bullish", "strength": 0.6, "reason": "Economic recovery indicators"},
    "Tesla": {"sentiment": "Bearish", "strength": -0.3, "reason": "Competition concerns"},
    "Bonds": {"sentiment": "Neutral", "strength": 0.1, "reason": "Stable interest rates"}
}

class PortfolioManager:
    def __init__(self):
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = {}
        if 'investment_history' not in st.session_state:
            st.session_state.investment_history = []
    
    def add_position(self, asset, amount, price):
        if asset in st.session_state.portfolio:
            current_value = st.session_state.portfolio[asset]['shares'] * st.session_state.portfolio[asset]['avg_price']
            new_shares = amount / price
            total_shares = st.session_state.portfolio[asset]['shares'] + new_shares
            new_avg_price = (current_value + amount) / total_shares
            st.session_state.portfolio[asset] = {'shares': total_shares, 'avg_price': new_avg_price}
        else:
            st.session_state.portfolio[asset] = {'shares': amount / price, 'avg_price': price}
    
    def get_portfolio_value(self):
        total_value = 0
        for asset, position in st.session_state.portfolio.items():
            if asset in SAMPLE_PRICES:
                total_value += position['shares'] * SAMPLE_PRICES[asset]['current']
        return total_value
    
    def get_portfolio_allocation(self):
        total_value = self.get_portfolio_value()
        if total_value == 0:
            return {}
        
        allocation = {}
        for asset, position in st.session_state.portfolio.items():
            if asset in SAMPLE_PRICES:
                value = position['shares'] * SAMPLE_PRICES[asset]['current']
                allocation[asset] = value / total_value
        return allocation

def create_portfolio_chart(portfolio_manager):
    allocation = portfolio_manager.get_portfolio_allocation()
    if not allocation:
        return go.Figure().add_annotation(text="No portfolio data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    fig = px.pie(
        values=list(allocation.values()),
        names=list(allocation.keys()),
        title="Current Portfolio Allocation"
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_performance_chart():
    # Generate sample performance data
    dates = pd.date_range(start=datetime.now() - timedelta(days=252), end=datetime.now(), freq='D')
    portfolio_values = np.cumsum(np.random.normal(0, 50, len(dates))) + 10000
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=portfolio_values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title="Portfolio Performance Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (€)",
        hovermode='x unified'
    )
    return fig

def simulate_dca(amount, period, months, use_ai=False):
    """Simulate DCA strategy with optional AI enhancement"""
    periods_per_month = {"weekly": 4, "monthly": 1, "quarterly": 0.33}
    investments_per_month = periods_per_month[period]
    amount_per_investment = amount / investments_per_month
    
    results = []
    total_invested = 0
    total_shares = 0
    
    for month in range(months):
        investments_this_month = int(investments_per_month) if period != "quarterly" else (1 if month % 3 == 0 else 0)
        
        for _ in range(investments_this_month):
            # Simulate price (in production, use real historical data)
            price = 45000 + np.random.normal(0, 2000)  # BTC example
            
            if use_ai:
                # AI adjustment based on sentiment
                ai_multiplier = 1 + (AI_SIGNALS["BTC"]["strength"] * 0.2)
                investment_amount = amount_per_investment * ai_multiplier
            else:
                investment_amount = amount_per_investment
            
            shares_bought = investment_amount / price
            total_invested += investment_amount
            total_shares += shares_bought
            
            results.append({
                "month": month,
                "price": price,
                "invested": investment_amount,
                "shares": shares_bought,
                "total_value": total_shares * price
            })
    
    return pd.DataFrame(results)

def generate_allocation_suggestion(profile, dca_amount):
    """Generate AI-powered allocation suggestion"""
    base_allocation = INVESTMENT_PROFILES[profile]["preferred_allocation"]
    suggestions = {}
    
    for asset, base_weight in base_allocation.items():
        if asset in AI_SIGNALS:
            # Adjust based on AI signal
            ai_adjustment = AI_SIGNALS[asset]["strength"] * 0.1
            adjusted_weight = max(0.05, min(0.6, base_weight + ai_adjustment))
            suggestions[asset] = {
                "amount": round(dca_amount * adjusted_weight, 2),
                "weight": adjusted_weight,
                "reason": AI_SIGNALS[asset]["reason"]
            }
    
    # Normalize weights to sum to 1
    total_weight = sum(s["weight"] for s in suggestions.values())
    for asset in suggestions:
        suggestions[asset]["weight"] /= total_weight
        suggestions[asset]["amount"] = round(dca_amount * suggestions[asset]["weight"], 2)
    
    return suggestions

def create_alerts():
    """Generate investment alerts based on AI signals and portfolio"""
    alerts = []
    
    for asset, signal in AI_SIGNALS.items():
        if abs(signal["strength"]) > 0.6:  # Strong signal
            alert_type = "Opportunity" if signal["strength"] > 0 else "Warning"
            alerts.append({
                "type": alert_type,
                "asset": asset,
                "message": f"{asset} shows {signal['sentiment']} signal: {signal['reason']}",
                "strength": abs(signal["strength"])
            })
    
    return sorted(alerts, key=lambda x: x["strength"], reverse=True)

def chatbot_response(question, portfolio_manager):
    """Simple chatbot responses based on keywords"""
    question_lower = question.lower()
    
    if "why reinforce" in question_lower or "why invest" in question_lower:
        asset = None
        for a in ["btc", "eth", "tesla", "s&p500"]:
            if a in question_lower:
                asset = a.upper()
                if asset == "S&P500":
                    asset = "S&P500"
                break
        
        if asset and asset in AI_SIGNALS:
            signal = AI_SIGNALS[asset]
            return f"Based on AI analysis, {asset} shows a {signal['sentiment']} signal with strength {signal['strength']:.1f}. Reason: {signal['reason']}"
    
    elif "return" in question_lower or "performance" in question_lower:
        # Calculate mock return
        mock_return = np.random.uniform(-5, 15)
        return f"Your portfolio has generated a {mock_return:.1f}% return over the analyzed period. This is {'above' if mock_return > 5 else 'below'} market average."
    
    elif "new asset" in question_lower or "interesting" in question_lower:
        return "Based on current market analysis, consider diversifying into emerging markets ETFs or renewable energy stocks. Always ensure this aligns with your risk profile."
    
    elif "allocation" in question_lower:
        total_value = portfolio_manager.get_portfolio_value()
        return f"Your current portfolio value is €{total_value:,.2f}. Based on your profile, consider rebalancing if any single asset exceeds your risk limits."
    
    else:
        return "I can help you with portfolio analysis, investment suggestions, and performance tracking. Try asking about specific assets, returns, or allocation advice!"

def main():
    st.title("💰 AI-Powered Investment Assistant")
    st.markdown("---")
    
    # Initialize portfolio manager
    portfolio_manager = PortfolioManager()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Investment Profile
        profile = st.selectbox(
            "Investment Profile",
            ["Prudent", "Balanced", "Aggressive"],
            help="Choose your risk tolerance level"
        )
        
        # DCA Settings
        st.subheader("DCA Settings")
        dca_amount = st.number_input("Monthly DCA Amount (€)", min_value=50, max_value=10000, value=200, step=50)
        dca_period = st.selectbox("DCA Frequency", ["weekly", "monthly", "quarterly"])
        
        # Available Assets
        st.subheader("Available Assets")
        available_assets = list(SAMPLE_PRICES.keys())
        selected_assets = st.multiselect(
            "Select assets to track",
            available_assets,
            default=list(INVESTMENT_PROFILES[profile]["preferred_allocation"].keys())
        )
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Portfolio", "📈 DCA Simulation", "🚨 Alerts", "🤖 AI Chatbot"])
    
    with tab1:
        st.header("Portfolio Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Portfolio input section
            st.subheader("Add Position")
            with st.form("add_position"):
                pos_asset = st.selectbox("Asset", selected_assets)
                pos_amount = st.number_input("Amount Invested (€)", min_value=0.0, step=10.0)
                pos_price = st.number_input("Purchase Price", min_value=0.0, value=float(SAMPLE_PRICES.get(pos_asset, {}).get('current', 0)))
                
                if st.form_submit_button("Add Position"):
                    if pos_amount > 0 and pos_price > 0:
                        portfolio_manager.add_position(pos_asset, pos_amount, pos_price)
                        st.success(f"Added {pos_asset} position!")
                        st.rerun()
            
            # CSV Upload
            st.subheader("Upload Portfolio CSV")
            uploaded_file = st.file_uploader("Choose CSV file", type="csv")
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write("Preview:", df.head())
                    if st.button("Import Portfolio"):
                        for _, row in df.iterrows():
                            if all(col in df.columns for col in ['asset', 'amount', 'price']):
                                portfolio_manager.add_position(row['asset'], row['amount'], row['price'])
                        st.success("Portfolio imported successfully!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error reading CSV: {str(e)}")
        
        with col2:
            # Portfolio metrics
            total_value = portfolio_manager.get_portfolio_value()
            st.metric("Total Portfolio Value", f"€{total_value:,.2f}")
            
            if st.session_state.portfolio:
                total_invested = sum(pos['shares'] * pos['avg_price'] for pos in st.session_state.portfolio.values())
                pnl = total_value - total_invested
                pnl_pct = (pnl / total_invested * 100) if total_invested > 0 else 0
                st.metric("P&L", f"€{pnl:,.2f}", f"{pnl_pct:+.1f}%")
        
        # Portfolio visualization
        if st.session_state.portfolio:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_portfolio_chart(portfolio_manager), use_container_width=True)
            with col2:
                st.plotly_chart(create_performance_chart(), use_container_width=True)
            
            # Portfolio table
            st.subheader("Position Details")
            portfolio_data = []
            for asset, position in st.session_state.portfolio.items():
                if asset in SAMPLE_PRICES:
                    current_price = SAMPLE_PRICES[asset]['current']
                    current_value = position['shares'] * current_price
                    pnl = current_value - (position['shares'] * position['avg_price'])
                    portfolio_data.append({
                        'Asset': asset,
                        'Shares': f"{position['shares']:.4f}",
                        'Avg Price': f"€{position['avg_price']:.2f}",
                        'Current Price': f"€{current_price:.2f}",
                        'Current Value': f"€{current_value:.2f}",
                        'P&L': f"€{pnl:.2f}"
                    })
            
            if portfolio_data:
                st.dataframe(pd.DataFrame(portfolio_data), use_container_width=True)
    
    with tab2:
        st.header("DCA Simulation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Simulation Parameters")
            sim_months = st.slider("Simulation Period (months)", 6, 60, 12)
            comparison_amounts = st.multiselect(
                "Compare DCA amounts (€)",
                [100, 200, 300, 500, 1000],
                default=[200, 400]
            )
        
        with col2:
            st.subheader("AI Enhancement")
            use_ai = st.checkbox("Enable AI-enhanced DCA", value=True)
            st.info("AI enhancement adjusts investment amounts based on market sentiment and technical indicators.")
        
        if st.button("Run Simulation"):
            results = {}
            for amount in comparison_amounts:
                results[f"€{amount} Standard"] = simulate_dca(amount, dca_period, sim_months, False)
                if use_ai:
                    results[f"€{amount} AI-Enhanced"] = simulate_dca(amount, dca_period, sim_months, True)
            
            # Plot results
            fig = go.Figure()
            for label, data in results.items():
                fig.add_trace(go.Scatter(
                    x=data['month'],
                    y=data['total_value'],
                    mode='lines+markers',
                    name=label
                ))
            
            fig.update_layout(
                title="DCA Strategy Comparison",
                xaxis_title="Month",
                yaxis_title="Portfolio Value (€)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.subheader("Simulation Summary")
            summary_data = []
            for label, data in results.items():
                final_value = data['total_value'].iloc[-1]
                total_invested = data['invested'].sum()
                total_return = (final_value - total_invested) / total_invested * 100
                summary_data.append({
                    'Strategy': label,
                    'Final Value': f"€{final_value:,.2f}",
                    'Total Invested': f"€{total_invested:,.2f}",
                    'Total Return': f"{total_return:+.1f}%"
                })
            
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    
    with tab3:
        st.header("Market Alerts & Recommendations")
        
        # Current month allocation suggestion
        st.subheader(f"Suggested Allocation for {dca_amount}€ Investment")
        suggestions = generate_allocation_suggestion(profile, dca_amount)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            suggestion_data = []
            for asset, suggestion in suggestions.items():
                suggestion_data.append({
                    'Asset': asset,
                    'Amount': f"€{suggestion['amount']:.2f}",
                    'Weight': f"{suggestion['weight']:.1%}",
                    'Reason': suggestion['reason']
                })
            
            st.dataframe(pd.DataFrame(suggestion_data), use_container_width=True)
        
        with col2:
            # Pie chart of suggested allocation
            fig = px.pie(
                values=[s['amount'] for s in suggestions.values()],
                names=list(suggestions.keys()),
                title="Suggested Allocation"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Alerts
        st.subheader("Market Alerts")
        alerts = create_alerts()
        
        for alert in alerts:
            alert_type = alert['type']
            icon = "🚨" if alert_type == "Warning" else "💡"
            color = "red" if alert_type == "Warning" else "green"
            
            with st.container():
                st.markdown(f"""
                <div style="padding: 10px; border-left: 4px solid {color}; background-color: rgba(0,0,0,0.05); margin: 5px 0;">
                    {icon} <strong>{alert_type}:</strong> {alert['message']}
                </div>
                """, unsafe_allow_html=True)
    
    with tab4:
        st.header("AI Investment Chatbot")
        st.markdown("Ask me about your portfolio, market conditions, or investment strategies!")
        
        # Chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for chat in st.session_state.chat_history:
            with st.container():
                st.markdown(f"**You:** {chat['question']}")
                st.markdown(f"**AI:** {chat['response']}")
                st.markdown("---")
        
        # Chat input
        question = st.text_input("Ask your question:", placeholder="Why reinforce BTC this month?")
        
        if st.button("Ask") and question:
            response = chatbot_response(question, portfolio_manager)
            st.session_state.chat_history.append({"question": question, "response": response})
            st.rerun()
        
        # Sample questions
        st.subheader("Sample Questions")
        sample_questions = [
            "Why reinforce BTC this month?",
            "What is my cumulative return over 6 months?",
            "Which new asset looks interesting now?",
            "Should I rebalance my portfolio?",
            "What's the market outlook for Tesla?"
        ]
        
        for sq in sample_questions:
            if st.button(sq, key=f"sample_{sq}"):
                response = chatbot_response(sq, portfolio_manager)
                st.session_state.chat_history.append({"question": sq, "response": response})
                st.rerun()

if __name__ == "__main__":
    main()
