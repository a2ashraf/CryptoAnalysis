import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class CryptoAnalysisError(Exception):
    """Custom exception for cryptocurrency analysis errors"""
    pass

def get_crypto_candles(coin='bitcoin', base_currency='usd', days=365):
    """
    Fetch historical price data for a cryptocurrency using CoinGecko API
    
    :param coin: Cryptocurrency ID (lowercase)
    :param base_currency: Base currency (lowercase)
    :param days: Number of historical days to retrieve
    :return: DataFrame with price data
    """
    # Predefined mapping for known coin variations
    coin_mapping = {
        'velodrome': 'velodrome-finance',
        'velo': 'velodrome-finance',
        'velero': 'velodrome-finance'
    }
    
    # Check if coin has a known mapping
    original_coin = coin
    coin = coin_mapping.get(coin, coin)
    
    # CoinGecko API endpoint for historical market data
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency={base_currency}&days={days}"
    
    print(f"DEBUG: Attempting to fetch data from URL: {url}")
    
    try:
        response = requests.get(url)
        data = response.json()
        
        print(f"DEBUG: Received data keys: {data.keys()}")
        print(f"DEBUG: Prices length: {len(data.get('prices', []))}")
        
        # If no data, try searching for the coin
        if 'prices' not in data or len(data['prices']) == 0:
            # Try to find the correct coin ID
            search_url = f"https://api.coingecko.com/api/v3/search?query={original_coin}"
            search_response = requests.get(search_url)
            search_data = search_response.json()
            
            if search_data.get('coins') and len(search_data['coins']) > 0:
                # Use the first matching coin
                coin = search_data['coins'][0]['id']
                # Retry with the new coin ID
                response = requests.get(f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency={base_currency}&days={days}")
                data = response.json()
            
            if 'prices' not in data or len(data['prices']) == 0:
                raise CryptoAnalysisError(f"Unable to fetch data for {original_coin}. Check the cryptocurrency name.")
        
        # Convert price data to DataFrame
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
        
        # Print raw data for debugging
        print(f"DEBUG: DataFrame initial shape: {df.shape}")
        print(f"DEBUG: First few rows:\n{df.head()}")
        
        # Validate DataFrame
        if len(df) < 2:
            raise CryptoAnalysisError(f"Insufficient data points for {original_coin}. Need at least 2 data points.")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Calculate additional columns for wave analysis
        df['open'] = df['close'].shift(1)
        
        # Calculate high and low using rolling window instead of groupby
        df['high'] = df['close'].rolling(window=4, min_periods=1).max()
        df['low'] = df['close'].rolling(window=4, min_periods=1).min()
        
        # Drop first row (which will have NaN open due to shift)
        df = df.dropna()
        
        # Print processed DataFrame for debugging
        print(f"DEBUG: DataFrame after processing shape: {df.shape}")
        print(f"DEBUG: DataFrame columns after processing: {df.columns}")
        
        # Ensure at least 2 rows remain after processing
        if len(df) < 2:
            raise CryptoAnalysisError(f"Insufficient processed data points for {original_coin}.")
        
        return df
    except Exception as e:
        # Print full exception details for debugging
        import traceback
        traceback.print_exc()
        raise CryptoAnalysisError(f"Error fetching data for {original_coin}: {str(e)}")

def detect_elliot_waves(df, coin):
    """
    Advanced Elliot Wave detection with comprehensive analysis
    """
    # Calculate additional technical indicators
    df['price_change'] = df['close'].diff()
    df['pct_change'] = df['close'].pct_change()
    
    # Calculate RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    def identify_wave_pattern(series):
        """
        Sophisticated wave pattern identification with Elliot Wave principles
        """
        waves = {
            'impulse_waves': [],
            'corrective_waves': [],
            'wave_details': [],
            'wave_labels': []
        }
        
        # Detect significant price movements
        thresholds = {
            'minor': 0.02,   # 2% minor movement
            'significant': 0.05,  # 5% significant movement
            'major': 0.10    # 10% major movement
        }
        
        # Track wave progression
        current_wave_type = None
        wave_count = 0
        
        for i in range(1, len(series)):
            change = series[i]
            
            if abs(change) > thresholds['minor']:
                wave_type = 'impulse' if change > 0 else 'corrective'
                wave_magnitude = 'minor'
                
                if abs(change) > thresholds['significant']:
                    wave_magnitude = 'significant'
                
                if abs(change) > thresholds['major']:
                    wave_magnitude = 'major'
                
                # Determine wave label based on progression
                if wave_type == 'impulse':
                    if current_wave_type != 'impulse':
                        wave_count = 1
                        current_wave_type = 'impulse'
                    else:
                        wave_count += 1
                    
                    wave_label = f'Impulse {wave_count}' if wave_count <= 5 else 'Impulse Ext'
                else:
                    if current_wave_type != 'corrective':
                        wave_count = 1
                        current_wave_type = 'corrective'
                    else:
                        wave_count += 1
                    
                    wave_label = f'Correction {chr(64 + wave_count)}' if wave_count <= 3 else 'Correction Ext'
                
                wave_details = {
                    'type': wave_type,
                    'magnitude': wave_magnitude,
                    'start_index': i-1,
                    'end_index': i,
                    'percentage_change': change * 100,
                    'label': wave_label
                }
                
                waves['wave_details'].append(wave_details)
                waves['wave_labels'].append(wave_label)
                
                if wave_type == 'impulse':
                    waves['impulse_waves'].append(wave_details)
                else:
                    waves['corrective_waves'].append(wave_details)
        
        return waves
    
    # Analyze waves
    waves = identify_wave_pattern(df['pct_change'].values)
    
    # Visualize waves
    plt.figure(figsize=(20, 10))
    
    # Price plot
    plt.subplot(2, 1, 1)
    plt.plot(df['timestamp'], df['close'], label='Price', color='blue')
    plt.title(f'{coin.capitalize()} Price with Elliot Wave Analysis')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    
    # Highlight waves with safety checks and labels
    for wave in waves['wave_details']:
        start_index = max(0, min(wave['start_index'], len(df) - 1))
        end_index = max(0, min(wave['end_index'], len(df) - 1))
        
        color = 'green' if wave['type'] == 'impulse' else 'red'
        alpha = 0.1 if wave['magnitude'] == 'minor' else 0.3 if wave['magnitude'] == 'significant' else 0.5
        
        plt.axvspan(
            df['timestamp'].iloc[start_index], 
            df['timestamp'].iloc[end_index], 
            color=color, alpha=alpha
        )
        
        # Add wave labels
        mid_index = (start_index + end_index) // 2
        plt.text(
            df['timestamp'].iloc[mid_index], 
            df['close'].iloc[mid_index], 
            wave['label'], 
            fontsize=8, 
            color='black', 
            verticalalignment='bottom'
        )
    
    # RSI subplot
    plt.subplot(2, 1, 2)
    plt.plot(df['timestamp'], df['rsi'], label='RSI', color='purple')
    plt.title('Relative Strength Index (RSI)')
    plt.xlabel('Timestamp')
    plt.ylabel('RSI')
    plt.axhline(y=70, color='r', linestyle='--')  # Overbought line
    plt.axhline(y=30, color='g', linestyle='--')  # Oversold line
    
    plt.tight_layout()
    
    # Save plot with unique name for each cryptocurrency
    plot_filename = f'{coin.lower()}_elliot_waves.png'
    plt.savefig(plot_filename)
    plt.close()
    
    return waves, df, plot_filename

def interpret_elliot_waves(waves, df, coin):
    """
    Interpret Elliot Wave patterns and provide investment insights
    """
    # Analyze wave characteristics
    impulse_waves = waves['impulse_waves']
    corrective_waves = waves['corrective_waves']
    
    # Current price and trend analysis
    current_price = df['close'].iloc[-1]
    price_trend = 'bullish' if impulse_waves and impulse_waves[-1]['type'] == 'impulse' else 'bearish'
    
    # Wave pattern interpretation
    wave_interpretation = {
        'total_waves': len(waves['wave_details']),
        'impulse_waves_count': len(impulse_waves),
        'corrective_waves_count': len(corrective_waves),
        'dominant_trend': price_trend
    }
    
    # Investment recommendation logic
    def get_investment_recommendation():
        # Basic recommendation based on wave patterns
        if len(impulse_waves) > len(corrective_waves) and price_trend == 'bullish':
            return "STRONG BUY", "The current wave pattern suggests a strong upward momentum."
        elif len(impulse_waves) > len(corrective_waves):
            return "MODERATE BUY", "The wave pattern indicates potential growth."
        elif len(corrective_waves) > len(impulse_waves):
            return "HOLD/CAUTION", "The market shows more corrective patterns, suggesting potential volatility."
        else:
            return "NEUTRAL", "The wave patterns are relatively balanced."
    
    recommendation, reason = get_investment_recommendation()
    
    # Layman's explanation
    explanation = f"""
    Elliot Wave Analysis for {coin.capitalize()}:
    
    Wave Pattern Overview:
    - Total Waves Detected: {wave_interpretation['total_waves']}
    - Impulse Waves: {wave_interpretation['impulse_waves_count']}
    - Corrective Waves: {wave_interpretation['corrective_waves_count']}
    - Current Market Trend: {wave_interpretation['dominant_trend'].capitalize()}
    
    Current Price: ${current_price:.2f}
    
    Investment Recommendation: {recommendation}
    Reason: {reason}
    
    Elliot Wave Explanation:
    Imagine the market as a series of waves. Impulse waves (green) represent strong price movements 
    in the main trend direction, while corrective waves (red) are smaller movements against the trend. 
    More impulse waves suggest a stronger market direction.
    """
    
    return explanation

def main():
    # Provide a list of some popular cryptocurrencies
    popular_coins = [
        'bitcoin', 'ethereum', 'cardano', 'solana', 'ripple', 
        'dogecoin', 'polkadot', 'chainlink', 'vechain', 'stellar',
        'velodrome-finance'  # Updated to full coin name
    ]
    
    print("Popular Cryptocurrencies:")
    print(", ".join(coin.capitalize() for coin in set(popular_coins)))
    
    while True:
        try:
            # User input for cryptocurrency
            coin = input("\nEnter the cryptocurrency name (or 'quit' to exit): ").lower().strip()
            
            if coin == 'quit':
                break
            
            # Fetch candle data
            df = get_crypto_candles(coin)
            
            # Detect waves
            waves, price_df, plot_filename = detect_elliot_waves(df, coin)
            
            # Interpret waves and get recommendation
            analysis = interpret_elliot_waves(waves, price_df, coin)
            
            # Print analysis
            print(analysis)
            
            # Option to view wave visualization
            view_graph = input("\nView wave visualization? (yes/no): ").lower().strip()
            if view_graph == 'yes':
                print(f"\nWave visualization saved as {plot_filename}")
                print("To view the plot, use the command:")
                print(f"open {plot_filename}")
        
        except CryptoAnalysisError as e:
            print(f"\nError: {e}")
        except Exception as e:
            print(f"\nUnexpected error occurred: {e}")

if __name__ == "__main__":
    main()
