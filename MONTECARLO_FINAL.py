import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import plotly.graph_objects as go
import opstrat as op
from datetime import datetime, timedelta

st.title("MONTECARLO SIMULATIONS")
st.caption("Andrea Martínez | 0233220")


st.divider()
    
# Lista de nombres de las pestañas
tabs = ["Learn More", "Gross Profit", "Forecasting Stock Prices", "Option Prices"]

# Seleccionar la pestaña actual
selected_tab = st.radio("Go to:", tabs)

st.divider()

if selected_tab == "Learn More":
    
    st.markdown(":large_blue_circle:")
    st.header("Welcome to the Monte Carlo Simulations World! 🎲")

    # Introduction to Monte Carlo Simulations
    st.write("""
    Monte Carlo simulations are a powerful statistical technique used to model and analyze complex systems by using random sampling. 
    They were first developed by Stanislaw Ulam during the 1940s while working on nuclear weapons projects at the Los Alamos National Laboratory. The name "Monte Carlo" was inspired by the Monte Carlo Casino in Monaco, known for its games of chance and randomness.

    Monte Carlo simulations are widely used in various fields, including finance, engineering, physics, and many others. 
    Some common applications include:

    - **Financial Risk Analysis:** Monte Carlo simulations are used to model and analyze the uncertainty and risk associated with financial investments and portfolios.
    - **Engineering and Manufacturing:** They help simulate and optimize complex systems and processes, such as product design, manufacturing processes, and supply chain management.
    - **Healthcare and Pharmaceuticals:** Monte Carlo simulations are used in drug discovery, clinical trials, and epidemiological studies to model and analyze the effects of different treatments and interventions.
    - **Project Management:** They aid in project planning and scheduling by simulating various scenarios and estimating project timelines and costs.

    Explore the possibilities of Monte Carlo simulations with our interactive app!
    """)

    # URL directa a una imagen en Wikimedia Commons
    image_url = "https://www.researchgate.net/profile/Francesc-Font-Clos/publication/229066089/figure/fig5/AS:667776196825106@1536221660637/Polish-mathematician-Stanislaw-Ulam-together-with-the-current-version-of-the-first-page.ppm"

    # Mostrar la imagen en la aplicación Streamlit
    st.image(image_url, caption="Stanislaw Ulam, inventor of Monte Carlo simulations", use_column_width=False)
    
    # Footer
    st.write("""
    LEARN MORE HERE: [Hitting the Jackpot: The Birth of the Monte Carlo Method](https://discover.lanl.gov/publications/actinide-research-quarterly/first-quarter-2023/hitting-the-jackpot-the-birth-of-the-monte-carlo-method/)
    """)
    
    
# Contenido de la pestaña 2
elif selected_tab == "Gross Profit":
    
    st.header("Predicting Gross Profit :dollar:")
    st.markdown("We will generate 1000 Gross Profit simulations, based on the average income and % of costs of a company :office:")

    #Ingreso de datos
    rev_m = st.number_input("Enter the average income of your business:") 
    rev_stdev = st.number_input("How much do incomes usually vary?")  
    PorCogs = st.number_input("What is the percentage of the costs? (Enter the number in decimal)", 0.05, 0.99)  

    if st.button("Generate Gross Profit"):
        if rev_m is not None and rev_stdev is not None and PorCogs is not None:
            iterations = 1000 
            rev = np.random.normal(rev_m, rev_stdev, iterations)

            #Fórmula de Costos
            COGS = - (rev * np.random.normal(PorCogs, 0.1))
            meancogs = COGS.mean()
            stdcogs = COGS.std()
            st.markdown(f"In this case, the mean of the COGS is {meancogs: .2f}, and its standard deviation is {stdcogs: .2f}")

            st.markdown("**:bulb: Gross profit is the difference between revenue and COGS**")

            #Se usa + porque los cogs los tenemos en -
            Gross_Profit = rev + COGS

            #Gráfica Gross Profit
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(iterations)), y=Gross_Profit, mode='lines', line=dict(color="#FF007F"), name='Gross Profit'))

            fig.update_layout(title="Gross Profit",
                              xaxis_title="Iteración",
                              yaxis_title="Gross Profit")

            st.plotly_chart(fig)

            # Calculo de max, min, mean, std de Gross_Profit
            max_gp = Gross_Profit.max()
            min_gp = Gross_Profit.min()
            mean_gp = Gross_Profit.mean()
            std_gp = Gross_Profit.std()

            # Diccionario con los resultados y tabla
            resultados = {
                '-': ['Max', 'Min', 'Mean', 'Standard Deviation'],
                'Result': [max_gp, min_gp, mean_gp, std_gp]
            }

            df_resultados = pd.DataFrame(resultados)
            st.markdown("**Gross Profit Results:**")
            st.table(df_resultados)

            #Histograma de Gross Profit
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=Gross_Profit, nbinsx=25,
                                       marker_color="#DAFF7D",
                                       name='Gross Profit'))
            fig.update_layout(title="Gross Profit",
                              xaxis_title="Gross Profit",
                              yaxis_title="Frequency")
            st.plotly_chart(fig)

            st.markdown(f":moneybag: The maximum profits that can be expected are {max_gp:.3f}, and the minimum ones are {min_gp:.3f}")
            
# Contenido de la pestaña 3
elif selected_tab == "Forecasting Stock Prices":
   
    st.header("Forecasting Stock Prices :chart_with_upwards_trend:")

    # Load the data from CSV
    url = "https://research-watchlists.s3.amazonaws.com/df_UniversidadPanamericana_ohlc.csv"
    df = pd.read_csv(url)

    # Convert the 'time' column to datetime format
    df['time'] = pd.to_datetime(df['time'])

    # Extract unique values from the 'Symbol' column
    unique_symbols = df['Symbol'].unique()

    # Create a select box to choose a symbol
    selected_symbol = st.selectbox('Select a symbol:', unique_symbols)

    # Filter the data for the selected symbol
    symbol_data = df[df['Symbol'] == selected_symbol]

    st.markdown(f"Let's get the data for the stock: {selected_symbol}")

    selected_df = df[df['Symbol'] == selected_symbol].copy()
    last_date = selected_df['time'].iloc[-1]
    last_close_price = selected_df['close'].iloc[-1]
    last_log_return = np.log(last_close_price / selected_df['close'].iloc[-2])

    st.markdown(f"**Last Date:** {last_date}")
    st.markdown(f"**Last Close Price:** {last_close_price}")
    st.markdown(f"**Last Log Return:** {last_log_return: .8f}")

    # Gráfico de los precios ajustados
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=symbol_data['time'], y=symbol_data['close'], mode='lines', name='Close Price'))
    fig.update_layout(title=f"Adjusted Prices ({selected_symbol})",
                  xaxis_title="Date",
                  yaxis_title="Adjusted Close Price")
    st.plotly_chart(fig)

    # Calculate the logarithmic returns of close prices
    log_returns = np.log(symbol_data['close'] / symbol_data['close'].shift(1))

    # Plot the logarithmic returns using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=symbol_data['time'], y=log_returns, mode='lines', name='Logarithmic Returns', marker=dict(color='red')))
    fig.update_layout(title=f"Logarithmic Returns ({selected_symbol})",
                      xaxis_title="Date",
                      yaxis_title="Logarithmic Returns")
    st.plotly_chart(fig)

    #Drift
    st.markdown(":heavy_plus_sign: **Drift**: is an approximation of future stock return rates, calculated daily. Formula:")
    formula = r"Drift = u - \frac{1}{2} \cdot \text{var}"
    st.latex(formula)


    #Información de Log_Returns y tabla
    u = log_returns.mean()
    var = log_returns.var()
    stdev = log_returns.std()

    u_formatted = f"{u:.6f}"
    var_formatted = f"{var:.6f}"
    stdev_formatted = f"{stdev:.6f}"

    drift = u - (0.5*var)
    drift_formatted = f"{float(drift):.5f}"

    datas = {
        "Variable": ["u(mean)", "var", "std dev..", "drift"],
        "Value": [u_formatted, var_formatted, stdev_formatted, drift_formatted]
    }
    df_table = pd.DataFrame(datas)
    st.table(df_table)

    #Movimiento browniano Z
    st.divider()
    st.markdown("**To get the second element of Brownian Motion (z):**")
    #Pronóstico 1000 días
    st.markdown("Random probabilities are generated to get the value *z*, forecasting prices for 1000 days across 10 series of future price predictions:")

    t_intervals = 1000
    iterations = 10
    daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(t_intervals, iterations)))
    st.write(daily_returns)

    S0 = selected_df['close'].iloc[-1]
    price_list = np.zeros_like(daily_returns)
    price_list[0] = S0

    #Loop
    for t in range(1, t_intervals):
        price_list[t] = price_list[t - 1] * daily_returns[t]

    #Ultimo valor de la acción
    S0_formatted = f"{S0:.2f}"
    st.markdown(f":red_circle: Last stock value = **{S0_formatted}**")
    st.markdown("Price list:")

    st.write(price_list)

    # Gráfico de líneas con Plotly
    fig = go.Figure()

    colors = ['#FF007F', '#00BFFF', '#8A2BE2', '#FF6347', '#3CB371', '#FFD700', '#FF4500', '#DAA520', '#1E90FF', '#32CD32']

    for i in range(10):
        fig.add_trace(go.Scatter(
            x=list(range(len(price_list))),
            y=price_list[:, i],
            mode='lines',
            line=dict(color=colors[i % len(colors)]),
            name=f'Simulation {i + 1}'
        ))

    fig.update_layout(width=800, height=500)
    fig.update_layout(
        title="Price List Chart (Simulations)",
        xaxis_title="Days",
        yaxis_title="Price"
    )
    st.plotly_chart(fig)

    st.markdown(f":small_red_triangle: The chart shows 10 possible future price scenarios for the next 1000 days.")
    # Calcular la simulación con el mayor y menor precio final
    final_prices = price_list[-1]
    max_price_index = np.argmax(final_prices)
    min_price_index = np.argmin(final_prices)

    max_price = final_prices[max_price_index]
    min_price = final_prices[min_price_index]

    # Mostrar los resultados en markdown
    st.markdown(f" :relieved: **Simulation with the highest price:** Simulation {max_price_index + 1} with a price of {max_price:.2f}")
    st.markdown(f" :worried: **Simulation with the lowest price:** Simulation {min_price_index + 1} with a price of {min_price:.2f}")

# Contenido de la pestaña 4
elif selected_tab == "Option Prices":

    st.header("Options :arrow_heading_down:")
    st.subheader("Black-Scholes-Merton Method")
    st.markdown(":black_circle: The Black-Scholes-Merton formula helps us calculate the price of a European call/put option, which can only be exercised on the expiration date.")

    # Formulas
    st.latex(r"""
    d_1 = \frac{\ln\left(\frac{S_0}{K}\right) + \left(r + \frac{\sigma^2}{2}\right)T}{\sigma \cdot \sqrt{T}}
    """)

    st.latex(r"""
    d_2 = d_1 - \sigma \cdot \sqrt{T}
    """)

    # Definitions of variables
    variables = {
        'Variable': ['S', 'K', 'r', 'stdev', 'T'],
        'Description': [
            'Stock Price',
            'Strike Price',
            'Risk-Free Rate',
            'Standard Deviation',
            'Time to Expiry (in days)'
        ]
    }

    df_variables = pd.DataFrame(variables)
    st.table(df_variables)

    #Ingreso de ticker
    ticker_input = st.text_input("Enter the stock ticker:")

    if not ticker_input:
        st.warning("Please enter a stock ticker")
    else:
        tickers = [ticker_input]

            # Fetch historical data from Yahoo Finance
        if ticker_input:
            # Calculate the date 5 years ago from today
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

            # Fetch data from Yahoo Finance
            try:
                symbol_data = yf.download(ticker_input, start=start_date)

                # Create a candlestick chart
                fig = go.Figure(data=[go.Candlestick(x=symbol_data.index,
                                                     open=symbol_data['Open'],
                                                     high=symbol_data['High'],
                                                     low=symbol_data['Low'],
                                                     close=symbol_data['Adj Close'])])

                # Updating layout to make the chart more informative and appealing
                fig.update_layout(title=f'Candlestick Chart for {ticker_input}',
                                  xaxis_title='Date',
                                  yaxis_title='Price',
                                  xaxis_rangeslider_visible=False)  # Disable range slider for better clarity

                st.markdown(f"This is a candlestick for {ticker_input} with daily data for the last year, it should be useful to buy options and make strategies:")
                # Display the candlestick chart
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error fetching data: {e}")

        # Datos de Yahoo Finance
        data = pd.DataFrame()
        for t in tickers:
                mydata = yf.download(t, start="2014-01-01")[["Adj Close"]]
                data = pd.concat([data, mydata.rename(columns={"Adj Close": t})], axis=1)
                
        # Input variables
        S0 = data.iloc[-1, 0]
        S0_formateado = f"{S0:.2f}"  # Current stock price
        r = 0.025  # Risk-free rate

        T_days = st.number_input('Enter Time to Expiry (T) (in days)')  # Time to expiry (in days)
        T = T_days / 365.0  # Convert days to years

        # Input for strike price
        K = st.number_input('Enter Strike Price (K)')

         #Obtención de los log returns y su desv. est.
        log_returns = np.log(1 + data.pct_change())
        stdev = log_returns.std() * 250 ** 0.5

        if isinstance(stdev, pd.Series):
            stdev = stdev.iloc[0]

        # Function to calculate d1 and d2
        def d1(S, K, r, stdev, T):
            return (np.log(S / K) + (r + stdev ** 2) * T) / (stdev * np.sqrt(T))

        def d2(S, K, r, stdev, T):
            return (np.log(S / K) + (r - stdev ** 2) * T) / (stdev * np.sqrt(T))

        # Function to calculate Black-Scholes-Merton price
        def BSM_call(S, K, r, stdev, T):
            return (S * norm.cdf(d1(S, K, r, stdev, T))) - (K * np.exp(-r * T) * norm.cdf(d2(S, K, r, stdev, T)))

            # Function to calculate Black-Scholes-Merton put price
        def BSM_put(S, K, r, stdev, T):
            return (K * np.exp(-r * T) * norm.cdf(-d2(S, K, r, stdev, T))) - (S * norm.cdf(-d1(S, K, r, stdev, T)))

        # Calculate d1, d2, and Call Option price using Black-Scholes-Merton method
        d1_result = d1(S0, K, r, stdev, T)
        d2_result = d2(S0, K, r, stdev, T)
        BSM_call_result = BSM_call(S0, K, r, stdev, T)
        BSM_put_result = BSM_put(S0, K, r, stdev, T)

        # Display results
        st.markdown(f":red_circle: **Stock Price** de {tickers}: {S0_formateado}")
        st.markdown(f"**Standard Deviation** of log returns: {stdev:.3f}")
        st.markdown(f"**d1**: {d1_result:.3f}")
        st.markdown(f"**d2**: {d2_result:.3f}")
        st.markdown(f"### :dollar: **Black-Scholes-Merton CALL Option Price**: {BSM_call_result:.2f}")
        st.markdown(f"### :dollar: **Black-Scholes-Merton PUT Option Price**: {BSM_put_result:.2f}")

        # Explicación
        with st.expander("What am I getting and how will it serve me in the future?", expanded=True):
            st.write(f"""
            By calculating the current price of the underlying asset and the price of the call/put option, we are obtaining important information to make financial decisions.

            - **<span style="color:green">Current price of the asset:</span>** This is the current price of the underlying asset (**{ticker_input}**), which can fluctuate in the market. Knowing this price allows you to evaluate the current value of your investment and make informed decisions about buying, selling, or holding the asset.

            - **<span style="color:green">Price of the call option:</span>** This is the theoretical price that you would have to **pay right now to acquire a call/put option** on the underlying asset, with a specified strike price and expiration date. This price is calculated using the *Black-Scholes model*, which takes into account various market factors. Knowing the price of the call option allows you to evaluate the cost of acquiring this financial contract and decide if it is a suitable investment for your financial goals.

            In the future, this information will be useful for making strategic decisions in the options and stocks market. For example, you could compare the price of the call option with the current price of the asset to determine if the option is overvalued or undervalued relative to the underlying asset. This information could also be used to plan hedging or speculation strategies in the options market.

        """, unsafe_allow_html=True)


        st.divider()

        # Deshabilitar el aviso de uso global de Pyplot
        st.set_option('deprecation.showPyplotGlobalUse', False)

        st.header("CALL/PUT")

        #Short Put Option
        st.markdown("""
        ### Long Put Option

        A **put option** provides the holder with the right, but not the obligation, to sell an underlying asset at a predetermined price (the strike price) within a specified timeframe. When an investor purchases a put option, it is known as a **long put** position.

        - **Purpose**: The buyer of a put option (long put) anticipates that the price of the underlying asset will decline before the expiration date. By acquiring the put option, they secure the right to sell the asset at the strike price, thus profiting from a potential decrease in its market value.
        - **Risk**: The risk for the long put position is limited to the premium paid for the option. If the price of the underlying asset does not decrease below the strike price before expiration, the option may expire worthless, resulting in a loss limited to the premium paid.

    """)

        op.single_plotter(spot=S0, strike=K, op_type='p', tr_type='b', op_pr=BSM_put_result, spot_range=40)
        st.pyplot()

        #Long Call Option:
        st.markdown("""
        #### Long Call Option:

        A **call option** gives the holder the right, but not the obligation, to buy an asset at a specified price (the strike price) within a specified period of time. When an investor purchases a call option, it is referred to as a **long call** position.

        - **Purpose**: The buyer of a call option (long call) expects the price of the underlying asset to rise above the strike price before the expiration date. By purchasing the call option, they have the opportunity to buy the asset at a lower price (the strike price) and can profit from the potential price increase.
        - **Risk**: The risk for the long call position is limited to the premium paid for the option. If the price of the underlying asset does not increase above the strike price before expiration, the option may expire worthless, resulting in a loss limited to the premium paid.

        """)


        op.single_plotter(spot=S0, strike=K, op_type='c', tr_type='b', op_pr=BSM_call_result, spot_range=40)
        st.pyplot()

        st.divider()

        st.header("Make a Strategy")

        # Initialize selected strategy variable
        selected_strategy = st.session_state.get("selected_strategy", None)

        # Show the button if a strategy is not selected
        if selected_strategy is None:
            if st.button("Show Strategy Details"):
                selected_strategy = "Show"

        # If a strategy is selected, show the details
        if selected_strategy == "Show":
            # Selección de la estrategia
            strategy = st.selectbox("Select a strategy:", ["Short Strangle", "Long Strangle", "Long Straddle", "Iron Condor"])

            if strategy == "Short Strangle":
                # Short strangle
                st.markdown("""
                ### Short Strangle

                The **short strangle** strategy involves selling both a call and a put option with different strike prices but the same expiration date. It is used when the trader expects minimal price movement in the underlying asset.

                - **Objective**: Profit from the premiums collected from selling both options if the price remains within the strike prices.
                - **Risk**: Unlimited on the upside and substantial on the downside if the price moves significantly beyond the strike prices.

                """)

                op_1 = {'op_type':'c', 'strike': K+20, 'tr_type':'s', 'op_pr': BSM_call_result}
                op_2 = {'op_type':'p', 'strike': K-20, 'tr_type':'s', 'op_pr': BSM_put_result}
                op.multi_plotter(spot = S0, spot_range=70, op_list=[op_1,op_2])
                st.pyplot()

                st.divider()
                st.write("""
                :bulb: **Are you interested in learning more about Option Strategies?:** [CLICK HERE](https://www.investopedia.com/trading/options-strategies/)
    """)
               
            # Long strangle
            elif strategy == "Long Strangle":
                st.markdown("""
                ### Long Strangle

                The **long strangle** strategy involves buying both a call and a put option with different strike prices but the same expiration date. It is used when the trader expects significant price movement in the underlying asset but is uncertain of the direction.

                - **Objective**: Profit from a large move in the underlying asset's price, either upward or downward.
                - **Risk**: Limited to the total premiums paid for both options.

                """)

                op_1 = {'op_type': 'c', 'strike': K + 20, 'tr_type': 'b', 'op_pr': BSM_call_result}
                op_2 = {'op_type': 'p', 'strike': K - 20, 'tr_type': 'b', 'op_pr': BSM_put_result}
                op.multi_plotter(spot=S0, spot_range=70, op_list=[op_1, op_2])
                st.pyplot()
                
                st.divider()
                st.write("""
                :bulb: **Are you interested in learning more about Option Strategies?:** [CLICK HERE](https://www.investopedia.com/trading/options-strategies/)
    """)

            # Long straddle
            elif strategy == "Long Straddle":
                st.markdown("""
                ### Long Straddle

                The **long straddle** strategy involves buying both a call and a put option with the same strike price and expiration date. It is used when the trader expects significant volatility in the underlying asset's price but is uncertain about the direction of the move.

                - **Objective**: Profit from a large move in the underlying asset's price, either upward or downward.
                - **Risk**: Limited to the total premiums paid for both options.

                """)
                op_1 = {'op_type': 'c', 'strike': K, 'tr_type': 'b', 'op_pr': BSM_call_result}
                op_2 = {'op_type': 'p', 'strike': K, 'tr_type': 'b', 'op_pr': BSM_put_result}
                op.multi_plotter(spot=S0, spot_range=70, op_list=[op_1, op_2])
                st.pyplot()
                
                st.divider()
                st.write("""
                :bulb: **Are you interested in learning more about Option Strategies?:** [CLICK HERE](https://www.investopedia.com/trading/options-strategies/)
    """)

            # Iron Condor
            elif strategy == "Iron Condor":
                st.markdown("""
                ### Iron Condor

                The **iron condor** strategy involves selling a lower-strike put and a higher-strike call, while simultaneously buying a further out-of-the-money put and call. This strategy is used when the trader expects low volatility in the underlying asset's price.

                - **Objective**: Profit from the premiums collected from selling the options if the price remains within a certain range.
                - **Risk**: Limited to the difference between the strike prices of the bought and sold options minus the net premium received.

                """)
                op_1 = {'op_type': 'c', 'strike': K + 5, 'tr_type': 's', 'op_pr': BSM_call_result*1.5}
                op_2 = {'op_type': 'c', 'strike': K + 10, 'tr_type': 'b', 'op_pr': BSM_call_result*1.2}
                op_3 = {'op_type': 'p', 'strike': K - 5, 'tr_type': 's', 'op_pr': BSM_put_result*0.75}
                op_4 = {'op_type': 'p', 'strike': K - 10, 'tr_type': 'b', 'op_pr': BSM_put_result*0.9}

                op_list=[op_1, op_2, op_3, op_4]
                op.multi_plotter(spot=S0, spot_range=30, op_list=op_list)
                st.pyplot()
                
                st.divider()
                st.write("""
                :bulb: **Are you interested in learning more about Option Strategies?:** [CLICK HERE](https://www.investopedia.com/trading/options-strategies/)
    """)

        # Store the selected strategy in the session state
        st.session_state["selected_strategy"] = selected_strategy

    
    
    
    
