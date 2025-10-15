# Arbitrage_Intraday_Imbalance

This GitHub repository presents my solution to the Case Study proposed by ENGIE for a **_Quant / Trading Assistant VIE position in Romania_**, which involved designing an algorithmic strategy to exploit **arbitrage opportunities** between the **Intraday market** and the **Imbalance mechanism** for the period from July to December 2024.

## Project Objective

The goal of this project is to simulate an **algorithmic imbalance optimization strategy** on the electricity market under realistic conditions.

Every 15 minutes, a trader can voluntarily create a **positive or negative imbalance** between the portfolio’s production and consumption. This imbalance is then settled by the grid operator at the **imbalance price**. By carefully choosing the imbalance, the trader can **profit from arbitrage** between this imbalance price and market prices (especially the Intraday price).

Three main cases can occur:

1. **Create a positive imbalance (voluntary overproduction):**  
   This means injecting more power than required into the grid. If the imbalance price is **higher** than the Intraday selling price, this is profitable.  
   ➝ **Sell through the imbalance**.

2. **Create a negative imbalance (voluntary underproduction):**  
   This means injecting less than expected. If the imbalance price is **lower** than the Intraday purchase price, this is also profitable.  
   ➝ **Buy through the imbalance**.

3. **Do nothing (stay balanced):**  
   If arbitrage is unfavorable (low or negative spread), the best decision is **not to create an imbalance** and simply close the position on the Intraday market.

The objective of this project is to **model, at each timestep, the optimal action** (sell, buy, or stay neutral) based on available data:  
load forecasts (`load_fcst`), production forecasts (`solar_fcst`, `wind_fcst`), actual production (`load_real`, `solar_real`, `wind_real`, `nuclear_real`, `fossil_gas_real`), price history (`ID_QH_VWAP`, `imb_price_pos`, `imb_price_neg`), reserve levels (`afrr_up`, `afrr_down`, `mfrr_up`, `mfrr_down`), and previous imbalances (`imb_volume`, `imbalance_status`).

This is fundamentally a **decision-making problem** rather than a pure regression one: the goal is not just to predict a price, but to **choose the action that maximizes profit** under real market rules.

## Project Steps

### 1. Pipeline Construction

This project was designed to realistically replicate the reasoning of an Intraday trading desk, with a strong focus on business logic consistency at each stage.

#### 1.a: Data Preparation and Feature Engineering

Extensive **feature engineering** was implemented to fully exploit the available information:

- **Forecast errors:** difference between forecasts and actuals (`load_err`, `solar_err`, `wind_err`),
- **Reserve indicators:** aggregation of available reserve capacity (`afrr_cover_ratio`, `mfrr_cover_ratio`),
- **Grid imbalance state:** synthetic indicator (`imbalance_status`) showing whether imbalances were covered by reserves,
- **Arbitrage spreads:** calculated opportunities between market prices (`spread_long`, `spread_short`),
- **Market behavior history:** weighted historical spread variable (`historical_spread`),
- **Temporal encoding:** features like hour, weekday, month, etc.,
- **Lagging:** time-lagged variables to capture dynamics (`*_lagged_4/5/6`).

All these variables were carefully selected to reflect the signals a trader would have in real time.

#### 1.b: Normalization

Variables were standardized with `StandardScaler`, a crucial step to stabilize the training of the deep learning model.

#### 1.c: Temporal Data Splitting

Data was split into a **training set** (until end of 2024) and a **test set** (from January 2025 onward), ensuring strict chronological separation.  
During training, a `TimeSeriesSplit` (5 folds) was used to respect the time order (no future leakage).

#### 1.d: Missing Data Handling

Some key columns contained missing values and were treated carefully:

- `load_fcst` (load forecast): imputed with **bidirectional SARIMA** to capture strong daily seasonality,
- `solar_fcst`, `wind_fcst`, `solar_real`, `wind_real` (production): interpolated with `interpolate(method='time')` due to their continuous nature,
- Remaining missing values were filled using **forward fill** (`ffill`) to prevent future data leakage.

These choices ensure that each row used for modeling accurately reflects the information available at time *t*.

### 2. Modeling and Training

#### 2.a: Problem Formulation

From a decision-support perspective, the goal is to predict the **volume to commit** based on the arbitrage signal in imbalance prices.  
The target variable `target_volume` is defined as:

- `+10` MW if the long spread (`spread_long`) is positive, justifying a **sell** at the imbalance price,
- `-10` MW if the short spread (`spread_short`) is positive, justifying a **buy** at the imbalance price,
- `0` otherwise.

The model must learn to **estimate the optimal volume to commit**, ranging from -10 to +10 MW, based on available signals.

#### 2.b: Model Architecture

The main model is a **fully connected neural network** implemented in PyTorch:

- Input: normalized explanatory variables (see section 1.a),
- 2 hidden layers with `ReLU` (128 then 32 neurons),
- Output layer with `Tanh`, scaled by 10 to output between -10 and 10,
- Optimizer: `Adam`,
- Loss function: `MSELoss` (Mean Squared Error), as the problem is formulated as a **continuous regression**.

A custom **asymmetric loss** penalizing incorrect directional signals was tested but reduced overall accuracy.  
The original model was therefore retained.

#### 2.c: Training

Training was performed with `TimeSeriesSplit` (5 folds) on each temporal segment, using:

- 20 training epochs,
- Batch size: 64,
- Training on past data only, validating on future data (business logic respected).

### 3. Performance Evaluation

Model evaluation focused on **decision quality** and its **simulated financial impact**.

#### 3.a: Realized PnL vs Optimal PnL

For each prediction, **realized PnL** is computed as:

- If the model **sells** (`prediction > 0`), profit = `spread_long × prediction`,
- If the model **buys** (`prediction < 0`), profit = `-spread_short × prediction`,
- Otherwise, profit = 0.

**Optimal PnL** is defined as the maximum possible profit if the ideal decision had been taken (`target_volume` on the correct spread).

The ratio `sum(realized PnL) / sum(optimal PnL)` gives the model’s **capture performance**, which reached about **62.55%**, a promising result.

#### 3.b: Qualitative Decision Analysis

Each prediction is categorized into one of five types to assess decision relevance in a trading context:

| Decision Type              | Description                                                             | Cases  | Percentage |
|---------------------------|-------------------------------------------------------------------------|--------|------------|
| Good prediction          | Correct position taken, aligned with spread direction                   | 6163   | 70.09 %    |
| Missed opportunity       | No action taken despite an exploitable spread                           | 378    | 4.30 %     |
| Unnecessary position     | Position taken without a significant spread                             | 294    | 3.34 %     |
| Wrong-way position       | Position taken in the **opposite direction** of the observed spread     | 1941   | 22.07 %    |
| Neutral                  | No action taken and no exploitable spread                               | 17     | 0.19 %     |

This classification provides an **intuitive view of the model’s errors** and highlights critical cases like **wrong-way trades**, which are particularly costly.
