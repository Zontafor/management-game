#%%
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from scipy.stats import norm
import numpy as np
import pandas as pd
from pathlib import Path
import re

class GameDataProcessor:
    """Process and manage game data from macro-enabled Excel workbooks"""
    def __init__(self, data_path):
        self.data_path = Path(str(data_path).replace('.xlsx', '.xlsm'))
        self.quarterly_data = None
        self.sheet_data = {}
        self.key_terms = [
            'Period', 'Quarter', 'Revenue', 'Cost', 'Profit', 'Sales', 'Market',
            'Quality', 'Production', 'Inventory', 'Price', 'Volume', 'Capacity'
        ]
        
    def load_data(self):
        """Load and process game data from Excel file"""
        try:
            print("Loading game data from Excel...")
            # Read all sheets with specific engine for macro workbooks
            excel_file = pd.ExcelFile(self.data_path, engine='openpyxl')
            sheets = excel_file.sheet_names
            print(f"Successfully loaded {len(sheets)} sheets")
            
            # Process each sheet
            for sheet_name in sheets:
                print(f"Processing sheet: {sheet_name}")
                try:
                    # Read the raw sheet without headers first
                    df = pd.read_excel(
                        excel_file,
                        sheet_name=sheet_name,
                        header=None,
                        na_values=['', 'NA', 'N/A', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', 
                                 '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 
                                 'NaN', 'n/a', 'nan', 'null']
                    )
                    
                    # Find and extract data clusters
                    processed_df = self._process_sheet_data(df, sheet_name)
                    
                    if processed_df is not None and not processed_df.empty:
                        self.sheet_data[sheet_name] = processed_df
                    
                except Exception as e:
                    print(f"Error processing sheet {sheet_name}: {str(e)}")
                    continue
            
            # Process quarterly data
            self._process_quarterly_data()
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            
    def _process_sheet_data(self, df, sheet_name):
        """Process sheet data using random search for data clusters"""
        try:
            if df.empty:
                return None

            # Find data clusters
            data_regions = self._find_data_regions(df)
            
            if not data_regions:
                return None
                
            # Process each region and combine them
            processed_dfs = []
            for region in data_regions:
                start_row, end_row, start_col, end_col = region
                region_df = df.iloc[start_row:end_row+1, start_col:end_col+1].copy()
                
                # Process the region
                processed_region = self._process_data_region(region_df)
                if processed_region is not None and not processed_region.empty:
                    processed_dfs.append(processed_region)
            
            # Combine all processed regions
            if processed_dfs:
                final_df = pd.concat(processed_dfs, axis=1)
                final_df = final_df.loc[:,~final_df.columns.duplicated()]
                return final_df
                
            return None
            
        except Exception as e:
            print(f"Error processing sheet {sheet_name}: {str(e)}")
            return None
            
    def _find_data_regions(self, df):
        """Find regions containing data using random search"""
        regions = []
        rows, cols = df.shape
        checked_cells = set()
        
        # Number of random starting points to try
        n_attempts = min(100, rows * cols // 50)  # Increased number of attempts
        
        # First try to find regions starting with key terms
        for row in range(rows):
            for col in range(cols):
                if (row, col) in checked_cells:
                    continue
                    
                cell_value = str(df.iloc[row, col]).strip()
                if self._is_potential_header(cell_value):
                    region = self._expand_region(df, row, col)
                    if region:
                        regions.append(region)
                        # Add cells in this region to checked cells
                        start_row, end_row, start_col, end_col = region
                        for r in range(start_row, end_row+1):
                            for c in range(start_col, end_col+1):
                                checked_cells.add((r, c))
        
        # Then try random locations for any missed data
        for _ in range(n_attempts):
            row = np.random.randint(0, rows)
            col = np.random.randint(0, cols)
            
            if (row, col) in checked_cells:
                continue
                
            cell_value = str(df.iloc[row, col]).strip()
            if self._is_potential_header(cell_value) or self._is_numeric(cell_value):
                region = self._expand_region(df, row, col)
                if region:
                    regions.append(region)
                    
                    start_row, end_row, start_col, end_col = region
                    for r in range(start_row, end_row+1):
                        for c in range(start_col, end_col+1):
                            checked_cells.add((r, c))
        
        return self._merge_regions(regions)
        
    def _is_potential_header(self, value):
        """Check if a value could be a header"""
        if not isinstance(value, str):
            return False
            
        value = value.lower().strip()
        # Check against key terms
        if any(term.lower() in value for term in self.key_terms):
            return True
            
        # Check for common business metrics patterns
        patterns = [
            r'total|sum|average|avg|mean|median|min|max|std|var',
            r'rate|ratio|percentage|percent|%',
            r'growth|change|delta|diff',
            r'year|month|quarter|period|week|day',
            r'value|amount|number|count|quantity',
            r'price|cost|revenue|profit|loss|margin'
        ]
        
        return any(re.search(pattern, value) for pattern in patterns)
        
    def _is_numeric(self, value):
        """Check if a value is numeric"""
        try:
            float(str(value).replace(',', '').replace('$', '').replace('%', ''))
            return True
        except:
            return False
            
    def _expand_region(self, df, row, col):
        """Expand region around a starting point"""
        rows, cols = df.shape
        start_row = end_row = row
        start_col = end_col = col
        
        # Expand upward
        while start_row > 0 and self._is_valid_cell(df.iloc[start_row-1, col]):
            start_row -= 1
            
        # Expand downward
        while end_row < rows-1 and self._is_valid_cell(df.iloc[end_row+1, col]):
            end_row += 1
            
        # Expand left
        while start_col > 0 and self._is_valid_cell(df.iloc[row, start_col-1]):
            start_col -= 1
            
        # Expand right
        while end_col < cols-1 and self._is_valid_cell(df.iloc[row, end_col+1]):
            end_col += 1
            
        # Verify region size
        if end_row - start_row >= 2 and end_col - start_col >= 1:
            return (start_row, end_row, start_col, end_col)
        return None
        
    def _is_valid_cell(self, value):
        """Check if a cell contains valid data"""
        if pd.isna(value):
            return False
            
        str_value = str(value).strip()
        return bool(str_value) and (self._is_numeric(str_value) or self._is_potential_header(str_value))
        
    def _merge_regions(self, regions):
        """Merge overlapping regions"""
        if not regions:
            return []
            
        # Sort regions by start row
        regions = sorted(regions, key=lambda x: (x[0], x[2]))
        
        merged = [regions[0]]
        for current in regions[1:]:
            previous = merged[-1]
            
            # Check if regions overlap
            if (current[0] <= previous[1] + 1 and 
                current[2] <= previous[3] + 1):
                # Merge regions
                merged[-1] = (
                    min(previous[0], current[0]),
                    max(previous[1], current[1]),
                    min(previous[2], current[2]),
                    max(previous[3], current[3])
                )
            else:
                merged.append(current)
                
        return merged
class GameDataProcessor(GameDataProcessor):  
    def _process_data_region(self, df):
        """Process a data region to create a structured DataFrame"""
        try:
            # Find the header row
            header_row = self._find_header_row(df)
            if header_row is None:
                return None
                
            # Set the header and remove rows above it
            headers = df.iloc[header_row]
            df = df.iloc[header_row + 1:].copy()
            df.columns = headers
            
            # Clean up the data
            df = self._clean_data(df)
            
            return df
            
        except Exception as e:
            print(f"Error processing data region: {str(e)}")
            return None
            
    def _clean_data(self, df):
        """Clean the processed DataFrame"""
        try:
            # Remove rows where all values are missing
            df = df.dropna(how='all')
            
            # Clean up column names
            df.columns = [str(col).strip() for col in df.columns]
            
            # Handle string/object columns first
            for col in df.select_dtypes(include=['object']).columns:
                # Forward fill string values
                df[col] = df[col].copy()  # Make a copy to avoid SettingWithCopyWarning
                mask = df[col].notna()
                df.loc[mask, col] = df.loc[mask, col].astype(str).replace('', np.nan)
                df[col] = df[col].fillna(method='ffill')
            
            # Handle numeric columns
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0)
            
            # Try to convert string columns that might be numeric
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    # Remove currency and thousand separators
                    temp_series = df[col].str.replace('$', '', regex=False) \
                                      .str.replace(',', '', regex=False) \
                                      .str.replace('%', '', regex=False)
                    numeric_series = pd.to_numeric(temp_series, errors='coerce')
                    if numeric_series.notna().any():
                        df[col] = numeric_series.fillna(0)
                except:
                    continue
            
            return df
            
        except Exception as e:
            print(f"Error in _clean_data: {str(e)}")
            return df
            
    def _find_header_row(self, df):
        """Find the most likely header row in a data region"""
        max_score = 0
        header_row = None
        
        for idx in range(len(df)):
            row = df.iloc[idx]
            score = sum(1 for val in row if self._is_potential_header(str(val)))
            if score > max_score:
                max_score = score
                header_row = idx
                
        return header_row
        
    def _process_quarterly_data(self):
        """Process quarterly data for analysis"""
        try:
            processed_data = []
            
            # First try Market Report
            if 'Market Report' in self.sheet_data:
                market_data = self._process_market_report()
                if market_data is not None:
                    processed_data.append(market_data)
            
            # Then try PROD-RPT and FIN-RPT
            if 'PROD-RPT' in self.sheet_data and 'FIN-RPT' in self.sheet_data:
                prod_fin_data = self._process_production_finance()
                if prod_fin_data is not None:
                    processed_data.append(prod_fin_data)
            
            # Finally try P sheets
            p_data = self._process_p_sheets()
            if p_data is not None:
                processed_data.append(p_data)
            
            # Combine all processed data
            if processed_data:
                self.quarterly_data = pd.concat([df for df in processed_data if df is not None], axis=1)
                self.quarterly_data = self.quarterly_data.loc[:,~self.quarterly_data.columns.duplicated()]
                self._add_derived_metrics()
                print("\nSuccessfully processed quarterly data")
                print(f"Final columns: {self.quarterly_data.columns.tolist()}")
            else:
                print("\nWarning: No quarterly data could be processed")
                
        except Exception as e:
            print(f"Error processing quarterly data: {str(e)}")
            
    def _process_market_report(self):
        """Process Market Report sheet"""
        try:
            if 'Market Report' in self.sheet_data:
                df = self.sheet_data['Market Report'].copy()
                return df
        except Exception as e:
            print(f"Error processing Market Report: {str(e)}")
        return None
        
    def _process_production_finance(self):
        """Process Production and Finance reports"""
        try:
            prod_df = self.sheet_data.get('PROD-RPT')
            fin_df = self.sheet_data.get('FIN-RPT')
            
            if prod_df is not None and fin_df is not None:
                # Merge on Period/Quarter if available
                if 'Period' in prod_df.columns and 'Period' in fin_df.columns:
                    return pd.merge(prod_df, fin_df, on='Period', how='outer')
            
        except Exception as e:
            print(f"Error processing Production/Finance reports: {str(e)}")
        return None
        
    def _process_p_sheets(self):
        """Process P sheets (periodic data)"""
        try:
            p_sheets = [sheet for sheet in self.sheet_data.keys() 
                       if sheet.startswith('P') and sheet[1:].replace('-', '').isdigit()]
            
            if p_sheets:
                dfs = []
                for sheet in sorted(p_sheets):
                    df = self.sheet_data[sheet].copy()
                    period = int(sheet[1:])
                    if 'Period' not in df.columns:
                        df['Period'] = period
                    dfs.append(df)
                
                return pd.concat(dfs, ignore_index=True)
                
        except Exception as e:
            print(f"Error processing P sheets: {str(e)}")
        return None
            
    def _add_derived_metrics(self):
        """Add derived metrics to quarterly data"""
        try:
            if self.quarterly_data is not None:
                # Profit calculation
                if 'Revenue' in self.quarterly_data.columns and 'Cost' in self.quarterly_data.columns:
                    self.quarterly_data['Profit'] = self.quarterly_data['Revenue'] - self.quarterly_data['Cost']
                
                # Market share calculation
                if 'Sales' in self.quarterly_data.columns and 'Total Market' in self.quarterly_data.columns:
                    self.quarterly_data['Market Share'] = self.quarterly_data['Sales'] / self.quarterly_data['Total Market']
                
                # Quality metric (if not already present)
                if 'Quality Rating' in self.quarterly_data.columns and 'Quality' not in self.quarterly_data.columns:
                    self.quarterly_data['Quality'] = self.quarterly_data['Quality Rating']
                    
        except Exception as e:
            print(f"Error adding derived metrics: {str(e)}")
            
    def get_latest_period_data(self):
        """Get data for the most recent period"""
        if self.quarterly_data is not None:
            try:
                if 'Period' in self.quarterly_data.columns:
                    latest_period = self.quarterly_data['Period'].max()
                    return self.quarterly_data[self.quarterly_data['Period'] == latest_period].iloc[0].to_dict()
                return self.quarterly_data.iloc[-1].to_dict()
            except Exception as e:
                print(f"Error getting latest period data: {str(e)}")
        return None
class ContractBiddingSystem:
    """Contract bidding optimization system"""
    def __init__(self, historical_data):
        self.historical_data = historical_data
        self.bid_models = {}
        self.win_probability_model = None
        self.min_margin = 0.01  # 1% minimum margin requirement
        self.base_cost = 50  # Base cost per unit
        
    def optimize_bid(self, contract_details, firm_capabilities):
        """Optimize bid for maximum expected value"""
        if not self._validate_contract(contract_details, firm_capabilities):
            return None
            
        variable_cost = self._calculate_variable_cost(
            contract_details['volume'],
            firm_capabilities['quality']
        )
        
        # Generate bid range
        min_bid = variable_cost * (1 + self.min_margin)
        max_bid = contract_details['current_bid']
        bid_range = np.linspace(min_bid, max_bid, 100)
        
        # Find optimal bid
        optimal_bid = None
        max_expected_value = float('-inf')
        
        for bid in bid_range:
            win_prob = self._calculate_win_probability(bid, contract_details)
            profit = (bid - variable_cost) * contract_details['volume']
            expected_value = win_prob * profit
            
            if expected_value > max_expected_value:
                max_expected_value = expected_value
                optimal_bid = bid
        
        return {
            'optimal_bid': optimal_bid,
            'win_probability': self._calculate_win_probability(optimal_bid, contract_details),
            'expected_value': max_expected_value,
            'risk_assessment': self._assess_bid_risk(optimal_bid, contract_details, variable_cost)
        }
    
    def _calculate_variable_cost(self, volume, quality):
        """Calculate variable cost based on volume and quality requirements"""
        # Base cost adjusted for quality
        quality_factor = quality ** 1.5  # Higher quality increases cost exponentially
        
        # Volume discount factor (economies of scale)
        volume_discount = max(0.8, 1 - np.log10(volume) * 0.05)  # Max 20% volume discount
        
        # Calculate final variable cost per unit
        variable_cost = self.base_cost * quality_factor * volume_discount
        
        return variable_cost
    
    def _validate_contract(self, contract, capabilities):
        """Validate contract against firm capabilities"""
        if contract['quality'] > capabilities['quality']:
            return False
        if contract['volume'] > capabilities['capacity']:
            return False
        return True
    
    def _calculate_win_probability(self, bid, contract):
        """Calculate probability of winning with bid"""
        current_bid = contract['current_bid']
        if bid >= current_bid:
            return 0.0
        
        # Calculate relative bid position
        bid_ratio = bid / current_bid
        
        # Base probability calculation
        base_prob = 1 - bid_ratio**2  # Quadratic scaling
        
        # Adjust for competition
        num_bidders = contract.get('num_bidders', 3)
        competition_factor = 1 - (num_bidders - 1) * 0.1
        
        win_prob = base_prob * competition_factor
        
        # Adjust for minimum bid reduction
        if (current_bid - bid) / current_bid < 0.01:
            win_prob *= 0.95  # Penalty for minimal reduction
        
        return max(0.0, min(1.0, win_prob))
    
    def _assess_bid_risk(self, bid, contract, variable_cost):
        """Assess risk level of bid"""
        margin = (bid - variable_cost) / bid
        volume_risk = contract['volume'] / 10000  # Normalized to typical volume
        competition_risk = contract.get('num_bidders', 3) / 5  # Normalized to max bidders
        
        risk_score = (
            0.4 * (1 - margin) +  # Lower margin = higher risk
            0.3 * volume_risk +    # Higher volume = higher risk
            0.3 * competition_risk # More bidders = higher risk
        )
        
        return {
            'risk_score': risk_score,
            'margin_risk': 1 - margin,
            'volume_risk': volume_risk,
            'competition_risk': competition_risk
        }

class PredictionSystem:
    """Future prediction system using historical data"""
    def __init__(self, historical_data):
        self.historical_data = historical_data
        self.models = {}
        self.metrics = ['quality', 'revenue', 'profit', 'market_share']
        
    def train_models(self):
        """Train prediction models for each metric"""
        print("\nTraining prediction models...")
        for metric in self.metrics:
            metric_cols = [col for col in self.historical_data.columns if metric.lower() in col.lower()]
            if metric_cols:
                # Use the first matching column
                self._train_metric_model(metric_cols[0])
            else:
                print(f"Warning: No matching columns found for metric '{metric}'")
            
    def predict_next_period(self, current_state):
        """Predict metrics for next period"""
        predictions = {}
        for metric in self.metrics:
            if metric in self.models:
                pred = self._predict_metric(metric, current_state)
                predictions[metric] = {
                    'value': pred,
                    'confidence': self._calculate_prediction_confidence(metric, pred)
                }
        return predictions
    
    def _train_metric_model(self, metric):
        """Train model for specific metric"""
        try:
            # Prepare features and target
            features = self._prepare_features(metric)
            target = self._prepare_target(metric)
            
            if features is not None and target is not None and len(features) > 0:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    features, target, test_size=0.2, random_state=42
                )
                
                # Train model
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42
                )
                model.fit(X_train, y_train)
                
                # Validate model
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                
                if test_score > 0.5:  # Only keep models with decent performance
                    self.models[metric] = {
                        'model': model,
                        'train_score': train_score,
                        'test_score': test_score,
                        'feature_names': features.columns.tolist()
                    }
                    print(f"Successfully trained model for {metric} (test score: {test_score:.2f})")
                else:
                    print(f"Warning: Poor model performance for {metric} (test score: {test_score:.2f})")
                    
        except Exception as e:
            print(f"Error training model for {metric}: {str(e)}")
    
    def _prepare_features(self, metric):
        """Prepare features for prediction"""
        try:
            df = self.historical_data.copy()
            
            # Create lag features
            for i in range(1, 4):  # Use last 3 periods
                df[f'{metric}_lag{i}'] = df[metric].shift(i)
            
            # Add trend features
            df['trend'] = np.arange(len(df))
            if 'Period' in df.columns:
                df['quarter'] = df['Period'] % 4 + 1
            
            # Remove rows with NaN
            df = df.dropna()
            
            feature_cols = [col for col in df.columns if col != metric]
            return df[feature_cols]
            
        except Exception as e:
            print(f"Error preparing features for {metric}: {str(e)}")
            return None
            
    def _prepare_target(self, metric):
        """Prepare target variable for prediction"""
        try:
            df = self.historical_data.copy()
            df = df.dropna(subset=[metric])
            return df[metric]
        except Exception as e:
            print(f"Error preparing target for {metric}: {str(e)}")
            return None
    
    def _predict_metric(self, metric, current_state):
        """Predict specific metric"""
        try:
            if metric not in self.models:
                return None
                
            model_info = self.models[metric]
            model = model_info['model']
            feature_names = model_info['feature_names']
            
            # Prepare prediction features
            features = self._prepare_prediction_features(metric, current_state, feature_names)
            
            if features is not None:
                # Make prediction
                prediction = model.predict(features.reshape(1, -1))[0]
                return prediction
            
        except Exception as e:
            print(f"Error predicting {metric}: {str(e)}")
        return None
            
    def _prepare_prediction_features(self, metric, current_state, feature_names):
        """Prepare features for making predictions"""
        try:
            features = []
            
            for feature in feature_names:
                if feature in current_state:
                    features.append(current_state[feature])
                elif feature == 'trend':
                    features.append(len(self.historical_data))
                elif feature == 'quarter':
                    period = current_state.get('Period', 0)
                    features.append(period % 4 + 1)
                elif feature.endswith('_lag1') and metric in current_state:
                    features.append(current_state[metric])
                elif feature.endswith('_lag2') and f'{metric}_lag1' in current_state:
                    features.append(current_state[f'{metric}_lag1'])
                elif feature.endswith('_lag3') and f'{metric}_lag2' in current_state:
                    features.append(current_state[f'{metric}_lag2'])
                else:
                    features.append(0)  # Default value for missing features
                    
            return np.array(features)
            
        except Exception as e:
            print(f"Error preparing prediction features for {metric}: {str(e)}")
            return None
    
    def _calculate_prediction_confidence(self, metric, prediction):
        """Calculate confidence level for prediction"""
        try:
            if prediction is None:
                return 0.0
                
            model_info = self.models[metric]
            test_score = model_info['test_score']
            
            # Base confidence on model performance
            base_confidence = test_score
            
            # Adjust for prediction magnitude
            historical_std = self.historical_data[metric].std()
            historical_mean = self.historical_data[metric].mean()
            
            if historical_std > 0:
                z_score = abs(prediction - historical_mean) / historical_std
                magnitude_factor = 1 / (1 + z_score)  # Decrease confidence for outlier predictions
            else:
                magnitude_factor = 1
                
            confidence = base_confidence * magnitude_factor
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            print(f"Error calculating confidence for {metric}: {str(e)}")
            return 0.0
if __name__ == "__main__":
    # Initialize processor and load data
    processor = GameDataProcessor("game_data/w01_MBA_c5_out.xlsm")
    processor.load_data()
    
    if processor.quarterly_data is not None:
        print("\nQuarterly data columns:", processor.quarterly_data.columns.tolist())
        
        # Initialize prediction system
        predictor = PredictionSystem(processor.quarterly_data)
        predictor.train_models()
        
        # Initialize contract bidding system
        bidding_system = ContractBiddingSystem(processor.quarterly_data)
        
        # Example contract
        contract = {
            'volume': 10000,
            'quality': 1.0,
            'current_bid': 100,
            'num_bidders': 3
        }
        
        # Example firm capabilities
        capabilities = {
            'quality': 1.2,
            'capacity': 50000,
            'efficiency': 0.9
        }
        
        # Get optimal bid
        bid_recommendation = bidding_system.optimize_bid(contract, capabilities)
        
        # Get predictions for next period
        current_state = processor.get_latest_period_data()
        if current_state:
            predictions = predictor.predict_next_period(current_state)
            
            # Print results
            print("\nContract Bid Optimization:")
            if bid_recommendation:
                print(f"Optimal Bid: ${bid_recommendation['optimal_bid']:.2f}")
                print(f"Win Probability: {bid_recommendation['win_probability']:.1%}")
                print(f"Expected Value: ${bid_recommendation['expected_value']:,.2f}")
                print("\nRisk Assessment:")
                for risk_type, risk_value in bid_recommendation['risk_assessment'].items():
                    print(f"  {risk_type}: {risk_value:.2%}")
            
            print("\nPredictions for Next Period:")
            for metric, pred in predictions.items():
                if pred['value'] is not None:
                    print(f"{metric}:")
                    print(f"  Predicted Value: {pred['value']:.2f}")
                    print(f"  Confidence: {pred['confidence']:.1%}")
                else:
                    print(f"{metric}: Unable to make prediction")
        else:
            print("\nWarning: Could not get current state data for predictions")
    else:
        print("\nError: No quarterly data available for analysis")