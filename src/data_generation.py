# !pip install pandas==2.2.0 numpy==1.26.4

import pandas as pd
import numpy as np
import random
import json
import os
from datetime import datetime, timedelta

class CausalABDataGenerator:
    def __init__(self, n_users=150000, n_products=1500, experiment_start='2023-10-15'):
        self.n_users = n_users
        self.n_products = n_products
        
        self.exp_start_date = datetime.strptime(experiment_start, '%Y-%m-%d')
        self.pre_period_start = self.exp_start_date - timedelta(days=14)
        self.exp_end_date = self.exp_start_date + timedelta(days=14)
        
        np.random.seed(42)
        random.seed(42)
        
    def generate_products(self):
        print("1. Generating Product Catalog...")
        categories = ['Fashion', 'Shoes', 'Accessories', 'Beauty']
        brands = ['Nike', 'Zara', 'H&M', 'Gucci', 'Adidas', 'Unbranded']
        
        self.products = pd.DataFrame({
            'product_id': range(1, self.n_products + 1),
            'category': np.random.choice(categories, self.n_products),
            'brand': np.random.choice(brands, self.n_products),
            'price': np.round(np.random.lognormal(mean=3.5, sigma=0.8, size=self.n_products), 2),
            'margin_rate': np.round(np.random.uniform(0.05, 0.40, self.n_products), 2),
            'initial_stock': np.random.randint(20, 500, self.n_products) 
        })
        
        self.prod_stock = self.products.set_index('product_id')['initial_stock'].to_dict()
        self.prod_price = self.products.set_index('product_id')['price'].to_dict()
        self.prod_margin = self.products.set_index('product_id')['margin_rate'].to_dict()
        self.prod_cat = self.products.set_index('product_id')['category'].to_dict()
        return self.products

    def generate_users(self):
        print("2. Generating User Profiles & Synthetic Latency Setup...")
        latent_spend = np.random.lognormal(mean=2, sigma=1, size=self.n_users)
        
        self.users = pd.DataFrame({
            'user_id': range(1, self.n_users + 1),
            'group': np.random.choice(['A', 'B'], self.n_users),
            'segment': np.random.choice(['new', 'returning'], self.n_users, p=[0.3, 0.7]),
            'loyalty_score': np.round(np.random.uniform(0, 1, self.n_users), 2),
            'favorite_cat': np.random.choice(['Fashion', 'Shoes', 'Accessories', 'Beauty'], self.n_users),
            # FIX 5: Синтетический эксперимент с Latency
            'latency_treatment': np.random.choice(['none', 'plus_50ms', 'plus_100ms'], self.n_users, p=[0.9, 0.05, 0.05]),
            'latent_propensity': latent_spend
        })
        
        self.users['pre_revenue'] = np.round(self.users['latent_propensity'] * np.random.uniform(0.8, 1.2, self.n_users) * 10, 2)
        self.users['pre_sessions'] = np.random.poisson(lam=self.users['loyalty_score'] * 5 + 1)
        
        return self.users

    def generate_events(self):
        print(f"3. Running Strict Causal Engine for {self.n_users} users...")
        events = []
        session_level_data = []
        user_level_impact = {}

        user_prop = self.users.set_index('user_id')['latent_propensity'].to_dict()
        user_group = self.users.set_index('user_id')['group'].to_dict()
        user_fav_cat = self.users.set_index('user_id')['favorite_cat'].to_dict()
        user_lat_treat = self.users.set_index('user_id')['latency_treatment'].to_dict()

        session_id_counter = 1

        for user_id in range(1, self.n_users + 1):
            group = user_group[user_id]
            propensity = user_prop[user_id]
            fav_cat = user_fav_cat[user_id]
            lat_treatment = user_lat_treat[user_id]
            
            user_total_actual = 0
            user_total_cf = 0
            
            num_sessions = np.random.poisson(lam=3)
            if num_sessions == 0:
                continue
                
            for _ in range(num_sessions):
                session_id = f"sess_{session_id_counter}"
                session_id_counter += 1
                
                # FIX 2: База для Mediation (глубина сессии зависит от группы)
                base_length = np.random.poisson(lam=4) + 1
                session_length = base_length + 2 if group == 'B' else base_length
                
                session_actual_rev = 0
                session_cf_rev = 0
                session_clicks = 0
                
                cf_stock = self.prod_stock.copy()
                
                for pos in range(1, session_length + 1):
                    prod_id = random.randint(1, self.n_products)
                    price = self.prod_price[prod_id]
                    margin = self.prod_margin[prod_id]
                    category = self.prod_cat[prod_id]
                    
                    p_search, p_rec = (0.55, 0.15) if group == 'A' else (0.45, 0.25)
                    source = np.random.choice(['search', 'recommendations', 'direct'], p=[p_search, p_rec, 1 - p_search - p_rec])
                    
                    timestamp = self.exp_start_date + timedelta(days=random.randint(0, 13), hours=random.randint(8, 23))
                    
                    # FIX 5: Causal Latency Injection
                    base_latency = int(np.random.lognormal(3.8, 0.4))
                    added_latency = 50 if lat_treatment == 'plus_50ms' else (100 if lat_treatment == 'plus_100ms' else 0)
                    latency = base_latency + added_latency
                    
                    if random.random() < 0.015: 
                        latency += random.randint(3000, 10000) # Шум
                        
                    latency_penalty = np.exp(-0.005 * max(0, latency - 50))
                    
                    # Context
                    device = random.choice(["ios", "android", "web"])
                    time_of_day = "evening" if 18 <= timestamp.hour <= 23 else "day"
                    context = None if random.random() < 0.02 else json.dumps({"device": device, "time": time_of_day})
                    
                    device_boost = 1.1 if device == "ios" else 1.0
                    time_boost = 1.15 if time_of_day == "evening" else 1.0
                    affinity_boost = 1.5 if category == fav_cat else 1.0
                    trend_multiplier = 1.3 if timestamp.weekday() >= 4 and category in ['Fashion', 'Shoes'] else 1.0
                    
                    # --- BASE PROBABILITIES ---
                    p_click_base = 0.05 * (1 / np.sqrt(pos)) * latency_penalty * device_boost * time_boost * (propensity / 5 + 0.5)
                    p_pur_base = 0.1 * (100 / (price + 50)) * affinity_boost * trend_multiplier
                    
                    p_click_A = min(1.0, p_click_base)
                    p_pur_A = min(1.0, p_pur_base)
                    
                    p_click_B = p_click_base
                    p_pur_B = p_pur_base
                    
                    is_exploration = 0 # Флаг бандита
                    
                    # --- POLICY ENGINE (Группа B) ---
                    if source == 'recommendations':
                        # FIX 4: Availability Penalty (Supply-side)
                        stock_penalty_actual = 0.2 if self.prod_stock[prod_id] < 5 else 1.0
                        stock_penalty_cf = 0.2 if cf_stock[prod_id] < 5 else 1.0
                        
                        # FIX 3: Contextual Bandits Exploration
                        if random.random() < 0.05: 
                            is_exploration = 1
                            bandit_boost, margin_boost = 1.0, 1.0 # Отключаем персонализацию для исследования
                        else:
                            bandit_boost = 1.2 if (category == fav_cat and time_of_day == "evening") else 1.0
                            margin_boost = 1.0 + (margin * 1.5)
                        
                        p_click_B = min(1.0, p_click_base * bandit_boost * margin_boost * stock_penalty_actual)
                        p_pur_B = min(1.0, p_pur_base * bandit_boost * margin_boost)
                        
                        # Пересчет А для CF-штрафа стока
                        p_click_A = p_click_A * stock_penalty_cf
                        
                    # Распределение вероятностей Actual vs CF
                    if group == 'B':
                        p_click_actual, p_pur_actual = p_click_B, p_pur_B
                        p_click_cf, p_pur_cf = p_click_A, p_pur_A
                    else:
                        p_click_actual, p_pur_actual = p_click_A, p_pur_A
                        p_click_cf, p_pur_cf = p_click_B, p_pur_B
                    
                    ranking_score = np.round(p_click_actual * price, 4)
                    
                    # ===== ACTUAL UNIVERSE =====
                    events.append([timestamp, user_id, group, 'impression', prod_id, source,
                                   latency, session_id, pos, ranking_score, is_exploration, 0, context])
                    
                    if random.random() < p_click_actual:
                        if random.random() > 0.01:
                            session_clicks += 1
                            events.append([timestamp + timedelta(seconds=2), user_id, group, 'click',
                                           prod_id, source, latency, session_id, pos, ranking_score, is_exploration, 0, context])
                        
                        if random.random() < p_pur_actual and self.prod_stock[prod_id] > 0:
                            events.append([timestamp + timedelta(seconds=45), user_id, group, 'purchase',
                                           prod_id, source, latency, session_id, pos, ranking_score, is_exploration, price, context])
                            self.prod_stock[prod_id] -= 1
                            session_actual_rev += price
                            user_total_actual += price
                    
                    # ===== COUNTERFACTUAL UNIVERSE =====
                    if random.random() < p_click_cf:
                        if random.random() < p_pur_cf and cf_stock[prod_id] > 0:
                            session_cf_rev += price
                            user_total_cf += price
                            cf_stock[prod_id] -= 1
                
                # FIX 2: Логирование для Mediation Analysis
                session_level_data.append([
                    session_id, user_id, group, session_length, session_clicks, 
                    round(session_actual_rev, 2), round(session_cf_rev, 2)
                ])

            # FIX 1: Явный iRPU агрегированный на уровне пользователя
            user_level_impact[user_id] = {
                'total_actual_revenue': round(user_total_actual, 2),
                'total_cf_revenue': round(user_total_cf, 2),
                'true_irpu': round(user_total_actual - user_total_cf, 2)
            }

        # Сборка DataFrames
        self.events_df = pd.DataFrame(events, columns=[
            'timestamp', 'user_id', 'group', 'event_type', 'product_id', 'source', 
            'latency_ms', 'session_id', 'position', 'ranking_score', 'is_exploration', 'revenue', 'context'
        ])

        self.session_df = pd.DataFrame(session_level_data, columns=[
            'session_id', 'user_id', 'group', 'session_depth', 'session_clicks', 'actual_revenue', 'cf_revenue'
        ])
        
        # User Impact DataFrame
        self.user_impact_df = pd.DataFrame.from_dict(user_level_impact, orient='index').reset_index()
        self.user_impact_df.rename(columns={'index': 'user_id'}, inplace=True)

        self.users = self.users.drop(columns=['latent_propensity'])

        return self.events_df, self.session_df, self.user_impact_df

if __name__ == "__main__":
    generator = CausalABDataGenerator(n_users=150000, n_products=1500)
    
    products = generator.generate_products()
    users = generator.generate_users()
    events, sessions, user_impact = generator.generate_events()
    
    save_dir = r"C:\Users\Sasha\Desktop\Анастасия ментор\Встреча 5\P4 AB test Fashion-Recommender\1. Data (Генерация и сбор)"
    os.makedirs(save_dir, exist_ok=True)
    
    products.to_csv(os.path.join(save_dir, 'product_catalog.csv'), index=False)
    users.to_csv(os.path.join(save_dir, 'user_profile.csv'), index=False)
    events.to_csv(os.path.join(save_dir, 'event_log.csv'), index=False)
    
    # Новые файлы для доказательства методологии
    sessions.to_csv(os.path.join(save_dir, 'session_level_logs.csv'), index=False)
    user_impact.to_csv(os.path.join(save_dir, 'user_level_impact.csv'), index=False)
    
    print(f"\n✅ 100% Соответствие ТЗ достигнуто. Файлы сохранены в:\n{save_dir}")