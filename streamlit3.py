import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.express as px
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import os

st.set_page_config(page_title="Dự đoán AQI Việt Nam", layout="wide")

CSV_FILE = os.path.join(os.getcwd(), 'all_cities_aqi_data.csv')
MODEL_FILE = os.path.join(os.getcwd(), 'models', 'aqi_lstm_model.h5')
FEATURES_FILE = os.path.join(os.getcwd(), 'models', 'features_lstm.txt')
SCALER_FILE = os.path.join(os.getcwd(), 'models', 'scaler_lstm.pkl')
RETRAIN_FLAG = os.path.join(os.getcwd(), 'retrain_flag.txt')

@st.cache_resource
def load_model_and_features():
    try:
        model = load_model(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        with open(FEATURES_FILE, 'r', encoding='utf-8') as f:
            features = f.read().strip().split(',')
        return model, scaler, features
    except FileNotFoundError as e:
        st.error(f"File không tồn tại: {str(e)}")
        return None, None, None
    except Exception as e:
        st.error(f"Lỗi tải mô hình hoặc scaler: {str(e)}")
        return None, None, None

model, scaler, features = load_model_and_features()
if model is None or scaler is None or features is None:
    st.error(
        "Không tìm thấy mô hình, scaler hoặc file đặc trưng. Vui lòng huấn luyện mô hình trước bằng `huanluyen6_aqi_only.py`.")
    st.stop()

@st.cache_data
def load_and_preprocess_data():
    try:
        df = pd.read_csv(CSV_FILE, encoding='utf-8-sig')
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f%z', errors='coerce')
        if df['timestamp'].isna().any():
            st.warning(f"Tìm thấy {df['timestamp'].isna().sum()} timestamp không hợp lệ. Các bản ghi này sẽ bị xóa.")
            df = df.dropna(subset=['timestamp'])
        df['aqi'] = pd.to_numeric(df['aqi'], errors='coerce')
        if df['aqi'].isna().any():
            st.warning("Tìm thấy giá trị thiếu trong AQI. Điền bằng trung bình.")
            df['aqi'] = df['aqi'].fillna(df['aqi'].mean())

        city_counts = df['city'].value_counts()
        valid_cities = city_counts[city_counts >= 24].index.tolist()
        if not valid_cities:
            st.error(
                "Không có thành phố nào có đủ dữ liệu (ít nhất 24 bản ghi). Vui lòng thu thập thêm dữ liệu.")
            st.stop()

        preferred_order = ['Hà Nội', 'Hồ Chí Minh', 'Đà Nẵng', 'Cần Thơ', 'Vinh']
        valid_cities = [city for city in preferred_order if city in valid_cities]

        return df, valid_cities
    except FileNotFoundError:
        st.error("Không tìm thấy file dữ liệu. Vui lòng thu thập dữ liệu trước.")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi xử lý dữ liệu: {str(e)}")
        st.stop()

df, valid_cities = load_and_preprocess_data()

def get_vietnam_time():
    return datetime.now(ZoneInfo("Asia/Bangkok"))

def get_aqi_category(aqi):
    if aqi <= 50:
        return "Tốt", "bg-green-500", "Không khí sạch, không gây nguy hiểm.", '#22c55e'
    elif aqi <= 100:
        return "Trung bình", "bg-yellow-500", "Chất lượng không khí chấp nhận được.", '#eab308'
    elif aqi <= 150:
        return "Không tốt cho nhóm nhạy cảm", "bg-orange-500", "Nhóm nhạy cảm có thể gặp vấn đề sức khỏe.", '#f97316'
    elif aqi <= 200:
        return "Có hại", "bg-red-500", "Mọi người bắt đầu cảm nhận được ảnh hưởng sức khỏe.", '#ef4444'
    elif aqi <= 300:
        return "Rất có hại", "bg-purple-500", "Cảnh báo sức khỏe: mọi người bị ảnh hưởng nghiêm trọng hơn.", '#a855f7'
    else:
        return "Nguy hiểm", "bg-maroon-500", "Cảnh báo sức khỏe khẩn cấp.", '#7f1d1d'

@st.cache_data
def preprocess_city_data(city, df, features, time_steps=24):
    city_data = df[df['city'] == city].sort_values('timestamp')
    if len(city_data) < time_steps:
        return None
    city_data = city_data.assign(
        year=city_data['timestamp'].dt.year,
        month=city_data['timestamp'].dt.month,
        day=city_data['timestamp'].dt.day,
        hour=city_data['timestamp'].dt.hour,
        day_of_week=city_data['timestamp'].dt.dayofweek,
        is_weekend=city_data['timestamp'].dt.dayofweek.isin([5, 6]).astype(int),
        sin_hour=np.sin(2 * np.pi * city_data['timestamp'].dt.hour / 24),
        cos_hour=np.cos(2 * np.pi * city_data['timestamp'].dt.hour / 24)
    )
    city_data['aqi_mean_3h'] = city_data['aqi'].shift(1).rolling(window=3, min_periods=1).mean()
    city_data = pd.get_dummies(city_data, columns=['city'], drop_first=True)
    for col in [f for f in features if f.startswith('city_')]:
        if col not in city_data.columns:
            city_data[col] = 0
    return city_data[features].tail(time_steps)

def create_sequence_for_prediction(city, future_datetime, df, features, scaler, time_steps=24, forecast_hours=6):
    city_data = preprocess_city_data(city, df, features, time_steps)
    if city_data is None:
        st.error(f"Không đủ dữ liệu lịch sử cho {city} (cần ít nhất {time_steps} bản ghi).")
        return None

    recent_data_scaled = scaler.transform(city_data)
    predictions = []
    current_sequence = recent_data_scaled.copy()
    for h in range(forecast_hours):
        sequence = np.expand_dims(current_sequence, axis=0)
        pred = model.predict(sequence, verbose=0)[0][0]  # Chỉ lấy AQI
        predictions.append(pred)

        new_data = {
            'year': future_datetime.year,
            'month': future_datetime.month,
            'day': future_datetime.day,
            'hour': future_datetime.hour,
            'day_of_week': future_datetime.weekday(),
            'is_weekend': 1 if future_datetime.weekday() >= 5 else 0,
            'sin_hour': np.sin(2 * np.pi * future_datetime.hour / 24),
            'cos_hour': np.cos(2 * np.pi * future_datetime.hour / 24),
            'aqi_mean_3h': df[df['city'] == city]['aqi'].tail(3).mean() if h == 0 else np.mean(
                predictions[-3:] if len(predictions) >= 3 else predictions[0])
        }
        for city_col in [col for col in features if col.startswith('city_')]:
            city_name = city_col.replace('city_', '')
            new_data[city_col] = 1 if city_name == city else 0

        new_data_scaled = scaler.transform(pd.DataFrame([new_data], columns=features))
        current_sequence = np.vstack((current_sequence[1:], new_data_scaled))
        future_datetime += timedelta(hours=1)

    return predictions

if 'selected_city' not in st.session_state:
    st.session_state.selected_city = 'Hà Nội' if 'Hà Nội' in valid_cities else valid_cities[0] if valid_cities else 'Hà Nội'
if 'selected_city_pred' not in st.session_state:
    st.session_state.selected_city_pred = 'Hà Nội' if 'Hà Nội' in valid_cities else valid_cities[0] if valid_cities else 'Hà Nội'
if 'selected_date' not in st.session_state:
    st.session_state.selected_date = get_vietnam_time().date()
if 'future_date' not in st.session_state:
    st.session_state.future_date = datetime.today().date()
if 'future_time' not in st.session_state:
    current_time = get_vietnam_time()
    rounded_minutes = (current_time.minute // 30) * 30
    st.session_state.future_time = current_time.replace(minute=rounded_minutes, second=0, microsecond=0).time()
if 'forecast_hours' not in st.session_state:
    st.session_state.forecast_hours = 6

st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        body { font-family: Arial, sans-serif; }
        .aqi-gauge { text-align: center; font-size: 3rem; font-weight: bold; padding: 1rem; border-radius: 0.5rem; color: white; }
        .sidebar .sidebar-content { background-color: #f8f9fa; }
        .prediction-box { background-color: #f0f4f8; padding: 1.5rem; border-radius: 0.5rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="text-center py-6">
        <h1 class="text-4xl font-bold text-white-800">Dự đoán Chỉ số Chất lượng Không khí (AQI)</h1>
        <p class="text-lg text-white-600">Theo dõi và dự đoán chất lượng không khí tại các thành phố lớn ở Việt Nam</p>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h2 class='text-2xl font-semibold text-white-800'>Lịch sử AQI</h2>", unsafe_allow_html=True)
    selected_city = st.selectbox("Chọn thành phố để xem lịch sử:", valid_cities, index=valid_cities.index(
        st.session_state.selected_city) if st.session_state.selected_city in valid_cities else 0)
    st.session_state.selected_city = selected_city

    selected_date = st.date_input("Chọn ngày để xem dữ liệu:", min_value=df['timestamp'].min().date(),
                                  max_value=df['timestamp'].max().date(),
                                  value=st.session_state.selected_date)
    st.session_state.selected_date = selected_date

    city_data = df[df['city'] == selected_city].sort_values('timestamp')
    if not city_data.empty:
        latest_data = city_data.iloc[-1]
        latest_aqi = latest_data['aqi']
        latest_timestamp = latest_data['timestamp']
        aqi_category, bg_color, health_impact, plotly_color = get_aqi_category(latest_aqi)
        st.markdown(f"""
            <div class='aqi-gauge {bg_color}'>
                AQI hiện tại: {latest_aqi:.1f} ({aqi_category})
            </div>
            <p class='text-white-600 mt-2'>Thời gian: {latest_timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p class='text-white-600'>{health_impact}</p>
        """, unsafe_allow_html=True)

        daily_data = city_data[city_data['timestamp'].dt.date == selected_date]
        if not daily_data.empty:
            st.markdown(
                f"<h3 class='text-xl font-semibold text-white-700 mt-4'>Lịch sử AQI tại {selected_city} - Ngày {selected_date}</h3>",
                unsafe_allow_html=True)
            try:
                fig = px.line(daily_data, x='timestamp', y=['aqi'],
                              title=f"Lịch sử AQI {selected_city} - Ngày {selected_date}",
                              labels={'timestamp': 'Thời gian', 'value': 'Giá trị', 'variable': 'Biến'},
                              color_discrete_sequence=[plotly_color])
                fig.update_layout(
                    xaxis_tickangle=45,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='black'),
                    title_font=dict(color='black'),
                    xaxis_title_font=dict(color='black'),
                    yaxis_title_font=dict(color='black'),
                    xaxis_tickfont=dict(color='black'),
                    yaxis_tickfont=dict(color='black')
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Lỗi vẽ biểu đồ lịch sử: {str(e)}")
        else:
            st.markdown(f"<p class='text-white-600'>Không có dữ liệu cho {selected_city} vào ngày {selected_date}.</p>",
                        unsafe_allow_html=True)

        last_24h_data = city_data[city_data['timestamp'] >= (city_data['timestamp'].max() - timedelta(hours=24))]
        if not last_24h_data.empty:
            st.markdown(
                f"<h3 class='text-xl font-semibold text-white-700 mt-4'>AQI 24 giờ gần nhất tại {selected_city}</h3>",
                unsafe_allow_html=True)
            try:
                fig_24h = px.line(last_24h_data, x='timestamp', y=['aqi'],
                                  title=f"AQI 24 giờ gần nhất {selected_city}",
                                  labels={'timestamp': 'Thời gian', 'value': 'Giá trị', 'variable': 'Biến'},
                                  color_discrete_sequence=[plotly_color])
                fig_24h.update_layout(
                    xaxis_tickangle=45,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='black'),
                    title_font=dict(color='black'),
                    xaxis_title_font=dict(color='black'),
                    yaxis_title_font=dict(color='black'),
                    xaxis_tickfont=dict(color='black'),
                    yaxis_tickfont=dict(color='black')
                )
                st.plotly_chart(fig_24h, use_container_width=True)
            except Exception as e:
                st.error(f"Lỗi vẽ biểu đồ 24h: {str(e)}")
        else:
            st.markdown("<p class='text-white-600'>Chưa có đủ dữ liệu 24 giờ cho thành phố này.</p>",
                        unsafe_allow_html=True)
    else:
        st.markdown("<p class='text-white-600'>Chưa có dữ liệu lịch sử cho thành phố này.</p>", unsafe_allow_html=True)

st.markdown("<div class='container mx-auto px-4'>", unsafe_allow_html=True)
st.markdown(
    f"<p class='text-lg text-white-600'>Thời gian hiện tại: {get_vietnam_time().strftime('%Y-%m-%d %H:%M:%S')}</p>",
    unsafe_allow_html=True)

st.markdown("<h2 class='text-2xl font-semibold text-white-800 mt-6'>Dự đoán AQI</h2>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])

with col1:
    selected_city_pred = st.selectbox("Chọn thành phố để dự đoán:", valid_cities, index=valid_cities.index(
        st.session_state.selected_city_pred) if st.session_state.selected_city_pred in valid_cities else 0)
    st.session_state.selected_city_pred = selected_city_pred

    future_date = st.date_input("Chọn ngày dự đoán:", min_value=datetime.today().date(),
                                value=st.session_state.future_date)
    st.session_state.future_date = future_date

    future_time = st.time_input("Chọn giờ bắt đầu:", value=st.session_state.future_time, step=1800)
    st.session_state.future_time = future_time

    st.session_state.forecast_hours = st.slider("Dự đoán trong bao nhiêu giờ tiếp theo:", 1, 12,
                                                st.session_state.forecast_hours)

with col2:
    if st.button("Dự đoán"):
        with st.spinner("Đang dự đoán..."):
            future_datetime = datetime.combine(future_date, future_time).replace(tzinfo=ZoneInfo("Asia/Bangkok"))
            predictions = create_sequence_for_prediction(selected_city_pred, future_datetime, df, features, scaler,
                                                         forecast_hours=st.session_state.forecast_hours)
            if predictions is not None:
                pred_data = []
                for h, aqi in enumerate(predictions):
                    pred_time = future_datetime + timedelta(hours=h)
                    aqi_category, bg_color, health_impact, plotly_color = get_aqi_category(aqi)
                    pred_data.append({
                        'Thời gian': pred_time,
                        'AQI': aqi,
                        'Mức độ AQI': aqi_category,
                        'Ảnh hưởng sức khỏe': health_impact
                    })

                pred_df = pd.DataFrame(pred_data)
                st.markdown("<h3 class='text-xl font-semibold text-white-800'>Kết quả dự đoán</h3>",
                            unsafe_allow_html=True)
                st.dataframe(
                    pred_df.style.format({'AQI': '{:.1f}'}))

                latest_aqi = pred_df['AQI'].iloc[-1]
                _, _, _, plotly_color = get_aqi_category(latest_aqi)
                fig = px.line(pred_df, x='Thời gian', y=['AQI'],
                              title=f"Dự đoán AQI tại {selected_city_pred}",
                              labels={'Thời gian': 'Thời gian', 'value': 'Giá trị', 'variable': 'Biến'},
                              color_discrete_sequence=[plotly_color])
                fig.update_layout(
                    xaxis_tickangle=45,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='black'),
                    title_font=dict(color='black'),
                    xaxis_title_font=dict(color='black'),
                    yaxis_title_font=dict(color='black'),
                    xaxis_tickfont=dict(color='black'),
                    yaxis_tickfont=dict(color='black')
                )
                st.plotly_chart(fig, use_container_width=True)

                if any(pred_df['AQI'] > 300):
                    st.markdown(
                        "<p class='text-red-600 font-bold'>CẢNH BÁO: AQI dự đoán vượt ngưỡng nguy hiểm (>300)!</p>",
                        unsafe_allow_html=True)

col_refresh, col_retrain = st.columns(2)
with col_refresh:
    if st.button("Làm mới dữ liệu"):
        st.rerun()
with col_retrain:
    if st.button("Huấn luyện lại mô hình"):
        with open(RETRAIN_FLAG, 'w', encoding='utf-8') as f:
            f.write("retrain")
        st.info("Đã yêu cầu huấn luyện lại mô hình. Vui lòng chạy lại file `huanluyen6_aqi_only.py`.")

st.markdown("</div>", unsafe_allow_html=True)