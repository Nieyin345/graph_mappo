import requests
import pandas as pd
import os
from data import GlobalGroundStations

# 设定下载一整年的数据 (8760 个小时，完美满足强化学习滑动窗口采样)
start_date = "2023-01-01"
end_date = "2023-12-31"

import numpy as np

def fetch_weather_for_gs(gs, start_date="2023-01-01", end_date="2023-12-31"):
    output_dir = "weather/2023"
    filename = f"{output_dir}/{gs['name']}_weather.csv"
    
    if os.path.exists(filename):
        print(f"[SKIP] {gs['name']} 天气数据已存在，跳过拉取...")
        return False
        
    print(f"正在拉取 {gs['name']} (Lat: {gs['lat']}, Lon: {gs['lon']}) 的全球无死角气象数据...")
    url_main = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params_main = {
        "latitude": gs["lat"],
        "longitude": gs["lon"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,rain,snowfall,cloud_cover_low,cloud_cover_mid,cloud_cover_high,wind_speed_10m,direct_radiation,visibility",
        "daily": "sunrise,sunset",
        "timezone": "UTC"
    }
    
    url_physics = "https://archive-api.open-meteo.com/v1/archive"
    params_physics = {
        "latitude": gs["lat"],
        "longitude": gs["lon"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "boundary_layer_height",
        "timezone": "UTC"
    }
    
    response_main = requests.get(url_main, params=params_main)
    response_physics = requests.get(url_physics, params=params_physics)
    
    if response_main.status_code == 200 and response_physics.status_code == 200:
        data_main = response_main.json()
        data_physics = response_physics.json()
        
        hourly_main = data_main["hourly"]
        hourly_physics = data_physics["hourly"]
        
        df = pd.DataFrame({
            "time": hourly_main["time"],
            "temperature_2m": hourly_main["temperature_2m"],
            "relative_humidity_2m": hourly_main["relative_humidity_2m"],
            "rain_mm": hourly_main["rain"],
            "snowfall_cm": hourly_main["snowfall"],
            "cloud_cover_low": hourly_main["cloud_cover_low"],
            "cloud_cover_mid": hourly_main["cloud_cover_mid"],
            "cloud_cover_high": hourly_main["cloud_cover_high"],
            "wind_speed_10m": hourly_main["wind_speed_10m"],
            "direct_radiation_w": hourly_main["direct_radiation"],
            "visibility_m": hourly_main["visibility"],
            "boundary_layer_height_m": hourly_physics["boundary_layer_height"]
        })
        
        # 解析 daily 层的日出日落，并与 hourly 数据合并
        daily = data_main.get("daily", {})
        if "sunrise" in daily and "sunset" in daily:
            df_daily = pd.DataFrame({
                "date": daily["time"],
                "sunrise": daily["sunrise"],
                "sunset": daily["sunset"]
            })
            # 提取 hourly 数据里的日期进行按天对齐合并
            df["date"] = df["time"].apply(lambda x: x.split("T")[0])
            df = pd.merge(df, df_daily, on="date", how="left")
            df.drop(columns=["date"], inplace=True)
        
        # 清理并重命名，直接保存真实物理量 (precipitation在前面被命名为了precipitation_mm，但之前代码有重命名，现在用rain_mm接收rain，所以去掉多余重命名)
        df.rename(columns={"snowfall_cm": "snow_cm"}, inplace=True)
        
        # 填充缺失值为 0
        df.fillna(0, inplace=True)
        
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(filename, index=False)
        print(f"[OK] Success! Data saved to {filename} ({len(df)} rows)")
        return True
    else:
        print(f"[ERROR] Download failed for {gs['name']}")
        if response_main.status_code != 200:
            print(f"Main API Error ({response_main.status_code}):", response_main.text)
        if response_physics.status_code != 200:
            print(f"Physics API Error ({response_physics.status_code}):", response_physics.text)
        return False

if __name__ == "__main__":
    import time
    # 遍历数据库中的所有地面站，不受 SimConfig.NUM_GS 限制
    all_stations = GlobalGroundStations.STATIONS
    print(f"共 {len(all_stations)} 个地面站需要下载天气数据...")
    for gs in all_stations:
        downloaded = fetch_weather_for_gs(gs)
        if downloaded:
            time.sleep(0.5)  # 仅在真实发起网络请求后等待，大幅减少耗时
    print(f"全部 {len(all_stations)} 个城市的天气数据处理完成！")
