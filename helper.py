import os
import math
import numpy as np
import pandas as pd
import xarray as xr
import cdsapi
from pyproj import Transformer

# Global Constants
R           = 6371.0  # Earth's radius in km

class GeoMath:
    """
    Pure geometric and Earth-centric calculations.
    """
    @staticmethod
    def latlon_to_cartesian(lat, lon, alt=0.0):
        """Convert lat/lon (deg) and altitude (km) to 3D Cartesian coordinates (x,y,z) (ECEF)."""
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        r = R + alt
        x = r * np.cos(lat_rad) * np.cos(lon_rad)
        y = r * np.cos(lat_rad) * np.sin(lon_rad)
        z = r * np.sin(lat_rad)
        return np.array([x, y, z])

    @staticmethod
    def calculate_3d_distance(lat1, lon1, alt1, lat2, lon2, alt2):
        """Calculate the true 3D straight-line distance (line-of-sight) between two points."""
        vec1 = GeoMath.latlon_to_cartesian(lat1, lon1, alt1)
        vec2 = GeoMath.latlon_to_cartesian(lat2, lon2, alt2)
        return np.linalg.norm(vec1 - vec2)

    @staticmethod
    def elevation_angle_deg(observer_lat, observer_lon, observer_alt,
                            target_lat, target_lon, target_alt):
        """Elevation angle from observer to target using local ENU geometry."""
        obs = GeoMath.latlon_to_cartesian(observer_lat, observer_lon, observer_alt)
        tgt = GeoMath.latlon_to_cartesian(target_lat, target_lon, target_alt)
        los = tgt - obs
        los_norm = np.linalg.norm(los)
        if los_norm <= 0:
            return 90.0

        lat_rad = np.radians(observer_lat)
        lon_rad = np.radians(observer_lon)
        up = np.array([
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad),
        ])
        sin_elev = np.dot(los, up) / los_norm
        sin_elev = np.clip(sin_elev, -1.0, 1.0)
        return float(np.degrees(np.arcsin(sin_elev)))

    @staticmethod
    def check_line_of_sight(lat1, lon1, alt1, lat2, lon2, alt2):
        """Check if the line of sight between two 3D points is blocked by the Earth's curvature."""
        vec1 = GeoMath.latlon_to_cartesian(lat1, lon1, alt1)
        vec2 = GeoMath.latlon_to_cartesian(lat2, lon2, alt2)
        
        d = vec2 - vec1
        d_mag_sq = np.dot(d, d)
        if d_mag_sq == 0:
            return True # Same point
            
        t = -np.dot(vec1, d) / d_mag_sq
        
        if 0 < t < 1:
            closest_point = vec1 + t * d
            min_dist = np.linalg.norm(closest_point)
            if min_dist < R:
                return False
                
        return True

    @staticmethod
    def is_line_of_sight_visible(node1, node2, t):
        """
        判断两个物理节点 (GS, HAP, SAT) 在时间步 t 是否能够直线可视（未被地球遮挡）。
        """
        from data import gs, hap, sat  # 局部导入防止循环依赖
        
        def extract_coords(node):
            if isinstance(node, gs):
                return node.la, node.lg, 0.0
            elif isinstance(node, hap) or isinstance(node, sat):
                return node.la[t], node.lg[t], node.H[t]
            else:
                raise TypeError(f"Unknown node type: {type(node)}")
                
        lat1, lon1, alt1 = extract_coords(node1)
        lat2, lon2, alt2 = extract_coords(node2)
        
        return GeoMath.check_line_of_sight(lat1, lon1, alt1, lat2, lon2, alt2)

    @staticmethod
    def cartesian_to_latlon(x, y, z):
        """Convert Cartesian coordinates back to lat/lon (deg)."""
        lat = np.degrees(np.arcsin(z / R))
        lon = np.degrees(np.arctan2(y, x))
        return lat, lon + 360

    @staticmethod
    def rotation_matrix_from_vectors(a, b):
        """Find rotation matrix that rotates vector a → b on the sphere."""
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        v = np.cross(a, b)
        c = np.dot(a, b)
        if np.isclose(c, 1.0):
            return np.eye(3)
        if np.isclose(c, -1.0):
            perp = np.array([1,0,0]) if not np.allclose(a,[1,0,0]) else np.array([0,1,0])
            v = np.cross(a, perp)
            v /= np.linalg.norm(v)
            H = np.array([[0, -v[2], v[1]],
                          [v[2], 0, -v[0]],
                          [-v[1], v[0], 0]])
            return -np.eye(3) + 2 * np.outer(v,v)
        s = np.linalg.norm(v)
        k = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
        Rm = np.eye(3) + k + k @ k * ((1-c) / (s**2))
        return Rm

    @staticmethod
    def shift_trajectory(lats, lons, new_lat0, new_lon0):
        """Shift a trajectory so that its first point moves to (new_lat0, new_lon0)."""
        lat0, lon0 = lats[0], lons[0]
        if lat0 != new_lat0 or lon0 != new_lon0:
            traj_xyz = np.array([GeoMath.latlon_to_cartesian(lat, lon) for lat, lon in zip(lats, lons)])
            start_vec = traj_xyz[0]
            new_start_vec = GeoMath.latlon_to_cartesian(new_lat0, new_lon0)
            Rm = GeoMath.rotation_matrix_from_vectors(start_vec, new_start_vec)
            rotated_xyz = traj_xyz @ Rm.T
            new_lats, new_lons = [], []
            for x, y, z in rotated_xyz:
                lat, lon = GeoMath.cartesian_to_latlon(x, y, z)
                new_lats.append(lat)
                new_lons.append(lon)
            return new_lats, new_lons
        else:
            return lats, lons

    @staticmethod
    def latlon_to_tangent(lon, lat, lon0, lat0):
        """Project lat/lon (deg) into local tangent plane (x,y) in km."""
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        lat0_rad = np.radians(lat0)
        lon0_rad = np.radians(lon0)
        dlat = lat_rad - lat0_rad
        dlon = lon_rad - lon0_rad
        x = R * dlon * np.cos(lat0_rad)
        y = R * dlat
        return x, y

    @staticmethod
    def tangent_to_latlon(x, y, lon0, lat0):
        """Convert local tangent plane (x,y) back to lat/lon (deg)."""
        lat0_rad = np.radians(lat0)
        lon0_rad = np.radians(lon0)
        dlat = y / R
        dlon = x / (R * np.cos(lat0_rad))
        lat = lat0_rad + dlat
        lon = lon0_rad + dlon
        return np.degrees(lon), np.degrees(lat)

class CoordinateTransformer:
    """
    Encapsulates map projections (Web Mercator ↔ WGS84)
    """
    to_xy_transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    to_latlon_transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    @classmethod
    def lonlat_to_xy(cls, lon, lat):
        x, y = cls.to_xy_transformer.transform(lon, lat)
        return [x, y]

    @classmethod
    def xy_to_lonlat(cls, x, y):
        lon, lat = cls.to_latlon_transformer.transform(x, y)
        return [lon + 360, lat]

class HapMobilityModel:
    """
    Manages HAP positions, reading from wind data or stratospheric balloon datasets.
    """
    @staticmethod
    def _compute_direction(u, v):
        return (np.degrees(np.arctan2(v, u)) + 360) % 360

    @staticmethod
    def _compute_speed(u, v):
        return np.sqrt(u**2 + v**2)

    @staticmethod
    def _download_wind_data(file_name, level):
        lat, lon         = 49.0, 279.0
        year, month, day = "2025", "07", "12"

        if os.path.exists(file_name):
            return

        c = cdsapi.Client()
        hours = ["00:00", "06:00", "12:00", "18:00"]
        delta = 0.125
        area = [lat + delta, lon - delta, lat - delta, lon + delta]

        c.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": ["u_component_of_wind", "v_component_of_wind"],
                "pressure_level": [level],
                "year": year,
                "month": month,
                "day": day,
                "time": hours,
                "area": area,
            },
            file_name
        )

    @staticmethod
    def update_hap_coordinates(method, hnodes, syst):
        level = "50"
        file_name = f"weather/wind_{level}hpa_era5.nc"
        
        if method == "wind":
            HapMobilityModel._download_wind_data(file_name, level)
            ds = xr.open_dataset(file_name)
            u = ds['u']
            v = ds['v']
            time = ds['valid_time']
            lat = ds['latitude']
            lon = ds['longitude']
            
            u = ds.u.sel(latitude=lat, longitude=lon, method="nearest").values.flatten()
            v = ds.v.sel(latitude=lat, longitude=lon, method="nearest").values.flatten()
            time = ds.valid_time.values
            hnodes_xy = np.zeros((len(syst.T),2))
            
        elif method == "stratotegic":
            df = pd.read_csv("dataset/balloon_sim_data.csv")
            df = df.drop_duplicates(subset="Time_s", keep="first").sort_values("Time_s")
            
            max_time = df["Time_s"].max()
            required_max_time = max((len(syst.T) - 1) * syst.THETA, max_time)
            new_times = np.arange(0, required_max_time + syst.THETA, syst.THETA)
            
            result = pd.DataFrame({"Time_s": new_times})
            result["Longitude_deg"] = np.interp(new_times, df["Time_s"], df["Longitude_deg"])
            result["Latitude_deg"] = np.interp(new_times, df["Time_s"], df["Latitude_deg"])
            result["Altitude_m"] = np.interp(new_times, df["Time_s"], df["Altitude_m"])
            
            lons = result["Longitude_deg"].tolist()
            lats = result["Latitude_deg"].tolist()
            
            new_lats = {}
            new_lons = {}
            
            for idx_hnode, hnode in enumerate(hnodes):
                new_lats[idx_hnode], new_lons[idx_hnode] = GeoMath.shift_trajectory(
                    lats, lons, hnode.la[0], hnode.lg[0]
                )
        else:
            raise ValueError("Method must be either 'wind' or 'stratotegic'")

        if method == "wind":
            for idx_hnode, hnode in enumerate(hnodes):
                hnode.H[0] = 15
        elif method == "stratotegic":
            for idx_hnode, hnode in enumerate(hnodes):
                hnode.H[0] = 25
        
        for t in syst.T[1:]:
            if method == "wind":
                for idx_hnode, hnode in enumerate(hnodes):
                    x, y = CoordinateTransformer.lonlat_to_xy(hnode.lg[t-1], hnode.la[t-1])
                    wind_idx = min(t - 1, len(time) - 1)
                    u_val = ds.u.sel(
                        valid_time=time[wind_idx], latitude=hnode.la[t-1], longitude=hnode.lg[t-1], method="nearest"
                    ).values.item()
                    v_val = ds.v.sel(
                        valid_time=time[wind_idx], latitude=hnode.la[t-1], longitude=hnode.lg[t-1], method="nearest"
                    ).values.item()
                    
                    x_new = x + u_val * syst.THETA
                    y_new = y + v_val * syst.THETA
                    hnodes_xy[t] = [x_new, y_new]
                    lon_new, lat_new = CoordinateTransformer.xy_to_lonlat(x_new, y_new)
                    hnode.lg[t] = lon_new
                    hnode.la[t] = lat_new
            elif method == "stratotegic":
                row = result.loc[result["Time_s"] == t * syst.THETA]
                if not row.empty:
                    for idx_hnode, hnode in enumerate(hnodes):
                        lon = new_lons[idx_hnode][t]
                        lat = new_lats[idx_hnode][t]
                        alt = row["Altitude_m"].values[0]
                        hnode.lg[t] = lon
                        hnode.la[t] = lat
                        hnode.H[t]  = alt / 1000


class KeplerianPropagator:
    """
    高精度开普勒轨道力学传播器。
    支持偏心率 e>0 带来的高度变化，并考虑地球自转对地面投影轨迹的影响。
    """
    GM_KEPLER = 398600.4418
    OMEGA_E   = 7.2921159e-5
    
    def __init__(self, a, e, i_deg, raan_deg, aop_deg, m0_deg):
        self.a = a
        self.e = e
        self.i = np.radians(i_deg)
        self.raan0 = np.radians(raan_deg)
        self.aop = np.radians(aop_deg)
        self.M0 = np.radians(m0_deg)
        self.n = np.sqrt(self.GM_KEPLER / (self.a**3))

    def _solve_kepler(self, M, tol=1e-8, max_iter=100):
        E = M
        for _ in range(max_iter):
            dE = (E - self.e * np.sin(E) - M) / (1.0 - self.e * np.cos(E))
            E = E - dE
            if abs(dE) < tol:
                break
        return E

    def get_position_at_time(self, t_seconds):
        raan = self.raan0 - self.OMEGA_E * t_seconds
        M = self.M0 + self.n * t_seconds
        E = self._solve_kepler(M)
        tan_half_nu = np.sqrt((1 + self.e) / (1 - self.e)) * np.tan(E / 2.0)
        nu = 2.0 * np.arctan(tan_half_nu)
        r = self.a * (1.0 - self.e * np.cos(E))
        altitude = r - R
        u = self.aop + nu
        x_orb = r * np.cos(u)
        y_orb = r * np.sin(u)
        X = x_orb * np.cos(raan) - y_orb * np.cos(self.i) * np.sin(raan)
        Y = x_orb * np.sin(raan) + y_orb * np.cos(self.i) * np.cos(raan)
        Z = y_orb * np.sin(self.i)
        lat = np.degrees(np.arcsin(Z / r))
        lon = np.degrees(np.arctan2(Y, X))
        lon = (lon + 180) % 360 - 180
        return lat, lon, altitude
