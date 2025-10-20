from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from logging import getLogger, StreamHandler, Formatter

from dataclasses import dataclass, InitVar, field
from typing import Any, Optional

import os
import datetime
import numpy as np
import pandas as pd

import shapely
from pyproj import Proj

__all__ = ["PP", "Trip", "Feeder"]


# logger
loglevel = os.environ.get("LOGLEVEL", "WARNING").upper()
log_format = "[%(asctime)s] %(levelname)s:%(filename)s %(lineno)d:%(message)s"
logger = getLogger(__name__)
formatter = Formatter(log_format)
handler = StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(loglevel)

def read_csv(file_path: str) -> pd.DataFrame:
    """
    Read CSV file using pandas with pyarrow engine.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the CSV data.
    """
    return pd.read_csv(file_path, engine="pyarrow")


@dataclass
class PP:
    """
    PP data definition. Hold Trip data inside. Trimming, cleanig etc.
    """
    trips: list["Trip"]
    utm_zone: int

    def __len__(self) -> int:
        return len(self.trips)
    
    def clip(self, start_time: datetime.datetime | None, end_time: datetime.datetime | None) -> "PP":
        """
        Clip GPS data by time.

        Args:
            start_time (datetime.datetime | None): Start time. If None, use trip's dep_time.
            end_time (datetime.datetime | None): End time. If None, use trip's arr_time.

        Returns:
            PP: New PP object with clipped trips.
        """
        new_trips = [t.clip(start_time, end_time) for t in self.trips]
        new_trips = [t for t in new_trips if len(t) > 0]
        new_pp = PP(new_trips, self.utm_zone)
        return new_pp

    def crop(self, polygon: list[tuple[float, float]] | shapely.geometry.Polygon) -> "PP":
        """
        Crop GPS data by polygon.

        Args:
            polygon (list[tuple[float, float]] | shapely.geometry.Polygon): Polygon to crop.
        Returns:
            PP: New PP object with cropped trips.
        """
        new_trips = [t.crop(polygon) for t in self.trips]
        new_trips = [t for t in new_trips if len(t) > 0]
        new_pp = PP(new_trips, self.utm_zone)
        return new_pp
    
    def split_by_thresh(self, thresh: int) -> "PP":
        """
        Split GPS data if no point is recorded in threshold (sec) .

        Args:
            thresh (int): Threshold in seconds.
        Returns:
            PP: New PP object with split trips.
        """
        new_trips = [t.split_by_thresh(thresh) for t in self.trips]
        new_pp = PP(new_trips, self.utm_zone)
        return new_pp
    
    def get_transport_mode(self, mode: str) -> "PP":
        """
        Extract feeders with specific transportation mode.

        Args:
            mode (str): Transportation mode.
        Returns:
            PP: New PP object with trips containing the specified mode.
        """
        new_trips = [t.get_transport_mode(mode) for t in self.trips]
        new_trips = [t for t in new_trips if len(t) > 0]
        new_pp = PP(new_trips, self.utm_zone)
        return new_pp
    
    def get_purposes(self, purpose: str) -> "PP":
        """
        Extract feeders with specific trip purpose.

        Args:
            purpose (str): Trip purpose.
        Returns:
            PP: New PP object with trips containing the specified purpose.
        """
        new_trips = [t for t in self.trips if t.purpose == purpose]
        new_pp = PP(new_trips, self.utm_zone)
        return new_pp
    
    def get_all_modes(self) -> list[str]:
        """
        Get all transportation modes.

        Returns:
            list[str]: List of all transportation modes.
        """
        modes = set()
        for trip in self.trips:
            for feeder in trip.feeders:
                modes.add(str(feeder.transport_mode))
        return list(modes)
    
    def extract(self, mode: Optional[str | list[str]] = None, purpose: Optional[str | list[str]] = None) -> "PP":
        """
        Extract feeders with specific transportation mode and trip purpose.

        Args:
            mode (Optional[str | list[str]], optional): Transportation mode. Defaults to None.
            purpose (Optional[str | list[str]], optional): Trip purpose. Defaults to None. 

        Returns:
            PP: New PP object with extracted trips.
        """
        if mode is None and purpose is None:
            return self
        if isinstance(mode, str):
            mode = [mode]
        if isinstance(purpose, str):
            purpose = [purpose]

        new_trips = []
        for trip in self.trips:
            if purpose is not None and trip.purpose not in purpose:
                continue
            if mode is None:
                new_trips.append(trip)
            else:
                for m in mode:
                    new_trip = trip.get_transport_mode(m)
                    if len(new_trip) > 0:
                        new_trips.append(new_trip)
        new_pp = PP(new_trips, self.utm_zone)
        return new_pp
    
    @staticmethod
    def load(base_dir: str, trip_file: str, feeder_file: str, loc_file: str, loc_time_format: str = "%Y-%m-%d %H:%M:%S.%f", feeder_time_format: str ="%Y-%m-%d %H:%M:%S", trip_time_format: str = "%Y-%m-%d %H:%M:%S", max_trip: int | None = None) -> "PP":
        """
        Load PP data from file.

        Args:
            base_dir (str): Base directory containing the data files.
            trip_file (str): Trip data file name.
            feeder_file (str): Feeder data file name.
            loc_file (str): Location data file name.
            loc_time_format (str, optional): Time format for location data. Defaults to "%Y-%m-%d %H:%M:%S.%f".
            feeder_time_format (str, optional): Time format for feeder data. Defaults to "%Y-%m-%d %H:%M:%S".
            trip_time_format (str, optional): Time format for trip data. Defaults to "%Y-%m-%d %H:%M:%S".
            max_trip (int | None, optional): Maximum number of trips. Defaults to None.
        Returns:
            PP: Loaded PP object.
        """
        # determine UTM zone from loc data
        utm_proj = None
        zone = None
        for cur_dir, dirs, files in os.walk(base_dir):
            if trip_file in files and feeder_file in files and loc_file in files:
                loc_path = os.path.join(cur_dir, loc_file)
                loc_data = read_csv(loc_path)  # ID[,accuracy,bearing,speed,ユーザーID,作成日時],経度,緯度,記録日時[,高度]
                lon_sample = loc_data["経度"].to_numpy()[0]
                zone = int(lon_sample // 6) + 31  # UTM zone calculation based on longitude
                utm_proj = Proj(proj="utm", zone=zone, ellps="WGS84")
                break
        if utm_proj is None or zone is None:
            raise ValueError("No location data found.")
        
        trips = []
        for cur_dir, dirs, files in os.walk(base_dir):
            if trip_file in files and feeder_file in files and loc_file in files:
                trip_path = os.path.join(cur_dir, trip_file)
                feeder_path = os.path.join(cur_dir, feeder_file)
                loc_path = os.path.join(cur_dir, loc_file)

                trip_data = read_csv(trip_path)  # ID,ユーザーID[,作成日時],出発時刻,到着時刻[,更新日時,有効性],目的コード（active）
                feeder_data = read_csv(feeder_path)  # ID,トリップID,ユーザーID[,作成日時,操作タイプ(1:出発、5:移動手段変更),更新日時,有効性],移動手段コード,記録日時
                loc_data = read_csv(loc_path)  # ID[,accuracy,bearing,speed,ユーザーID,作成日時],経度,緯度,記録日時[,高度]

                feeder_data.set_index("ID", inplace=True, drop=False)
                trip_data.set_index("ID", inplace=True, drop=False)

                trip_data["出発時刻"] = np.vectorize(lambda x: datetime.datetime.strptime(x, trip_time_format))(trip_data["出発時刻"].astype(str).to_numpy())
                trip_data["到着時刻"] = np.vectorize(lambda x: datetime.datetime.strptime(x, trip_time_format))(trip_data["到着時刻"].astype(str).to_numpy())
                feeder_data["記録日時"] = np.vectorize(lambda x: datetime.datetime.strptime(x, feeder_time_format))(feeder_data["記録日時"].astype(str).to_numpy())
                loc_data["記録日時"] = np.vectorize(lambda x: datetime.datetime.strptime(x, loc_time_format))(loc_data["記録日時"].astype(str).to_numpy())

                # Sort by time
                loc_data.sort_values(by=["ユーザーID", "記録日時"], inplace=True)
                feeder_data.sort_values(by=["ユーザーID", "トリップID", "記録日時"], inplace=True)
                trip_data.sort_values(by=["ユーザーID", "出発時刻"], inplace=True)

                loc_coords = loc_data[["経度", "緯度"]].to_numpy()
                loc_data[["x", "y"]] = np.array([utm_proj(lon, lat) for lon, lat in loc_coords])

                uids = np.unique(trip_data["ユーザーID"])
                for uid in uids:
                    user_loc = loc_data[loc_data["ユーザーID"] == uid]
                    user_trip = trip_data[trip_data["ユーザーID"] == uid]
                    tids = np.unique(user_trip["ID"])
                    for tid in tids:
                        trip_feeder = feeder_data[feeder_data["トリップID"] == tid]
                        fids = sorted(np.unique(trip_feeder["ID"]))

                        feeders = []
                        for i in range(len(fids) - 1):
                            target_loc = user_loc[(user_loc["記録日時"] >= trip_feeder.loc[fids[i], "記録日時"]) & (user_loc["記録日時"] < trip_feeder.loc[fids[i + 1], "記録日時"])]
                            if len(target_loc) > 0:
                                feeders.append(Feeder(fids[i], trip_feeder.loc[fids[i], "移動手段コード"], target_loc["記録日時"], target_loc[["x", "y"]]))
                        target_loc = user_loc[user_loc["記録日時"] >= trip_feeder.loc[fids[-1], "記録日時"]]
                        if len(target_loc) > 0:
                            feeders.append(Feeder(fids[-1], trip_feeder.loc[fids[-1], "移動手段コード"].astype(str), target_loc["記録日時"].to_numpy(), target_loc[["x", "y"]].to_numpy()))

                        trip = Trip(tid, uid, feeders, purpose=user_trip.loc[tid, "目的コード（active）"].astype(str))
                        trips.append(trip)

                        if max_trip is not None and len(trips) >= max_trip:
                            break
                    if max_trip is not None and len(trips) >= max_trip:
                        break
                if max_trip is not None and len(trips) >= max_trip:
                    break
            if max_trip is not None and len(trips) >= max_trip:
                break

        if len(trips) == 0:
            raise ValueError("No trip data found.")

        pp = PP(trips, zone)
        return pp
    
    @staticmethod
    def save(pp: "PP", trip_file: str, feeder_file: str, loc_file: str, loc_time_format: str = "%Y-%m-%d %H:%M:%S.%f", feeder_time_format: str ="%Y-%m-%d %H:%M:%S", trip_time_format: str = "%Y-%m-%d %H:%M:%S") -> None:
        """
        Save PP data to file.

        Args:
            pp (PP): PP object to save.
            trip_file (str): Trip data file name.
            feeder_file (str): Feeder data file name.
            loc_file (str): Location data file name.
            loc_time_format (str, optional): Time format for location data. Defaults to "%Y-%m-%d %H:%M:%S.%f".
            feeder_time_format (str, optional): Time format for feeder data. Defaults to "%Y-%m-%d %H:%M:%S".
            trip_time_format (str, optional): Time format for trip data. Defaults to "%Y-%m-%d %H:%M:%S".
        """
        if not os.path.exists(os.path.dirname(trip_file)):
            os.makedirs(os.path.dirname(trip_file))
        if not os.path.exists(os.path.dirname(feeder_file)):
            os.makedirs(os.path.dirname(feeder_file))
        if not os.path.exists(os.path.dirname(loc_file)):
            os.makedirs(os.path.dirname(loc_file))

        inv_utm_proj = Proj(proj="utm", zone=pp.utm_zone, ellps="WGS84", inverse=True)

        trip_data = []
        feeder_data = []
        loc_data = []

        loc_id = 1
        for trip in pp.trips:
            if trip.dep_time is None or trip.arr_time is None:
                logger.warning(f"Trip {trip.id} has no dep_time or arr_time.")
                continue
            trip_data.append([trip.id, trip.user_id, trip.dep_time.strftime(trip_time_format), trip.arr_time.strftime(trip_time_format), trip.purpose])
            for feeder in trip.feeders:
                if feeder.dep_time is None:
                    logger.warning(f"Feeder {feeder.id} has no dep_time.")
                    continue
                feeder_data.append([feeder.id, trip.id, trip.user_id, feeder.transport_mode, feeder.dep_time.strftime(feeder_time_format)])
                for i in range(len(feeder)):
                    time, point = feeder[i]
                    point = inv_utm_proj(point[0], point[1])
                    loc_data.append([loc_id, trip.user_id, point[0], point[1], time.strftime(loc_time_format)])
                    loc_id += 1

        trip_df = pd.DataFrame(trip_data, columns=["ID", "ユーザーID", "出発時刻", "到着時刻", "目的コード（active）"])
        feeder_df = pd.DataFrame(feeder_data, columns=["ID", "トリップID", "ユーザーID", "移動手段コード", "記録日時"])
        loc_df = pd.DataFrame(loc_data, columns=["ID", "ユーザーID", "経度", "緯度", "記録日時"])

        trip_df.to_csv(trip_file, index=False)
        feeder_df.to_csv(feeder_file, index=False)
        loc_df.to_csv(loc_file, index=False)


@dataclass
class Trip:
    """
    Trip data definition. Single trip has single trip purpose. Hold Feeder data inside.
    """
    id: int
    user_id: int
    feeders: list["Feeder"]

    purpose: str = ""
    dep_time: Optional[datetime.datetime] = None
    arr_time: Optional[datetime.datetime] = None

    def __post_init__(self) -> None:
        # sort feeders by dep_time
        if len(self.feeders) == 0:
            self.dep_time = None
            self.arr_time = None
            return
        feeders = sorted(self.feeders)
        for i in range(1, len(feeders)):
            t1 = feeders[i - 1].arr_time
            t2 = feeders[i].dep_time
            if t1 is not None and t2 is not None:
                if t1 > t2:
                    logger.warning("Feeders are not in order.")
            else:
                logger.warning("Feeders are not in order.")
        self.dep_time = feeders[0].dep_time
        self.arr_time = feeders[-1].arr_time

    def __len__(self) -> int:
        return len(self.feeders)
    
    def clip(self, start_time: Optional[datetime.datetime], end_time: Optional[datetime.datetime]) -> "Trip":
        """
        Clip GPS data by time.

        Args:
            start_time (datetime.datetime | None): Start time. If None, use trip's dep_time.
            end_time (datetime.datetime | None): End time. If None, use trip's arr_time.
        Returns:
            Trip: New Trip object with clipped feeders.
        """
        if start_time is None:
            start_time = self.dep_time
        if end_time is None:
            end_time = self.arr_time
        new_feeders = [f.clip(start_time, end_time) for f in self.feeders]
        new_feeders = [f for f in new_feeders if len(f) > 0]
        new_trip = Trip(self.id, self.user_id, new_feeders)
        return new_trip
    
    def crop(self, polygon: list[tuple[float, float]] | shapely.geometry.Polygon) -> "Trip":
        """
        Crop GPS data by polygon.

        Args:
            polygon (list[tuple[float, float]] | shapely.geometry.Polygon): Polygon to crop.

        Returns:
            Trip: New Trip object with cropped feeders.
        """
        new_feeders = [f.crop(polygon) for f in self.feeders]
        new_feeders = [f for f in new_feeders if len(f) > 0]
        new_trip = Trip(self.id, self.user_id, new_feeders)
        return new_trip
    
    def get_transport_mode(self, mode: str) -> "Trip":
        """
        Extract feeders with specific transportation mode.
        """
        new_feeders = [f for f in self.feeders if f.transport_mode == mode]
        new_trip = Trip(self.id, self.user_id, new_feeders)
        return new_trip
    
    def split_by_thresh(self, thresh: int) -> "Trip":
        """
        Split GPS data if no point is recorded in threshold (sec) .

        Args:
            thresh (int): Threshold in seconds.
        Returns:
            Trip: New Trip object with split feeders.
        """
        new_feeders = []
        for f in self.feeders:
            new_feeders.extend(f.split_by_thresh(thresh))
        new_trip = Trip(self.id, self.user_id, new_feeders)
        return new_trip


@dataclass
class Feeder:
    """
    Feeder data definition. Single feeder has single transportation mode. Hold GPS data inside.
    """
    id: int
    transport_mode: str
    gps_times: Optional[np.ndarray]
    gps_points: Optional[np.ndarray]  # (x, y) in rectangular coordinate

    dep_time: Optional[datetime.datetime] = None
    arr_time: Optional[datetime.datetime] = None

    def __post_init__(self) -> None:
        if self.gps_times is None or len(self.gps_times) == 0:
            self.gps_times = None
            self.gps_points = None
            return
        self.gps_times = np.array(self.gps_times)
        self.gps_points = np.array(self.gps_points)

        idxs = np.argsort(self.gps_times)
        self.gps_times = self.gps_times[idxs]
        self.gps_points = self.gps_points[idxs]

        self.dep_time = self.gps_times[0]
        self.arr_time = self.gps_times[-1]

    def __len__(self) -> int:
        if self.gps_points is None:
            return 0
        return len(self.gps_points)

    def __getitem__(self, idx: int) -> tuple[datetime.datetime, tuple[float, float]]:
        """
        Get GPS time and point at index idx.
        Args:
            idx (int): Index of the GPS data.
        Returns:
            tuple[datetime.datetime, tuple[float, float]]: GPS time and point at index idx.
        """
        if self.gps_times is None or self.gps_points is None:
            raise IndexError("Feeder has no GPS data.")
        return (self.gps_times[idx], self.gps_points[idx])

    def __lt__(self, other: "Feeder") -> bool:
        """
        Compare two feeders by departure time.
        Args:
            other (Feeder): Other feeder to compare with.
        Returns:
            bool: True if this feeder departs before the other feeder, False otherwise.
        """
        if self.dep_time is None or other.dep_time is None:
            raise NotImplementedError("Cannot compare with None.")
        return self.dep_time < other.dep_time

    def __gt__(self, other: "Feeder") -> bool:
        """
        Compare two feeders by departure time.
        Args:
            other (Feeder): Other feeder to compare with.
        Returns:
            bool: True if this feeder departs after the other feeder, False otherwise.
        """
        if self.dep_time is None or other.dep_time is None:
            raise NotImplementedError("Cannot compare with None.")
        return self.dep_time > other.dep_time

    def __le__(self, other: "Feeder") -> bool:
        if self.dep_time is None or other.dep_time is None:
            raise NotImplementedError("Cannot compare with None.")
        return self.dep_time <= other.dep_time

    def __ge__(self, other: "Feeder") -> bool:
        """
        Compare two feeders by departure time.
        Args:
            other (Feeder): Other feeder to compare with.
        Returns:
            bool: True if this feeder departs after the other feeder, False otherwise.
        """
        if self.dep_time is None or other.dep_time is None:
            raise NotImplementedError("Cannot compare with None.")
        return self.dep_time >= other.dep_time

    def __eq__(self, other: object) -> bool:
        """
        Check if two feeders are equal by id, dep_time, arr_time, gps_times and gps_points.
        Args:
            other (object): Other object to compare with.
        Returns:
            bool: True if all attributes are equal, False otherwise.
        """
        if not isinstance(other, Feeder):
            return NotImplemented
        
        equal_gps_times = True
        if self.gps_times is not None and other.gps_times is not None:
            equal_gps_times = np.array_equal(self.gps_times, other.gps_times)
        equal_gps_points = True
        if self.gps_points is not None and other.gps_points is not None:
            equal_gps_points = np.array_equal(self.gps_points, other.gps_points)

        return self.id == other.id and self.dep_time == other.dep_time and self.arr_time == other.arr_time and equal_gps_times and equal_gps_points

    def clip(self, start_time: Optional[datetime.datetime], end_time: Optional[datetime.datetime]) -> "Feeder":
        """
        Clip GPS data by time.
        Args:
            start_time (datetime.datetime | None): Start time. If None, use feeder's dep_time.
            end_time (datetime.datetime | None): End time. If None, use feeder's arr_time.
        Returns:
            Feeder: New Feeder object with clipped GPS data.
        """
        if self.gps_times is None or self.gps_points is None:
            return Feeder(self.id, self.transport_mode, np.array([]), np.array([]))
        if start_time is None:
            start_time = self.dep_time
        if end_time is None:
            end_time = self.arr_time
        idxs = np.where((self.gps_times >= start_time) & (self.gps_times <= end_time))[0]
        new_trip = Feeder(self.id, self.transport_mode, self.gps_times[idxs], self.gps_points[idxs])
        return new_trip

    def crop(self, polygon: list[tuple[float, float]] | shapely.geometry.Polygon) -> "Feeder":
        """
        Crop GPS data by polygon.
        Args:
            polygon (list[tuple[float, float]] | shapely.geometry.Polygon): Polygon to crop.
        Returns:
            Feeder: New Feeder object with cropped GPS data.
        """
        if not isinstance(polygon, shapely.geometry.Polygon):
            polygon = shapely.geometry.Polygon(polygon)

        if self.gps_points is None or self.gps_times is None:
            logger.warning("Feeder has no GPS data.")
            return Feeder(self.id, self.transport_mode, np.array([]), np.array([]))
        idxs = [i for i, p in enumerate(self.gps_points) if polygon.contains(shapely.geometry.Point(p))]
        if len(idxs) == 0:
            logger.warning("No GPS points in the polygon.")
            return Feeder(self.id, self.transport_mode, np.array([]), np.array([]))
        new_trip = Feeder(self.id, self.transport_mode, self.gps_times[idxs], self.gps_points[idxs])
        return new_trip

    def split(self, time: datetime.datetime) -> tuple["Feeder", "Feeder"]:
        """
        Split GPS data by time.
        Args:
            time (datetime.datetime): Time to split the GPS data.
        Returns:
            tuple[Feeder, Feeder]: Two new Feeder objects split by the specified time.
        Raises:
            ValueError: If time is not between dep_time and arr_time.
        """
        if self.dep_time is None or self.arr_time is None:
            raise ValueError("Feeder has no dep_time or arr_time.")
        if time < self.dep_time or time > self.arr_time:
            raise ValueError("time must be between dep_time and arr_time.")
        if self.gps_times is None or self.gps_points is None:
            raise ValueError("Feeder has no GPS data.")
        
        idx = np.where(self.gps_times <= time)[0][-1]
        trip1 = Feeder(self.id, self.transport_mode, self.gps_times[:idx], self.gps_points[:idx])
        trip2 = Feeder(self.id, self.transport_mode, self.gps_times[idx:], self.gps_points[idx:])
        return trip1, trip2

    def split_by_thresh(self, thresh: int) -> list["Feeder"]:
        """
        Split GPS data if no point is recorded in threshold (sec) .
        Args:
            thresh (int): Threshold in seconds.
        Returns:
            list[Feeder]: List of new Feeder objects with split GPS data.
        """
        if self.gps_times is None or self.gps_points is None:
            logger.warning("Feeder has no GPS data.")
            return [Feeder(self.id, self.transport_mode, np.array([]), np.array([]))]
        idxs = np.where(np.diff(self.gps_times) > np.timedelta64(thresh, "s"))[0]
        trips = []
        if len(idxs) == 0:
            trips.append(self)
        else:
            idxs = idxs + 1
            idxs = np.append(idxs, len(self.gps_times) - 1)
            idxs = np.insert(idxs, 0, 0)
            for i in range(len(idxs) - 1):
                trips.append(self.clip(self.gps_times[int(idxs[i])], self.gps_times[int(idxs[i + 1])]))
        return trips