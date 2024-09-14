from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, Enum as SQLAlchemyEnum
# from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from typing import List, Optional
from datetime import date
from enum import Enum
import requests
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import time
from urllib.parse import quote_plus

from dotenv import dotenv_values

secrets = dotenv_values("../../secrets.env")

# Database setup
password = quote_plus(secrets["POSTGRESQL_PASSWORD"])
SQLALCHEMY_DATABASE_URL = f"postgresql://postgres:{password}@172.23.16.1/logistic_op"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

app = FastAPI()

# Enum definitions
class GoodsType(str, Enum):
    DURABLE = "durable"
    FRAGILE = "fragile"
    COOL_CHAIN = "cool_chain"

class VehicleType(str, Enum):
    TRUCK = "truck"
    VAN = "van"
    REFRIGERATED = "refrigerated"

# SQLAlchemy models
class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    goods = Column(String)
    goods_type = Column(SQLAlchemyEnum(GoodsType))
    expected_delivery_date = Column(Date)
    quoted_cost = Column(Float)
    source_location = Column(String)
    destination_location = Column(String)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

class Transporter(Base):
    __tablename__ = "transporters"

    id = Column(Integer, primary_key=True, index=True)
    vehicle_type = Column(SQLAlchemyEnum(VehicleType))
    vehicle_size = Column(String)
    capacity = Column(Float)
    availability = Column(Date)
    goods_types = Column(String)  # Stored as comma-separated values
    full_load_cost = Column(Float)
    partial_load_cost = Column(Float)
    location = Column(String)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

# Pydantic models
class CustomerOrderCreate(BaseModel):
    goods: str
    goods_type: GoodsType
    expected_delivery_date: date
    quoted_cost: float
    source_location: str
    destination_location: str

class TransporterCreate(BaseModel):
    vehicle_type: VehicleType
    vehicle_size: str
    capacity: float
    availability: date
    goods_types: List[GoodsType]
    full_load_cost: float
    partial_load_cost: float
    location: str

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Geocoding function with error handling and retries
def geocode(address, max_retries=3):
    for attempt in range(max_retries):
        try:
            url = f"https://nominatim.openstreetmap.org/search?format=json&q={address}"
            response = requests.get(url, headers={'User-Agent': 'YourAppName/1.0'})
            response.raise_for_status()  # Raise an exception for bad status codes
            data = response.json()
            if data:
                return float(data[0]["lat"]), float(data[0]["lon"])
            else:
                print(f"No geocoding results found for address: {address}")
                return None, None
        except requests.exceptions.RequestException as e:
            print(f"Request failed on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                print(f"Max retries reached. Geocoding failed for address: {address}")
                return None, None
        except (KeyError, IndexError, ValueError) as e:
            print(f"Error parsing geocoding response: {e}")
            return None, None
        time.sleep(1)  # Wait for 1 second before retrying to avoid rate limiting

# Distance calculation function with error handling
def calculate_distance(lat1, lon1, lat2, lon2):
    try:
        url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        if data["code"] == "Ok":
            print(data)
            return data["routes"][0]["distance"]
        else:
            print(f"OSRM API returned non-Ok status: {data['code']}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error calculating distance: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"Error parsing distance calculation response: {e}")
        return None

@app.post("/order")
async def create_order(order: CustomerOrderCreate, db: Session = Depends(get_db)):
    db_order = Order(**order.model_dump())
    db_order.latitude, db_order.longitude = geocode(order.destination_location)
    if db_order.latitude is None or db_order.longitude is None:
        raise HTTPException(status_code=400, detail="Failed to geocode destination location")
    db.add(db_order)
    db.commit()
    db.refresh(db_order)
    return {"message": "Order created successfully", "order_id": db_order.id}

@app.post("/transporter")
async def register_transporter(transporter: TransporterCreate, db: Session = Depends(get_db)):
    db_transporter = Transporter(**transporter.dict(exclude={'goods_types'}))
    db_transporter.goods_types = ",".join(transporter.goods_types)
    db_transporter.latitude, db_transporter.longitude = geocode(transporter.location)
    if db_transporter.latitude is None or db_transporter.longitude is None:
        raise HTTPException(status_code=400, detail="Failed to geocode transporter location")
    db.add(db_transporter)
    db.commit()
    db.refresh(db_transporter)
    return {"message": "Transporter registered successfully", "transporter_id": db_transporter.id}

@app.get("/optimize")
async def optimize_orders(db: Session = Depends(get_db)):
    orders = db.query(Order).all()
    transporters = db.query(Transporter).all()

    if not orders or not transporters:
        raise HTTPException(status_code=400, detail="No orders or transporters available")

    optimized_routes = []

    for transporter in transporters:
        transporter_goods_types = transporter.goods_types.split(",")
        compatible_orders = [
            order for order in orders
            if order.goods_type.value in transporter_goods_types
            and order.expected_delivery_date >= transporter.availability
        ]

        if not compatible_orders:
            continue

        # Create distance matrix
        num_locations = len(compatible_orders) + 1  # +1 for the transporter's location
        distance_matrix = np.zeros((num_locations, num_locations))

        for i in range(num_locations):
            for j in range(i + 1, num_locations):
                if i == 0:
                    lat1, lon1 = transporter.latitude, transporter.longitude
                else:
                    lat1, lon1 = compatible_orders[i - 1].latitude, compatible_orders[i - 1].longitude

                if j == 0:
                    lat2, lon2 = transporter.latitude, transporter.longitude
                else:
                    lat2, lon2 = compatible_orders[j - 1].latitude, compatible_orders[j - 1].longitude

                distance = calculate_distance(lat1, lon1, lat2, lon2)
                if distance is None:
                    # If distance calculation fails, use a large value to discourage this route
                    distance = 1e9
                distance_matrix[i][j] = distance_matrix[j][i] = distance

        # Create data model
        data = {}
        data['distance_matrix'] = distance_matrix.tolist()
        data['num_vehicles'] = 1
        data['depot'] = 0

        # Create routing index manager
        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                               data['num_vehicles'], data['depot'])

        # Create Routing Model
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(data['distance_matrix'][from_node][to_node])

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Capacity constraint
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return 1 if from_node != 0 else 0  # Assuming each order has a demand of 1

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            [int(transporter.capacity)],  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity')

        # Setting first solution heuristic
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.time_limit.seconds = 10  # Set a time limit for the solver

        # Solve the problem
        try:
            solution = routing.SolveWithParameters(search_parameters)
        except Exception as e:
            print(f"Error solving routing problem for transporter {transporter.id}: {str(e)}")
            continue

        if solution:
            route_distance = 0
            route_orders = []
            index = routing.Start(0)
            while not routing.IsEnd(index):
                if manager.IndexToNode(index) != 0:  # Skip the depot
                    route_orders.append(compatible_orders[manager.IndexToNode(index) - 1])
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)

            optimized_routes.append({
                "transporter": transporter,
                "orders": route_orders,
                "total_distance": route_distance,
                "total_cost": transporter.full_load_cost if len(route_orders) == transporter.capacity else transporter.partial_load_cost
            })
        else:
            print(f"No solution found for transporter {transporter.id}")

    return optimized_routes

# Create tables
Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)