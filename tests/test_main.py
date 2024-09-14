import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import date, timedelta
from urllib.parse import quote_plus

from src.logistics_op.main import app, Base, get_db, Order, Transporter, geocode, calculate_distance

# Use an in-memory SQLite database for testing
password = quote_plus('Arnika@1996')
SQLALCHEMY_DATABASE_URL = f"postgresql://postgres:{password}@172.23.16.1/logistic_op"
# engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
engine = create_engine(SQLALCHEMY_DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

def test_create_order(db_session):
    response = client.post(
        "/order",
        json={
            "goods": "Test Goods",
            "goods_type": "durable",
            "expected_delivery_date": "2024-08-30",
            "quoted_cost": 100.0,
            "source_location": "New York, NY",
            "destination_location": "Los Angeles, CA"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "order_id" in data
    assert data["message"] == "Order created successfully"

def test_register_transporter(db_session):
    response = client.post(
        "/transporter",
        json={
            "vehicle_type": "truck",
            "vehicle_size": "Large",
            "capacity": 10.0,
            "availability": "2024-08-24",
            "goods_types": ["durable"],
            "full_load_cost": 1000.0,
            "partial_load_cost": 500.0,
            "location": "New York, NY"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "transporter_id" in data
    assert data["message"] == "Transporter registered successfully"

def test_optimize_orders(db_session):
    # First, add some test data
    test_create_order(db_session)
    test_register_transporter(db_session)

    response = client.get("/optimize")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if len(data) > 0:
        assert "transporter" in data[0]
        assert "orders" in data[0]
        assert "total_distance" in data[0]
        assert "total_cost" in data[0]

def test_geocode():
    lat, lon = geocode("New York, NY")
    assert lat is not None
    assert lon is not None
    assert 40 < lat < 41  # Approximate latitude for New York
    assert -74.5 < lon < -73.5  # Approximate longitude for New York

def test_calculate_distance():
    distance = calculate_distance(40.7128, -74.0060, 34.0522, -118.2437)  # New York to Los Angeles
    assert distance is not None
    assert 3900000 < distance < 4000000  # Approximate distance in meters

def test_invalid_order(db_session):
    response = client.post(
        "/order",
        json={
            "goods": "Test Goods",
            "goods_type": "INVALID_TYPE",  # Invalid goods type
            "expected_delivery_date": str(date.today() + timedelta(days=7)),
            "quoted_cost": 100.0,
            "source_location": "New York, NY",
            "destination_location": "Los Angeles, CA"
        }
    )
    assert response.status_code == 422  # Unprocessable Entity

def test_invalid_transporter(db_session):
    response = client.post(
        "/transporter",
        json={
            "vehicle_type": "INVALID_TYPE",  # Invalid vehicle type
            "vehicle_size": "Large",
            "capacity": 10.0,
            "availability": str(date.today()),
            "goods_types": ["DURABLE", "FRAGILE"],
            "full_load_cost": 1000.0,
            "partial_load_cost": 500.0,
            "location": "Chicago, IL"
        }
    )
    assert response.status_code == 422  # Unprocessable Entity

def test_optimize_no_data(db_session):
    # Clear the database
    db_session.query(Order).delete()
    db_session.query(Transporter).delete()
    db_session.commit()

    response = client.get("/optimize")
    assert response.status_code == 400
    assert response.json()["detail"] == "No orders or transporters available"

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])