import React, { useState } from "react";
import "./App.css";

function App() {
  const [numRiders, setNumRiders] = useState(50);
  const [numDrivers, setNumDrivers] = useState(25);
  const [historicalCost, setHistoricalCost] = useState(149.99);
  const [competitorPrice, setCompetitorPrice] = useState(114);
  const [locationCategory, setLocationCategory] = useState("Urban");
  const [timeOfBooking, setTimeOfBooking] = useState("Evening");
  const [vehicleType, setVehicleType] = useState("Premium");
  const [loyaltyStatus, setLoyaltyStatus] = useState("Gold");
  const [pastRides, setPastRides] = useState(45);
  const [avgRatings, setAvgRatings] = useState(4.5);
  const [rideDuration, setRideDuration] = useState(60);
  const [predictedPrice, setPredictedPrice] = useState(null);

  const calculatePrice = () => {
    let price = historicalCost;
    const demandSupplyRatio = numRiders / (numDrivers || 1);
    price *= 1 + (demandSupplyRatio - 1) * 0.1;
    price = (price + competitorPrice * 1.05) / 2;

    if (vehicleType === "Premium") price *= 1.2;
    else if (vehicleType === "Economy") price *= 0.9;

    if (loyaltyStatus === "Gold") price *= 0.9;
    else if (loyaltyStatus === "Silver") price *= 0.95;

    if (locationCategory === "Urban") price *= 1.05;
    else if (locationCategory === "Suburban") price *= 1.02;
    else price *= 0.98;

    if (timeOfBooking === "Morning") price *= 1.03;
    else if (timeOfBooking === "Evening") price *= 1.06;
    else price *= 0.97;

    price += pastRides * 0.3;
    price += avgRatings * 2;
    price += rideDuration * 0.5;

    if (price < historicalCost * 1.1) price = historicalCost * 1.1;
    setPredictedPrice(price);
  };

  return (
    <div className="app-container">
      <h1 className="main-title">Dynamic Pricing</h1>

      <div className="content-wrapper">
        {/* Left Side - Ride Details */}
        <div className="form-container">
          <h2>Ride Details</h2>
          <div className="form-grid">
            <div className="form-group">
              <label>Number of Riders</label>
              <input
                type="number"
                value={numRiders}
                onChange={(e) => setNumRiders(Number(e.target.value))}
              />

              <label>Historical Cost ($)</label>
              <input
                type="number"
                step="0.01"
                value={historicalCost}
                onChange={(e) => setHistoricalCost(Number(e.target.value))}
              />

              <label>Location Category</label>
              <select
                value={locationCategory}
                onChange={(e) => setLocationCategory(e.target.value)}
              >
                <option>Urban</option>
                <option>Suburban</option>
                <option>Rural</option>
              </select>

              <label>Vehicle Type</label>
              <select
                value={vehicleType}
                onChange={(e) => setVehicleType(e.target.value)}
              >
                <option>Economy</option>
                <option>Premium</option>
              </select>

              <label>Past Rides</label>
              <input
                type="number"
                value={pastRides}
                onChange={(e) => setPastRides(Number(e.target.value))}
              />
            </div>

            <div className="form-group">
              <label>Number of Drivers</label>
              <input
                type="number"
                value={numDrivers}
                onChange={(e) => setNumDrivers(Number(e.target.value))}
              />

              <label>Competitor Price ($)</label>
              <input
                type="number"
                step="0.01"
                value={competitorPrice}
                onChange={(e) => setCompetitorPrice(Number(e.target.value))}
              />

              <label>Time of Booking</label>
              <select
                value={timeOfBooking}
                onChange={(e) => setTimeOfBooking(e.target.value)}
              >
                <option>Morning</option>
                <option>Afternoon</option>
                <option>Evening</option>
                <option>Night</option>
              </select>

              <label>Loyalty Status</label>
              <select
                value={loyaltyStatus}
                onChange={(e) => setLoyaltyStatus(e.target.value)}
              >
                <option>Regular</option>
                <option>Silver</option>
                <option>Gold</option>
              </select>

              <label>Average Ratings</label>
              <input
                type="number"
                step="0.1"
                value={avgRatings}
                onChange={(e) => setAvgRatings(Number(e.target.value))}
              />
            </div>
          </div>

          <label>Expected Ride Duration (minutes)</label>
          <input
            type="number"
            value={rideDuration}
            onChange={(e) => setRideDuration(Number(e.target.value))}
          />

          <button onClick={calculatePrice}>Get Optimal Price</button>
        </div>

        {/* Right Side - Result */}
        <div className="result-container">
          <h2>Recommended Price</h2>
          {predictedPrice ? (
            <div className="result-box">
              <h3>Optimal Price:</h3>
              <p className="price">₹{predictedPrice.toFixed(2)}</p>
            </div>
          ) : (
            <p className="placeholder-text">
              Fill out ride details and click <strong>Get Optimal Price</strong>.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
