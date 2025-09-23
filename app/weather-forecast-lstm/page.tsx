"use client"

import { useState } from "react"
import { Navigation } from "@/components/navigation"
import { motion } from "framer-motion"
import { Cloud, Thermometer, Droplets, CalendarDays } from "lucide-react"

export default function LSTMWeatherForecastPage() {
  const [location, setLocation] = useState("")
  const [days, setDays] = useState("7")
  const [forecastData, setForecastData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleChange = (e) => {
    if (e.target.name === "location") {
      setLocation(e.target.value)
    } else if (e.target.name === "days") {
      setDays(e.target.value)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setForecastData(null)

    try {
      const response = await fetch("https://yamxxx1-BackendCropix.hf.space/weather_forecast_lstm/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ city: location, days: parseInt(days) }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      if (data.error) {
        setError(data.error)
      } else {
        setForecastData(data)
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-white">
      <Navigation />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-4">LSTM Weather Forecast</h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Get long-term weather predictions powered by LSTM models for strategic farming decisions.
          </p>
        </motion.div>

        <form onSubmit={handleSubmit} className="bg-white shadow-md rounded px-8 pt-6 pb-8 mb-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="location">
                Location (City)
              </label>
              <input
                className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                id="location"
                type="text"
                name="location"
                value={location}
                onChange={handleChange}
                placeholder="Enter city name (e.g., London)"
                required
              />
            </div>
            <div>
              <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="days">
                Days to Forecast
              </label>
              <input
                className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                id="days"
                type="number"
                name="days"
                value={days}
                onChange={handleChange}
                min="1"
                max="30"
                required
              />
            </div>
          </div>
          <div className="flex items-center justify-between">
            <button
              className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
              type="submit"
              disabled={loading}
            >
              {loading ? "Forecasting..." : "Get LSTM Forecast"}
            </button>
          </div>
        </form>

        {loading && <p className="text-blue-500">Loading...</p>}
        {error && <p className="text-red-500">Error: {error}</p>}

        {forecastData && forecastData.forecast && ( // Check if forecastData and forecastData.forecast exist
          <div className="grid grid-cols-1 gap-8">
            <motion.div
              className="bg-white rounded-2xl shadow-lg border-2 border-green-200 p-6"
              initial={{ opacity: 0, x: 30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.4 }}
            >
              <div className="flex items-center mb-6">
                <CalendarDays className="w-8 h-8 text-green-600 mr-3" />
                <h2 className="text-2xl font-bold text-gray-900">LSTM Forecast for {forecastData.city}</h2>
              </div>

              <div className="space-y-4">
                {forecastData.forecast.map((dayForecast, index) => (
                  <div key={index} className="bg-gray-50 p-4 rounded-lg shadow-sm">
                    <h3 className="text-lg font-semibold text-gray-800 mb-2">Day {dayForecast.day}</h3>
                    <div className="grid grid-cols-2 gap-2">
                      {Object.entries(dayForecast).map(([key, value]) => {
                        if (key === "day") return null; // Skip displaying the day number again
                        return (
                          <div key={key} className="flex items-center">
                            {key === "temperature" && <Thermometer className="w-5 h-5 text-blue-500 mr-2" />}
                            {key === "humidity" && <Droplets className="w-5 h-5 text-green-500 mr-2" />}
                            {key === "rainfall" && <Cloud className="w-5 h-5 text-indigo-500 mr-2" />}
                            <p className="text-sm text-gray-700">
                              <span className="font-medium capitalize">{key.replace(/_/g, " ")}:</span> {typeof value === 'number' ? value.toFixed(2) : value}
                            </p>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-4 p-4 bg-green-50 rounded-lg">
                <p className="text-sm text-green-800">
                  <strong>Farming Tip:</strong> Use this long-term forecast to plan planting and harvesting schedules.
                </p>
              </div>
            </motion.div>
          </div>
        )}
      </main>
    </div>
  )
}