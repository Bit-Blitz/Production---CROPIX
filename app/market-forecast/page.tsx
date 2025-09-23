"use client"

import { useState } from "react"
import { Navigation } from "@/components/navigation"
import { motion } from "framer-motion"
import { TrendingUp, CalendarDays } from "lucide-react"

export default function MarketForecastPage() {
  const [cropName, setCropName] = useState("")
  const [weeksToForecast, setWeeksToForecast] = useState("4")
  const [forecastData, setForecastData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const cropOptions = ["Wheat", "Rice", "Corn", "Soybeans"]

  const handleChange = (e) => {
    if (e.target.name === "cropName") {
      setCropName(e.target.value)
    } else if (e.target.name === "weeksToForecast") {
      setWeeksToForecast(e.target.value)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setForecastData(null)

    try {
      const response = await fetch("https://yamxxx1-BackendCropix.hf.space/forecast_market_prices/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ crop_name: cropName, weeks_to_forecast: parseInt(weeksToForecast) }),
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
          <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-4">Market Price Forecast</h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Predict future market prices for various crops to optimize your selling strategy.
          </p>
        </motion.div>

        <form onSubmit={handleSubmit} className="bg-white shadow-md rounded px-8 pt-6 pb-8 mb-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="cropName">
                Crop Name
              </label>
              <select
                className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                id="cropName"
                name="cropName"
                value={cropName}
                onChange={handleChange}
                required
              >
                <option value="">Select a crop</option>
                {cropOptions.map((crop) => (
                  <option key={crop} value={crop}>
                    {crop}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="weeksToForecast">
                Weeks to Forecast
              </label>
              <input
                className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                id="weeksToForecast"
                type="number"
                name="weeksToForecast"
                value={weeksToForecast}
                onChange={handleChange}
                min="1"
                max="52"
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
              {loading ? "Forecasting..." : "Get Market Forecast"}
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
                <TrendingUp className="w-8 h-8 text-green-600 mr-3" />
                <h2 className="text-2xl font-bold text-gray-900">Forecast for {cropName}</h2>
              </div>

              <div className="space-y-4">
                {Object.entries(forecastData.forecast[cropName]).map(([date, price]) => (
                  <div key={date} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg shadow-sm">
                    <h3 className="font-semibold text-gray-800 flex items-center">
                      <CalendarDays className="w-5 h-5 text-purple-500 mr-2" />
                      {new Date(date).toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" })}
                    </h3>
                    <p className="text-lg font-bold text-green-600">${price.toFixed(2)}</p>
                  </div>
                ))}
              </div>

              <div className="mt-4 p-4 bg-green-50 rounded-lg">
                <p className="text-sm text-green-800">
                  <strong>Market Insight:</strong> This forecast helps you anticipate price movements and plan your sales.
                </p>
              </div>
            </motion.div>
          </div>
        )}
      </main>
    </div>
  )
}
