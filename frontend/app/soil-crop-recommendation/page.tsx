"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Navigation } from "@/components/navigation";
import { motion } from "framer-motion";

export default function SoilCropRecommendationPage() {
  const [formData, setFormData] = useState({
    N: "",
    P: "",
    K: "",
    temperature: "",
    humidity: "",
    ph: "",
    rainfall: "",
  });
  const [recommendation, setRecommendation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setRecommendation(null);

    try {
      const response = await fetch("https://yamxxx1-BackendCropix.hf.space/recommend_soil_crop/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          N: parseFloat(formData.N),
          P: parseFloat(formData.P),
          K: parseFloat(formData.K),
          temperature: parseFloat(formData.temperature),
          humidity: parseFloat(formData.humidity),
          ph: parseFloat(formData.ph),
          rainfall: parseFloat(formData.rainfall),
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setRecommendation(data.recommended_crop);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-emerald-100">
      <Navigation />

      <main className="max-w-2xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-8"
        >
          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold text-gray-900 mb-4 text-balance leading-tight">
            Soil Crop Recommendation
          </h1>
          <p className="text-base sm:text-lg text-gray-700 leading-relaxed max-w-md mx-auto">
            Get recommendations for the best crop to grow based on your soil conditions.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <Card className="border-2 border-green-200 shadow-lg hover:shadow-xl transition-shadow duration-200 bg-white/80 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-xl sm:text-2xl text-green-700 font-bold">
                Enter Soil Conditions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="N" className="text-sm font-medium text-gray-700">
                      Nitrogen (N)
                    </Label>
                    <Input
                      id="N"
                      type="number"
                      name="N"
                      value={formData.N}
                      onChange={handleChange}
                      placeholder="Enter Nitrogen content"
                      className="border-green-300 focus:border-green-500 focus:ring-green-500 transition-all duration-200 hover:border-green-400"
                      required
                    />
                  </div>
                  <div>
                    <Label htmlFor="P" className="text-sm font-medium text-gray-700">
                      Phosphorus (P)
                    </Label>
                    <Input
                      id="P"
                      type="number"
                      name="P"
                      value={formData.P}
                      onChange={handleChange}
                      placeholder="Enter Phosphorus content"
                      className="border-green-300 focus:border-green-500 focus:ring-green-500 transition-all duration-200 hover:border-green-400"
                      required
                    />
                  </div>
                  <div>
                    <Label htmlFor="K" className="text-sm font-medium text-gray-700">
                      Potassium (K)
                    </Label>
                    <Input
                      id="K"
                      type="number"
                      name="K"
                      value={formData.K}
                      onChange={handleChange}
                      placeholder="Enter Potassium content"
                      className="border-green-300 focus:border-green-500 focus:ring-green-500 transition-all duration-200 hover:border-green-400"
                      required
                    />
                  </div>
                  <div>
                    <Label htmlFor="temperature" className="text-sm font-medium text-gray-700">
                      Temperature (°C)
                    </Label>
                    <Input
                      id="temperature"
                      type="number"
                      name="temperature"
                      value={formData.temperature}
                      onChange={handleChange}
                      placeholder="Enter temperature"
                      className="border-green-300 focus:border-green-500 focus:ring-green-500 transition-all duration-200 hover:border-green-400"
                      required
                    />
                  </div>
                  <div>
                    <Label htmlFor="humidity" className="text-sm font-medium text-gray-700">
                      Humidity (%)
                    </Label>
                    <Input
                      id="humidity"
                      type="number"
                      name="humidity"
                      value={formData.humidity}
                      onChange={handleChange}
                      placeholder="Enter humidity"
                      className="border-green-300 focus:border-green-500 focus:ring-green-500 transition-all duration-200 hover:border-green-400"
                      required
                    />
                  </div>
                  <div>
                    <Label htmlFor="ph" className="text-sm font-medium text-gray-700">
                      pH Value
                    </Label>
                    <Input
                      id="ph"
                      type="number"
                      name="ph"
                      value={formData.ph}
                      onChange={handleChange}
                      placeholder="Enter pH value"
                      className="border-green-300 focus:border-green-500 focus:ring-green-500 transition-all duration-200 hover:border-green-400"
                      required
                    />
                  </div>
                  <div>
                    <Label htmlFor="rainfall" className="text-sm font-medium text-gray-700">
                      Rainfall (mm)
                    </Label>
                    <Input
                      id="rainfall"
                      type="number"
                      name="rainfall"
                      value={formData.rainfall}
                      onChange={handleChange}
                      placeholder="Enter rainfall"
                      className="border-green-300 focus:border-green-500 focus:ring-green-500 transition-all duration-200 hover:border-green-400"
                      required
                    />
                  </div>
                </div>
                <Button
                  type="submit"
                  disabled={loading}
                  className="w-full bg-green-600 hover:bg-green-700 disabled:bg-green-400 text-white py-3 text-base sm:text-lg font-semibold rounded-lg shadow-md hover:shadow-lg transition-all duration-200 transform hover:scale-105 disabled:transform-none"
                >
                  {loading ? (
                    <div className="flex items-center justify-center">
                      <svg
                        className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                      >
                        <circle
                          className="opacity-25"
                          cx="12"
                          cy="12"
                          r="10"
                          stroke="currentColor"
                          strokeWidth="4"
                        ></circle>
                        <path
                          className="opacity-75"
                          fill="currentColor"
                          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                        ></path>
                      </svg>
                      Recommending...
                    </div>
                  ) : (
                    "Get Recommendation"
                  )}
                </Button>
              </form>
            </CardContent>
          </Card>
        </motion.div>

        {error && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <Card className="mt-8 border-2 border-red-300 bg-red-50 shadow-lg bg-white/80 backdrop-blur-sm">
              <CardContent className="pt-6">
                <p className="text-red-700 text-center">Error: {error}</p>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {recommendation && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <Card className="mt-8 border-2 border-green-300 bg-green-50 shadow-lg animate-in slide-in-from-bottom-4 duration-500 bg-white/80 backdrop-blur-sm">
              <CardContent className="pt-6">
                <div className="text-center">
                  <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4 animate-bounce">
                    <svg className="w-8 h-8 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                  <h2 className="text-xl sm:text-2xl font-bold text-green-800 mb-2">Recommendation:</h2>
                  <p className="text-green-700 leading-relaxed">
                    The recommended crop for your soil conditions is: <span className="font-semibold">{recommendation}</span>
                  </p>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </main>
    </div>
  );
}