"use client";

import { useState } from "react";

export default function FertilizerRecommendationPage() {
  const [formData, setFormData] = useState({
    Crop: "",
    Current_N: "",
    Current_P: "",
    Current_K: "",
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
      const response = await fetch("https://yamxxx1-BackendCropix.hf.space/recommend_fertilizer/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          Crop: formData.Crop,
          Current_N: parseFloat(formData.Current_N),
          Current_P: parseFloat(formData.Current_P),
          Current_K: parseFloat(formData.Current_K),
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setRecommendation(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold mb-6">Fertilizer Recommendation</h1>

      <form onSubmit={handleSubmit} className="bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="Crop">
              Crop Type
            </label>
            <input
              className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
              id="Crop"
              type="text"
              name="Crop"
              value={formData.Crop}
              onChange={handleChange}
              placeholder="e.g., Wheat, Rice"
              required
            />
          </div>
          <div>
            <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="Current_N">
              Current Nitrogen (N) in Soil
            </label>
            <input
              className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
              id="Current_N"
              type="number"
              name="Current_N"
              value={formData.Current_N}
              onChange={handleChange}
              placeholder="Enter current Nitrogen content"
              required
            />
          </div>
          <div>
            <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="Current_P">
              Current Phosphorus (P) in Soil
            </label>
            <input
              className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
              id="Current_P"
              type="number"
              name="Current_P"
              value={formData.Current_P}
              onChange={handleChange}
              placeholder="Enter current Phosphorus content"
              required
            />
          </div>
          <div>
            <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="Current_K">
              Current Potassium (K) in Soil
            </label>
            <input
              className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
              id="Current_K"
              type="number"
              name="Current_K"
              value={formData.Current_K}
              onChange={handleChange}
              placeholder="Enter current Potassium content"
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
            {loading ? "Recommending..." : "Get Recommendation"}
          </button>
        </div>
      </form>

      {loading && <p className="text-blue-500">Loading...</p>}
      {error && <p className="text-red-500">Error: {error}</p>}

      {recommendation && (
        <div className="mt-6 p-4 bg-green-100 border border-green-400 text-green-700 rounded">
          <h2 className="text-xl font-bold mb-2">Recommended Fertilizer:</h2>
          <p>Nitrogen (N): <span className="font-semibold">{recommendation.recommended_N}</span></p>
          <p>Phosphorus (P): <span className="font-semibold">{recommendation.recommended_P}</span></p>
          <p>Potassium (K): <span className="font-semibold">{recommendation.recommended_K}</span></p>
        </div>
      )}
    </div>
  );
}