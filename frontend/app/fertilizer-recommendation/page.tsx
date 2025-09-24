"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Navigation } from "@/components/navigation";
import dynamic from 'next/dynamic';
import { motion } from "framer-motion"; // Import motion from framer-motion

const MotionDiv = dynamic(() => import('framer-motion').then(mod => mod.motion.div), { ssr: false });
const MotionH1 = dynamic(() => import('framer-motion').then(mod => mod.motion.h1), { ssr: false });
const MotionP = dynamic(() => import('framer-motion').then(mod => mod.motion.p), { ssr: false });

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
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-emerald-100 dark:from-gray-900 dark:to-gray-800">
      <Navigation />
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12">
        <MotionH1
          className="text-3xl sm:text-4xl font-bold text-foreground mb-4 sm:mb-6 text-center"
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          Fertilizer Recommendation
        </MotionH1>
        <MotionP
          className="text-lg sm:text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed text-center mb-8 sm:mb-12"
          initial={{ opacity: 0, y: -30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          Get precise fertilizer recommendations to optimize your crop yield and soil health.
        </MotionP>

        <MotionDiv
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
        >
          <Card className="border-2 border-border hover:shadow-lg transition-shadow duration-200">
            <CardHeader>
              <CardTitle className="text-lg sm:text-xl text-primary">Enter Soil and Crop Details</CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="Crop">Crop Type</Label>
                    <Input
                      id="Crop"
                      type="text"
                      name="Crop"
                      value={formData.Crop}
                      onChange={handleChange}
                      placeholder="e.g., Wheat, Rice"
                      required
                      className="border-border focus:border-primary focus:ring-primary transition-colors"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="Current_N">Current Nitrogen (N) in Soil</Label>
                    <Input
                      id="Current_N"
                      type="number"
                      name="Current_N"
                      value={formData.Current_N}
                      onChange={handleChange}
                      placeholder="Enter current Nitrogen content"
                      required
                      className="border-border focus:border-primary focus:ring-primary transition-colors"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="Current_P">Current Phosphorus (P) in Soil</Label>
                    <Input
                      id="Current_P"
                      type="number"
                      name="Current_P"
                      value={formData.Current_P}
                      onChange={handleChange}
                      placeholder="Enter current Phosphorus content"
                      required
                      className="border-border focus:border-primary focus:ring-primary transition-colors"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="Current_K">Current Potassium (K) in Soil</Label>
                    <Input
                      id="Current_K"
                      type="number"
                      name="Current_K"
                      value={formData.Current_K}
                      onChange={handleChange}
                      placeholder="Enter current Potassium content"
                      required
                      className="border-border focus:border-primary focus:ring-primary transition-colors"
                    />
                  </div>
                </div>
                <Button
                  type="submit"
                  disabled={loading}
                  className="w-full bg-primary hover:bg-primary/90 text-primary-foreground py-2 px-4 rounded-md text-lg font-semibold transition-colors"
                >
                  {loading ? "Recommending..." : "Get Recommendation"}
                </Button>
              </form>
            </CardContent>
          </Card>
        </MotionDiv>

        {loading && <MotionP className="text-blue-500 text-center mt-4">Loading...</MotionP>}
        {error && <MotionP className="text-red-500 text-center mt-4">Error: {error}</MotionP>}

        {recommendation && (
          <MotionDiv
            className="mt-8 p-6 bg-green-100 border border-green-400 text-green-700 rounded-lg shadow-md"
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
          >
            <h2 className="text-2xl font-bold mb-4">Recommended Fertilizer:</h2>
            <p className="text-lg">Nitrogen (N): <span className="font-semibold">{recommendation.recommended_N}</span></p>
            <p className="text-lg">Phosphorus (P): <span className="font-semibold">{recommendation.recommended_P}</span></p>
            <p className="text-lg">Potassium (K): <span className="font-semibold">{recommendation.recommended_K}</span></p>
          </MotionDiv>
        )}
      </main>
    </div>
  );
}