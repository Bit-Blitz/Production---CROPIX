'use client';

import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Navigation } from "@/components/navigation"
import React, { useState } from 'react';

export default function ContactPage() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: '',
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { id, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [id]: value,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    const webhookUrl = 'https://discord.com/api/webhooks/1419957815279091728/JwqE_9XWVITTIst5Kn9akmkRHNO2EDMmWoOZdAUecl1TVCsJGfQ6jTHUaFwYXtWhS5Y8';

    const payload = {
      embeds: [
        {
          title: `New Contact Form Submission from ${formData.name}`,
          color: 0x0099ff, // Blue color
          fields: [
            { name: 'Name', value: formData.name, inline: true },
            { name: 'Email', value: formData.email, inline: true },
            { name: 'Subject', value: formData.subject },
            { name: 'Message', value: formData.message },
          ],
          timestamp: new Date().toISOString(),
        },
      ],
    };

    try {
      const response = await fetch(webhookUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (response.ok) {
        alert('Thank you for your message! We will get back to you soon.');
        setFormData({
          name: '',
          email: '',
          subject: '',
          message: '',
        });
      } else {
        alert('There was an error sending your message. Please try again later.');
        console.error('Webhook failed to send:', response.status, response.statusText);
      }
    } catch (error) {
      alert('There was an error sending your message. Please try again later.');
      console.error('Error sending webhook:', error);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation />

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12">
        <div className="text-center mb-8 sm:mb-12">
          <h1 className="text-3xl sm:text-4xl font-bold text-foreground mb-4 sm:mb-6 text-balance">Contact Us</h1>
          <p className="text-lg sm:text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
            Have questions about Cropix? We'd love to hear from you. Send us a message and we'll respond as soon as
            possible.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 sm:gap-12">
          {/* Contact Form */}
          <Card className="border-2 border-border hover:shadow-lg transition-shadow duration-200">
            <CardHeader>
              <CardTitle className="text-lg sm:text-xl text-primary">Send us a message</CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="space-y-2">
                  <Label htmlFor="name">Name</Label>
                  <Input
                    id="name"
                    value={formData.name}
                    onChange={handleChange}
                    className="border-border focus:border-primary focus:ring-primary transition-colors"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="email">Email</Label>
                  <Input
                    id="email"
                    type="email"
                    value={formData.email}
                    onChange={handleChange}
                    className="border-border focus:border-primary focus:ring-primary transition-colors"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="subject">Subject</Label>
                  <Input
                    id="subject"
                    value={formData.subject}
                    onChange={handleChange}
                    className="border-border focus:border-primary focus:ring-primary transition-colors"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="message">Message</Label>
                  <Textarea
                    id="message"
                    rows={5}
                    value={formData.message}
                    onChange={handleChange}
                    className="border-border focus:border-primary focus:ring-primary transition-colors"
                  />
                </div>

                <Button type="submit" className="w-full bg-primary hover:bg-primary/90 text-white transition-all duration-200 transform hover:scale-105 shadow-md hover:shadow-lg">
                  Send Message
                </Button>
              </form>
            </CardContent>
          </Card>

          {/* Contact Information */}
          <div className="space-y-6 sm:space-y-8">
            <Card className="border-border hover:shadow-md transition-shadow duration-200">
              <CardContent className="pt-6">
                <div className="flex items-start space-x-4">
                  <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center flex-shrink-0">
                    <svg className="w-5 h-5 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M3 8l7.89 4.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                      />
                    </svg>
                  </div>
                  <div>
                    <h3 className="font-semibold text-foreground">Email</h3>
                    <p className="text-muted-foreground">support@cropix.com</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-border hover:shadow-md transition-shadow duration-200">
              <CardContent className="pt-6">
                <div className="flex items-start space-x-4">
                  <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center flex-shrink-0">
                    <svg className="w-5 h-5 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z"
                      />
                    </svg>
                  </div>
                  <div>
                    <h3 className="font-semibold text-foreground">Phone</h3>
                    <p className="text-muted-foreground">+1 (555) 123-4567</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-border hover:shadow-md transition-shadow duration-200">
              <CardContent className="pt-6">
                <div className="flex items-start space-x-4">
                  <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center flex-shrink-0">
                    <svg className="w-5 h-5 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"
                      />
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"
                      />
                    </svg>
                  </div>
                  <div>
                    <h3 className="font-semibold text-foreground">Office</h3>
                    <p className="text-muted-foreground leading-relaxed">
                      123 Agriculture Ave
                      <br />
                      Farm City, FC 12345
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <div className="bg-card p-6 rounded-xl border border-border">
              <h3 className="font-semibold text-foreground mb-2">Quick Start</h3>
              <p className="text-muted-foreground mb-4 leading-relaxed">
                Ready to get crop recommendations? Try our tool now!
              </p>
              <Link href="/crop-recommendation">
                <Button
                  size="lg"
                  className="bg-primary hover:bg-primary/90 text-primary-foreground px-10 py-6 text-xl font-semibold rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 relative overflow-hidden"
                >
                  Get Started
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
