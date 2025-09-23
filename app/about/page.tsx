"use client"

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Navigation } from "@/components/navigation";
import dynamic from 'next/dynamic';

const MotionDiv = dynamic(() => import('framer-motion').then(mod => mod.motion.div), { ssr: false });
const MotionH1 = dynamic(() => import('framer-motion').then(mod => mod.motion.h1), { ssr: false });

import { Github, Linkedin, Instagram } from "lucide-react";

const teamMembers = [
  {
    name: "Rachit Tiwari",
    role: "Lead Developer",
    bio: "Passionate about AI and its applications in agriculture. Rachit leads the development of Cropix's core features.",
    social: {
      github: "https://github.com/rachittiwari",
      linkedin: "https://www.linkedin.com/in/rachittiwari",
      instagram: "https://www.instagram.com/rachittiwari",
    },
  },
  {
    name: "Shaikh Mohammad Warsi",
    role: "AI/ML Engineer",
    bio: "Specializing in machine learning models for disease detection and market forecasting. Shaikh ensures our AI is accurate and efficient.",
    social: {
      github: "https://github.com:shaikhwarsi",
      linkedin: "https://www.linkedin.com/in/shaikhwarsi",
      instagram: "https://www.instagram.com/shaikhwarsi",
    },
  },
  {
    name: "Aaryan Maurya",
    role: "Data Scientist",
    bio: "Analyzing vast datasets to extract valuable insights for crop recommendation and optimization. Aaryan's work drives our data-driven decisions.",
    social: {
      github: "https://github.com/aaryanmaurya",
      linkedin: "https://www.linkedin.com/in/aaryanmaurya",
      instagram: "https://www.instagram.com/aaryanmaurya",
    },
  },
  {
    name: "Sanidhya",
    role: "CNN and LSTM Specialist",
    bio: "Focusing on advanced deep learning techniques for image recognition and sequence prediction in agricultural data.",
    social: {
      github: "#",
      linkedin: "#",
      instagram: "#",
    },
  },
  {
    name: "Suryansh",
    role: "Research Paper and PPT Specialist",
    bio: "Dedicated to compiling comprehensive research papers and creating impactful presentations to showcase Cropix's innovations.",
    social: {
      github: "#",
      linkedin: "#",
      instagram: "#",
    },
  },
];

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-card to-background">
      <Navigation />

      <main className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 sm:py-24 lg:py-32">
        <MotionDiv
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, ease: "easeOut" }}
          className="text-center"
        >
          <MotionH1
            className="text-4xl sm:text-5xl md:text-6xl font-bold text-foreground mb-6"
            initial={{ opacity: 0, y: -50, scale: 0.8 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ duration: 1.2, ease: "easeOut", delay: 0.5 }}
          >
            About <span className="text-primary">Cropix</span>
          </MotionH1>
          <p className="text-xl sm:text-2xl text-muted-foreground mb-12 max-w-3xl mx-auto">
            Empowering farmers with AI-driven insights for smarter, more sustainable agriculture.
          </p>
        </MotionDiv>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {teamMembers.map((member, index) => (
            <MotionDiv
              key={index}
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 * index, ease: "easeOut" }}
              className="bg-card p-6 rounded-lg shadow-lg text-center"
            >
              <h2 className="text-2xl font-bold text-foreground mb-2">{member.name}</h2>
              <h3 className="text-primary text-lg mb-4">{member.role}</h3>
              <p className="text-muted-foreground mb-4">{member.bio}</p>
              <div className="flex justify-center gap-4">
                {member.social.github && (
                  <Link href={member.social.github} target="_blank" rel="noopener noreferrer">
                    <Button variant="ghost" size="icon">
                      <Github className="w-6 h-6 text-foreground hover:text-primary" />
                    </Button>
                  </Link>
                )}
                {member.social.linkedin && (
                  <Link href={member.social.linkedin} target="_blank" rel="noopener noreferrer">
                    <Button variant="ghost" size="icon">
                      <Linkedin className="w-6 h-6 text-foreground hover:text-primary" />
                    </Button>
                  </Link>
                )}
                {member.social.instagram && (
                  <Link href={member.social.instagram} target="_blank" rel="noopener noreferrer">
                    <Button variant="ghost" size="icon">
                      <Instagram className="w-6 h-6 text-foreground hover:text-primary" />
                    </Button>
                  </Link>
                )}
              </div>
            </MotionDiv>
          ))}
        </div>
      </main>
    </div>
  );
}
