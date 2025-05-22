import "./globals.css"
import { Inter } from 'next/font/google'
import type { Metadata } from "next"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "Mental Health Counselor Chatbot",
  description: "A chatbot UI for mental health counseling with EEG visualization",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} dark:bg-background dark:text-foreground min-h-screen`}>
        {children}
      </body>
    </html>
  )
} 